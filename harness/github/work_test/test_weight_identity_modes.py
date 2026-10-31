"""Strict checkpoint hashes and the in-process lease. CPU only."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from saebase.store import CheckpointLeaseError, VerifiedCheckpointSession, sha256_state_dict
from saebase.store.identity import _hash_items, _state_items


class Box(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(4))


def test_strict_hash_matches_independent_bytes_and_ignores_nothing():
    model = Box()
    assert sha256_state_dict(model) == _hash_items(_state_items(model))
    assert sha256_state_dict(model) == sha256_state_dict(model)


def test_data_edit_changes_strict_hash_without_touching_version():
    model = Box()
    first = sha256_state_dict(model)
    version = int(model.weight._version)
    with torch.no_grad():
        model.weight.data.add_(1)
    assert int(model.weight._version) == version
    assert sha256_state_dict(model) != first


def test_inplace_copy_load_replace_and_move_change_strict_hash():
    model = Box()
    first = sha256_state_dict(model)
    with torch.no_grad():
        model.weight.add_(1)
    added = sha256_state_dict(model)
    assert added != first
    with torch.no_grad():
        model.weight.copy_(torch.zeros_like(model.weight))
    copied = sha256_state_dict(model)
    assert copied != added
    with torch.no_grad():
        model.scale.add_(1)
    buffered = sha256_state_dict(model)
    assert buffered != copied
    replacement = Box()
    with torch.no_grad():
        replacement.weight.copy_(model.weight)
        replacement.scale.copy_(model.scale)
    assert sha256_state_dict(replacement) == sha256_state_dict(model)
    moved = replacement.to(dtype=torch.float64)
    assert sha256_state_dict(moved) != sha256_state_dict(model)
    fresh = Box()
    fresh.load_state_dict(model.state_dict())
    assert sha256_state_dict(fresh) == sha256_state_dict(model)


def test_shared_storage_mutation_is_visible_to_a_strict_hash():
    shared = nn.Parameter(torch.arange(4, dtype=torch.float32))

    class Left(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = shared

    class Right(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = shared

    left, right = Left(), Right()
    assert sha256_state_dict(left) == sha256_state_dict(right)
    with torch.no_grad():
        right.weight.data.add_(3)
    assert sha256_state_dict(left) != sha256_state_dict(Box())
    assert sha256_state_dict(left) == sha256_state_dict(right)


def test_lease_reuses_digest_until_bytes_change_then_refuses_publication():
    model = Box()
    other = Box()
    session = VerifiedCheckpointSession.mint(model)
    minted = sha256_state_dict(model)
    assert session.leased_sha256(model) == minted
    with pytest.raises(CheckpointLeaseError, match="not pinned"):
        session.leased_sha256(other)
    with torch.no_grad():
        model.weight.data.add_(1)
    # The heuristic can still match after a .data write. Publication must not.
    assert session.leased_sha256(model) == minted
    with pytest.raises(CheckpointLeaseError, match="bytes differ"):
        session.strict_recheck()
    with pytest.raises(CheckpointLeaseError, match="closed"):
        session.leased_sha256(model)


def test_lease_signature_miss_and_process_boundary():
    model = Box()
    session = VerifiedCheckpointSession.mint(model)
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(CheckpointLeaseError, match="storage signature"):
        session.leased_sha256(model)
    with pytest.raises(CheckpointLeaseError, match="cannot leave"):
        session.__getstate__()
    again = VerifiedCheckpointSession.mint(model)
    again.strict_recheck()
    assert again.leased_sha256(model) == sha256_state_dict(model)


def test_distinct_instances_do_not_share_a_lease():
    first, second = Box(), Box()
    session = VerifiedCheckpointSession.mint(first)
    with pytest.raises(CheckpointLeaseError, match="not pinned"):
        session.leased_sha256(second)
    with pytest.raises(ValueError, match="duplicate"):
        VerifiedCheckpointSession.mint(first, first)
