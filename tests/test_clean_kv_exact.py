"""CPU oracle for Qwen-compatible clean-history KV editing.

No pretrained model or external GPU package is required. Real Qwen on RTX 5070
is the mandatory later validation, since attention kernels may alter TopK.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from examples._synthetic_common import ResearchToySAE
from models.hf_hooks_backend import HFHooksModel
from saebase.api import ModelAdapter
from saebase.interventions.kv_cache_exact import (
    CleanPrefixKVDecoder,
    KVParityError,
    compare_cached_step_with_reference,
)
from saebase.interventions.specs import Condition


class ToyCache:
    def __init__(self, token_ids: torch.Tensor):
        self.token_ids = token_ids

    def get_seq_length(self) -> int:
        return int(self.token_ids.shape[1])


class Block(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.1 + 0.2


class ToyQwen3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=1)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block()])
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        attention_mask=None,
    ):
        self.calls += 1
        prior = past_key_values.token_ids if past_key_values is not None else input_ids[:, :0]
        ids = torch.cat((prior, input_ids), dim=1)
        if cache_position is not None:
            assert torch.equal(cache_position.cpu(), torch.arange(prior.shape[1], ids.shape[1]))
        if attention_mask is not None:
            assert attention_mask.shape[1] == ids.shape[1]
        sums = ids.float().cumsum(dim=1)[:, -input_ids.shape[1] :]
        h = sums.unsqueeze(-1).expand(-1, -1, 4).contiguous() * self.scale / 10
        for layer in self.model.layers:
            h = layer(h)
        y = torch.stack((h[..., 0], -h[..., 1], h[..., 2] + h[..., 3]), dim=-1)
        return SimpleNamespace(logits=y, past_key_values=ToyCache(ids) if use_cache else None)


def build():
    raw = ToyQwen3().eval()
    hf = HFHooksModel(raw, SimpleNamespace(), model_name="toy-qwen3", device="cpu")
    sae = ResearchToySAE().eval()
    return hf, ModelAdapter(hf), sae


def test_incremental_clean_kv_matches_full_prefix_stepwise():
    hf, adapter, sae = build()
    decoder = CleanPrefixKVDecoder(hf, sae, layer=0, hook_name="blocks.0.hook_resid_post")
    decoder.start(torch.tensor([[1, 2, 3]], dtype=torch.long))
    for next_id in (4, 5, 6, 7):
        row = compare_cached_step_with_reference(
            decoder,
            adapter,
            sae,
            Condition("ablate", "ablate", (0,)),
            atol=1e-5,
            rtol=1e-4,
        )
        assert row["pass"] and row["feature_flips"] == 0
        decoder.append(next_id)
    decoder.close()


def test_incremental_identity_and_scale_stay_exact():
    hf, adapter, sae = build()
    decoder = CleanPrefixKVDecoder(hf, sae, layer=0, hook_name="blocks.0.hook_resid_post")
    decoder.start(torch.tensor([[3]], dtype=torch.long))
    for condition, token in (
        (Condition("i", "identity"), 4),
        (Condition("s", "scale", (0,), 0.5), 5),
        (Condition("a", "add", (0,), 0.1), 6),
    ):
        row = compare_cached_step_with_reference(
            decoder, adapter, sae, condition, atol=1e-5, rtol=1e-4
        )
        assert row["pass"]
        decoder.append(token)


def test_kv_refuses_unsupported_model_and_bad_sequence():
    hf, adapter, sae = build()
    hf._model.config.model_type = "other"
    with pytest.raises(ValueError, match="Qwen3"):
        CleanPrefixKVDecoder(hf, sae, layer=0, hook_name="blocks.0.hook_resid_post")
    hf._model.config.model_type = "qwen3"
    decoder = CleanPrefixKVDecoder(hf, sae, layer=0, hook_name="blocks.0.hook_resid_post")
    with pytest.raises(RuntimeError, match="started"):
        decoder.step(Condition("i", "identity"))
    with pytest.raises(ValueError):
        decoder.start(torch.zeros((1, 0), dtype=torch.long))
    decoder.start(torch.tensor([[1, 2]], dtype=torch.long))
    with pytest.raises(RuntimeError, match="only once"):
        decoder.start(torch.tensor([[1, 2]], dtype=torch.long))
