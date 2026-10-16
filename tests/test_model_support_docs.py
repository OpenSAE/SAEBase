"""Offline tests for current-main support documentation and evidence checks."""

from copy import deepcopy

import pytest

from scripts.generate_model_support import (
    ROOT, EVIDENCE_REL, _json, load_support, render_blocks, replace_block, validate_data,
)


def test_current_docs_match_committed_stage_d_evidence():
    support = load_support()
    assert support["total"] == sum(len(t["selected_layers"]) for t in support["catalog"])
    assert support["reliability"] <= support["benchmark"] <= support["pair"] <= support["total"]
    for name, rendered in render_blocks(support).items():
        content = (ROOT / name).read_text(encoding="utf-8")
        assert replace_block(content, rendered, name) == content, name


def test_bad_final_matrix_counts_are_rejected():
    catalog = _json(ROOT / "conf/coverage/stage_bc.json")
    matrix = _json(ROOT / EVIDENCE_REL / "final/stage_d_final_matrix.json")
    status = _json(ROOT / EVIDENCE_REL / "status.json")
    changed = deepcopy(matrix)
    changed["n_pair_verified"] += 1
    with pytest.raises(ValueError, match="n_pair_verified"):
        validate_data(catalog, changed, status, ROOT)


def test_missing_markers_fail_closed():
    with pytest.raises(ValueError, match="marker pair"):
        replace_block("# missing markers\n", "text", "README.md")
