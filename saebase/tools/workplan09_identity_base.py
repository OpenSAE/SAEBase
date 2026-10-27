#!/usr/bin/env python3
"""Identity replay on the pinned base commit, kept separate from candidate diffs.

Uses the base tree's BatchEngine. Same-width and cross-width errors are both
recorded. Prompt text is not written.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE = Path("/workspace/SAEBase-wp09-base")
sys.path.insert(0, str(BASE))

import torch  # noqa: E402

from saebase.api import ActivationSite, ModelAdapter  # noqa: E402
from saebase.interventions import (  # noqa: E402
    BatchEngine,
    Condition,
    ExperimentIdentity,
    MeasurementSample,
    ResultWriter,
)
from saebase.sae.loading import load_model_and_sae  # noqa: E402
from saebase.sae.registry import get_profile  # noqa: E402
from saebase.store import SampleRecord, TokenSelection, sha256_state_dict  # noqa: E402

SITES = (
    ("pythia410m_mlp_topk", 4, "pythia410m", "pythia410m_pythia410m_mlp_topk_L4"),
    (
        "r1_distill_qwen15b_resid_l20",
        20,
        "r1_distill_qwen15b",
        "r1_distill_qwen15b_r1_distill_qwen15b_resid_l20_L20",
    ),
)
RUN = Path("/workspace/experiments/performance/workplan09/rtx5070-20261021")


def samples_from(path: Path) -> tuple[list[MeasurementSample], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for row in payload["rows"]:
        mode = row["selection_mode"]
        selection = (
            TokenSelection("indices", tuple(row["indices"]))
            if mode == "indices"
            else TokenSelection("last_valid")
        )
        samples.append(
            MeasurementSample(
                SampleRecord(
                    row["sample_id"],
                    tuple(row["token_ids"]),
                    selection=selection,
                    prompt_group_id=row["group_id"],
                    split="wp09-pilot",
                ),
                readout_position=int(row["readout_position"]),
                target_token_id=int(row["target_token_id"]),
            )
        )
    return samples, int(payload["pad_token_id"])


def max_abs(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return max(float((a.float() - b.float()).abs().max().item()) for a, b in zip(left, right))


def measure(sae_key: str, layer: int, model_key: str, folder: str) -> dict[str, object]:
    raw, sae, meta = load_model_and_sae(sae_key, layer, device="cuda")
    adapter = ModelAdapter(raw)
    samples, pad = samples_from(RUN / "sites" / folder / "shared" / "samples.json")
    group = samples[:8]
    hook = get_profile(sae_key).hook_name(layer)
    identity = ExperimentIdentity(
        sha256_state_dict(raw),
        sha256_state_dict(sae),
        ActivationSite(layer, hook),
        model_key=model_key,
        sae_key=sae_key,
        backend=adapter.backend,
    )
    engine = BatchEngine(adapter, sae, identity, pad_token_id=pad)
    snaps, logits = engine._baselines(group)
    pairs = [(sample, snap.activation) for sample, snap in zip(group, snaps)]
    same = engine._patched_logits(pairs)
    half = len(group) // 2
    narrow = engine._patched_logits(pairs[:half])
    row: dict[str, object] = {
        "commit": "339fbb603bc6d999fd126a91ea8709fee1c599c2",
        "model_key": model_key,
        "hook": meta.get("hook_name"),
        "same_width_max_abs": max_abs(logits, same),
        "cross_width_max_abs": max_abs(logits[:half], narrow),
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with ResultWriter(tmp, identity) as writer:
                engine.run(
                    group,
                    [Condition("identity", "identity")],
                    writer,
                    sample_batch=8,
                    microbatch=4,
                    tolerance=1e-5,
                    verify_identity=True,
                )
        row["old_protocol_gate"] = "PASS"
    except RuntimeError as exc:
        row["old_protocol_gate"] = "FAIL"
        row["error"] = str(exc)[:300]
    del raw, sae, adapter, engine
    torch.cuda.empty_cache()
    return row


def main() -> None:
    report = [measure(*spec) for spec in SITES]
    for row in report:
        print(json.dumps(row), flush=True)
    (RUN / "identity_width_base.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
