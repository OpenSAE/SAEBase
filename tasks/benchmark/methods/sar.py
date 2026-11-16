"""Invoke the Capture-owned SAR estimator from a frozen benchmark protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

from saebase.api.paper_sources import load_paper_module
from tasks.capture.sar import evaluate_sar


def source_identity(path, paper):
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / (paper + ".py")
    if not candidate.is_file():
        raise FileNotFoundError(f"source not accessible: {candidate}")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def run_sar(metric_ids, model, sae, site, *, source, options, samples, seed):
    digest = source_identity(source, "sar")
    module = load_paper_module(source, paper="sar")
    cfg = options.get("sar", {})
    if not isinstance(cfg, dict):
        raise ValueError("sar options must be object")
    sample = next((s for s in samples if s.record.split not in {"train", "calibration"}), None)
    if sample is None:
        raise ValueError("SAR requires an evaluation sample")
    t = sample.target_token_id
    f = sample.reference_token_id
    if t is None or f is None:
        raise ValueError("SAR requires target and reference token IDs")

    def readout(logits):
        return float(logits[t] - logits[f])

    import torch

    result = evaluate_sar(
        model,
        sae,
        site=site,
        paper=module,
        readout=readout,
        tokens=torch.tensor([sample.tokens], dtype=torch.long),
        K=int(cfg.get("K", 3)),
        M=int(cfg.get("M", 80)),
        n_fit=int(cfg.get("n_fit", 55)),
        seed=seed,
        p=float(cfg.get("p", 0.5)),
        readout_position=sample.readout_position,
    )
    return result.metrics, len(result.masks), digest
