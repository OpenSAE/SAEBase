"""Invoke unchanged SAR/RSS paper estimators via existing SAEBase interfaces."""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from tasks.LLM.benchmark.integrations import (
    evaluate_sar,
    evaluate_rss,
    collect_rss_features,
    load_paper_module,
)


def source_identity(path, paper):
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / (paper + ".py")
    if not candidate.is_file():
        raise FileNotFoundError(f"source not accessible: {candidate}")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def run_paper(metric_ids, model, sae, site, *, source, options, samples, seed):
    """Return (metrics, n, source_sha), preserving raw estimator output.

    SAR uses a frozen coalitions protocol; RSS requires explicitly disjoint
    selection and held-out token sequences and never selects on held-out data.
    """
    paper = "sar" if all(x.startswith("sar.") for x in metric_ids) else "rss"
    digest = source_identity(source, paper)
    module = load_paper_module(source, paper=paper)
    if paper == "sar":
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
    cfg = options.get("rss", {})
    if not isinstance(cfg, dict):
        raise ValueError("rss options must be object")
    clean = [s for s in samples if s.record.variant_role == "clean"]
    select = [s for s in samples if s.record.variant_role == "selection"]
    held = [s for s in samples if s.record.variant_role == "held_out"]
    if len(clean) != 1 or not select or not held:
        raise ValueError("RSS requires 1 clean, nonempty selection, nonempty held_out variants")
    selected_groups = {s.record.sample_id for s in select}
    if selected_groups.intersection(s.record.sample_id for s in held):
        raise ValueError("selection and held_out overlap")

    def prompt(s):
        # RSS existing integration expects text; caller provides exact texts
        mapping = cfg.get("prompts", {})
        result = mapping.get(s.record.sample_id)
        if not isinstance(result, str) or not result:
            raise ValueError(
                f"RSS requires text for sample_id {s.record.sample_id} in options.rss.prompts"
            )
        # M1's exact token IDs, rather than unchecked text, define the data
        # identity and split. Do not silently evaluate a different sequence.
        ids = model.tokens(prompt=result)[0].detach().cpu().tolist()
        if ids != list(s.record.token_ids):
            raise ValueError(f"RSS text/token IDs mismatch for sample_id {s.record.sample_id}")
        return result

    features = collect_rss_features(
        model,
        sae,
        site=site,
        clean_prompt=prompt(clean[0]),
        selection_variants=[prompt(s) for s in select],
        held_out_variants=[prompt(s) for s in held],
    )
    result = evaluate_rss(
        features,
        paper=module,
        rho=float(cfg.get("rho", 0.8)),
        kappa=float(cfg.get("kappa", 0.8)),
        tau=float(cfg.get("tau", 1e-5)),
        M=int(cfg.get("M", 400)),
        K=int(cfg.get("K", 20)),
    )
    return (
        {
            "mean_rss": result.mean_rss,
            "held_out_persistence": float(np.mean(result.held_out_rates)),
        },
        len(held),
        digest,
    )
