"""Read complete saved codes and retain RSS-excluded, numerically specific candidates."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from omegaconf import OmegaConf
from saebase.api import SAEIdentity
from saebase.api.neuronpedia import annotate_features
from tasks.capture.operation import neuronpedia_settings


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    cases = []
    all_ids = set()
    for path in sorted(root.glob("case_*/rss_case.json")):
        result = json.loads(path.read_text())
        with np.load(path.parent / "rss_activations.npz") as arrays:
            clean, variants, delta = (
                arrays["clean"],
                arrays["variants"],
                arrays["delta"].astype(np.float64),
            )
        params = result["rss_parameters"]
        magnitude = np.abs(delta)
        h = magnitude.mean(0)
        up, down = (delta > params["tau"]).sum(0), (delta < -params["tau"]).sum(0)
        moved = up + down
        recurrence = moved / len(delta)
        consistency = np.divide(np.maximum(up, down), moved, out=np.zeros_like(h), where=moved > 0)
        in_pool = np.zeros(len(h), dtype=bool)
        in_pool[np.argsort(-h, kind="stable")[: params["M"]]] = True
        support = {r["feature_id"] for r in result["rss_features"]}
        selected = np.array([f in support for f in range(len(h))])
        eligible = in_pool & (recurrence >= params["rho"]) & (consistency >= params["kappa"])
        np.savez_compressed(
            path.parent / "rss_feature_statistics.npz",
            mean_absolute_change=h,
            recurrence=recurrence,
            consistency=consistency,
            in_top_M=in_pool,
            eligible=eligible,
            rss_selected=selected,
        )
        groups = []
        for v in range(len(delta)):
            others = np.delete(magnitude, v, axis=0).max(0)
            ids = [
                int(f)
                for f in np.argsort(-magnitude[v], kind="stable")
                if f not in support
                and delta[v, f] > params["tau"]
                and magnitude[v, f] >= 5 * others[f]
            ]
            rows = []
            for rank, f in enumerate(ids, 1):
                reasons = []
                if not in_pool[f]:
                    reasons.append("outside_top_M")
                if recurrence[f] < params["rho"]:
                    reasons.append("recurrence_below_rho")
                if consistency[f] < params["kappa"]:
                    reasons.append("consistency_below_kappa")
                if eligible[f]:
                    reasons.append("outside_top_K")
                rows.append(
                    {
                        "feature_id": f,
                        "candidate_rank": rank,
                        "clean_activation": float(clean[f]),
                        "variant_activations": variants[:, f].tolist(),
                        "delta": delta[:, f].tolist(),
                        "recurrence": float(recurrence[f]),
                        "consistency": float(consistency[f]),
                        "mean_absolute_change": float(h[f]),
                        "in_top_M": bool(in_pool[f]),
                        "rss_selected": False,
                        "exclusion_reasons": reasons,
                        "other_max_change": float(others[f]),
                        "dominance_ratio": float(magnitude[v, f] / others[f])
                        if others[f]
                        else None,
                    }
                )
            all_ids.update(ids)
            groups.append({"variant": v + 1, "context": result["contexts"][v], "features": rows})
        all_ids.update(support)
        cases.append((path, result, groups))
    pair = SimpleNamespace(
        identity=SAEIdentity(**cases[0][1]["identity"]), sae=SimpleNamespace(cfg=None)
    )
    info, annotations = annotate_features(
        pair, sorted(all_ids), neuronpedia_settings(OmegaConf.create({"task": {}}))
    )
    (root / "batch_annotations.json").write_text(
        json.dumps({"info": info, "features": annotations}, ensure_ascii=False, indent=2) + "\n"
    )
    for path, result, groups in cases:
        for group in groups:
            for row in group["features"]:
                row["annotation"] = annotations[row["feature_id"]]
        payload = {
            "candidate_rule": "All unselected features with positive variant delta > tau and >=5 times the maximum absolute change of the other four variants, ordered by variant change. Recurrence failure preferred for display, not mandatory in this batch specification.",
            "original_capture": str(path),
            "neuronpedia": info,
            "groups": groups,
        }
        (path.parent / "context_candidates.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        print(path.parent.name, "candidates", [len(g["features"]) for g in groups])


if __name__ == "__main__":
    main()
