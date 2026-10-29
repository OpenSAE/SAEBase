from __future__ import annotations

from pathlib import Path

from src.schema import dot, require_keys


def execute(payload, workdir: Path):
    require_keys(payload, {"samples", "encoder", "decoder", "ablate"})
    if not payload["samples"]:
        raise ValueError("no samples")
    encoder, decoder = payload["encoder"], payload["decoder"]
    if len(encoder) != len(payload["samples"][0]["vector"]) or len(decoder) != len(encoder[0]):
        raise ValueError("shape")
    rows = []
    deltas = []
    for sample in sorted(payload["samples"], key=lambda item: item["id"]):
        vector = sample["vector"]
        dense = [
            max(0, dot(vector, [encoder[i][j] for i in range(len(encoder))]))
            for j in range(len(encoder[0]))
        ]
        index = payload["ablate"]
        if type(index) is not int or index < 0 or index >= len(dense):
            raise ValueError("feature")
        before = sum(
            dot(dense, [decoder[j][i] for j in range(len(decoder))]) for i in range(len(vector))
        )
        changed = dense[:]
        changed[index] = 0
        after = sum(
            dot(changed, [decoder[j][i] for j in range(len(decoder))]) for i in range(len(vector))
        )
        rows.append(
            {
                "id": sample["id"],
                "features": dense,
                "prediction_before": before,
                "prediction_after": after,
            }
        )
        deltas.append(after - before)
    return {
        "samples": rows,
        "ablation_feature": payload["ablate"],
        "mean_prediction_delta": sum(deltas) / len(deltas),
        "coverage": {"evaluated": len(rows), "total": len(payload["samples"])},
    }
