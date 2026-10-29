from __future__ import annotations

from pathlib import Path

from src.schema import finite, require_keys, unique


def execute(payload, workdir: Path):
    require_keys(payload, {"tokens", "features", "topk"})
    ids = [row["id"] for row in payload["tokens"]]
    unique(ids)
    width = payload["features"]
    if type(width) is not int or width < 1 or not 1 <= payload["topk"] <= width:
        raise ValueError("dimensions")
    ranked = {}
    provenance = []
    for token in payload["tokens"]:
        activations = token["activations"]
        if len(activations) != width or not all(finite(value) for value in activations):
            raise ValueError("activation")
        selected = sorted(range(width), key=lambda i: (-activations[i], i))[: payload["topk"]]
        ranked[token["id"]] = selected
        for feature in selected:
            provenance.append(
                {"token": token["id"], "feature": feature, "value": activations[feature]}
            )
    return {
        "token_ids": sorted(ids),
        "ranked_ids": ranked,
        "provenance": sorted(provenance, key=lambda row: (row["token"], row["feature"])),
        "unresolved": [],
    }
