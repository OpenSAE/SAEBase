from __future__ import annotations

from pathlib import Path

from src.schema import dot, require_keys, unique


def execute(payload, workdir: Path):
    require_keys(payload, {"samples", "encoder", "decoder", "k"})
    encoder, decoder = payload["encoder"], payload["decoder"]
    if not encoder or not decoder or not all(isinstance(row, list) for row in encoder + decoder):
        raise ValueError("weights")
    d_in, width = len(encoder), len(encoder[0])
    if (
        not all(len(row) == width for row in encoder)
        or len(decoder) != width
        or not all(len(row) == d_in for row in decoder)
    ):
        raise ValueError("matrix shape")
    k = payload["k"]
    if type(k) is not int or not 0 <= k <= width:
        raise ValueError("k")
    unique([row["id"] for row in payload["samples"]])
    encoded, decoded, reconstruction = {}, {}, {}
    for row in sorted(payload["samples"], key=lambda item: item["id"]):
        vector = row["vector"]
        if len(vector) != d_in:
            raise ValueError("input width")
        dense = [max(0, dot(vector, [encoder[i][j] for i in range(d_in)])) for j in range(width)]
        selected = sorted(range(width), key=lambda j: (-dense[j], j))[:k]
        sparse = [dense[j] if j in selected else 0 for j in range(width)]
        reconstructed = [dot(sparse, [decoder[j][i] for j in range(width)]) for i in range(d_in)]
        encoded[row["id"]] = {"active_ids": selected, "values": sparse}
        decoded[row["id"]] = reconstructed
        reconstruction[row["id"]] = sum((a - b) ** 2 for a, b in zip(vector, reconstructed)) / d_in
    return {
        "shape": {"d_in": d_in, "features": width},
        "encoded": encoded,
        "decoded": decoded,
        "mse": reconstruction,
    }
