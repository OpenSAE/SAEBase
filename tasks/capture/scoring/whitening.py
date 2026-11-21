"""Capture-owned diagonal z-score fitting on all real prompt-token occurrences.

Uses the existing model capture API and native SAE encode/decode. Statistics
are run-local artifacts, never a second Dataset or resource cache. Last-token
snapshots stay on CPU so ranking does not need a second model forward.
"""

from saebase.runtime.configuration.task_options import task_value

from dataclasses import dataclass, fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "saebase.capture.zscore.all_prompt_tokens.v1"


class Moments:
    """FP64 parallel central-moment merging, weighted by token occurrences."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, values):
        if values.ndim != 2 or not values.shape[0] or not bool(torch.isfinite(values).all()):
            raise ValueError("Expected finite nonempty token codes")
        values = values.double()
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        count = values.shape[0]
        m2 = variance * count
        if self.count == 0:
            self.mean, self.m2 = mean, m2
        else:
            delta = mean - self.mean
            total = self.count + count
            self.m2 = self.m2 + m2 + delta.square() * (self.count * count / total)
            self.mean = self.mean + delta * (count / total)
        self.count += count


@dataclass
class Normalization:
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray
    identity: dict
    metadata: dict

    def rank(self, codes, top_feature_num):
        raw = codes.detach().cpu().numpy().astype(np.float64)
        if raw.shape != self.mean.shape:
            raise ValueError("Whitening and SAE feature dimensions differ")
        valid = self.std > 0
        score = np.zeros_like(raw)
        score[valid] = (raw[valid] - self.mean[valid]) / self.std[valid]
        if not np.isfinite(score).all():
            raise ValueError("Whitened scores are non-finite")
        ids = np.flatnonzero(valid)
        top = ids[np.lexsort((ids, -np.abs(score[ids])))[:top_feature_num]]
        return score, top.tolist()


@torch.no_grad()
def fit_normalization(cfg, pair, rows, directory):
    from saebase.api.features import encode_activations, snapshot_from_activation
    from saebase.runtime.sae_operations import pair_state_fingerprints
    from saebase.runtime.experiment_results import write_json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    moments = Moments()
    snapshots = []
    digest = hashlib.sha256()
    chunk_size = task_value(cfg.task, "whiten_chunk_size", 256)
    for index, row in enumerate(rows):
        ids = pair.model.tokens(prompt=row["prompt"])
        activation = pair.model.capture_sequence(tokens=ids, site=pair.site)
        for start in range(0, activation.shape[0], chunk_size):
            codes = encode_activations(pair.sae, activation[start : start + chunk_size])
            moments.update(codes)
            del codes
        # Preserve the existing single-token native encoding for emitted values.
        snapshot = snapshot_from_activation(
            pair.sae, tokens=ids, site=pair.site, activation=activation[-1]
        )
        snapshots.append(
            replace(
                snapshot,
                **{
                    field.name: getattr(snapshot, field.name).detach().cpu()
                    for field in fields(snapshot)
                    if isinstance(getattr(snapshot, field.name), torch.Tensor)
                },
            )
        )
        digest.update(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "prompt": row["prompt"],
                    "tokens": ids.detach().cpu().tolist(),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        )
        write_json(
            directory / "whiten_progress.json",
            {
                "phase": "normalization",
                "processed_prompts": index + 1,
                "requested_prompts": len(rows),
                "token_count": moments.count,
            },
        )
        del activation, snapshot, ids
    if moments.count == 0:
        raise ValueError("Whitening population has no prompt tokens")
    mean = moments.mean.cpu().numpy()
    variance = (moments.m2 / moments.count).cpu().numpy()
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("Invalid whitening population moments")
    std = np.sqrt(variance)
    model_sha, sae_sha = pair_state_fingerprints(pair)
    content = hashlib.sha256()
    for values in (mean, variance):
        content.update(values.astype("<f8").tobytes())
    content.update(str(moments.count).encode())
    identity = {
        "protocol_id": PROTOCOL,
        "content_sha256": content.hexdigest(),
        "model_sha256": model_sha,
        "sae_sha256": sae_sha,
        "cohort_sha256": digest.hexdigest(),
        "token_count": moments.count,
        "layer": pair.site.layer,
        "hook_name": pair.site.hook_name,
        "whiten_chunk_size": chunk_size,
    }
    from models.numerics import identity as numerical_identity

    identity["numerics"] = numerical_identity()
    artifact = directory / "capture_normalization.npz"
    temporary = artifact.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        mean=mean,
        std=std,
        variance=variance,
        valid_features=std > 0,
        count=np.array(moments.count),
    )
    temporary.replace(artifact)
    metadata = {
        "identity": identity,
        "prompt_count": len(rows),
        "feature_count": len(mean),
        "valid_feature_count": int((std > 0).sum()),
        "zero_variance_policy": "zero score; excluded from top ranking",
        "population": "all selected prompt token occurrences, including BOS/instructions/suffix, no padding or appended gold",
        "variance": "population, denominator N",
        "accumulator_dtype": "float64",
        "native_activation": "unchanged last-token single-sample SAE encoding",
        "feature_filter": "none; no SAR-specific semantic blocklist",
        "artifact": "metadata/capture_normalization.npz",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    write_json(directory / "capture_normalization.json", metadata)
    return Normalization(mean, std, variance, identity, metadata), snapshots
