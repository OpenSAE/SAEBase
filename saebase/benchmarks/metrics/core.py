"""Exact token-weighted SAE health metrics on M1 activation/code rows."""
from __future__ import annotations
import numpy as np
import torch
from saebase.store import ShardedReader
from saebase.api.features import _sae_input
from saebase.sae.contracts import reconstruction_stats

def calculate_core(activation_path, code_path, sae, *, sample_ids):
    acts = ShardedReader(activation_path)
    codes = ShardedReader(code_path)
    if acts.kind != "activations" or codes.kind != "codes":
        raise ValueError("expected M1 activation and code readers")
    if codes.manifest["identity"].get("activation_key") != acts.manifest["identity_key"]:
        raise ValueError("code provenance differs from activation cache")
    totals = {"core.l0":0., "core.reconstruction_mse":0., "core.fvu":0., "core.explained_variance":0.}
    n = 0
    for sample_id in sample_ids:
        a, c = acts.get(sample_id), codes.get(sample_id)
        if a.positions != c.positions or a.record.sample_id != c.record.sample_id:
            raise ValueError("activation/code sample alignment mismatch")
        h = np.asarray(a.values, dtype=np.float64)
        z = np.asarray(c.values, dtype=np.float32)
        if h.ndim != 2 or z.ndim != 2 or h.shape[0] != z.shape[0]:
            raise ValueError("activation and codes require aligned [tokens,features] arrays")
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(z)):
            raise ValueError("nonfinite activation/code")
        try:
            device = next(sae.parameters()).device
            dtype = next(sae.parameters()).dtype
        except (StopIteration, AttributeError, TypeError):
            device, dtype = torch.device("cpu"), torch.float32
        with torch.no_grad():
            decoded = sae.decode(torch.as_tensor(z, device=device, dtype=dtype)).detach().float().cpu().numpy()
        if decoded.shape != h.shape or not np.all(np.isfinite(decoded)):
            raise ValueError("SAE decode mismatch or nonfinite output")
        for original, predicted, latent in zip(h, decoded, z):
            # FVU here is exactly 1 - legacy activation_health fve,
            # including its per-token 1e-12 denominator stabilizer.
            stats = reconstruction_stats(torch.as_tensor(original, dtype=torch.float64),
                                         torch.as_tensor(predicted, dtype=torch.float64))
            totals["core.l0"] += int(np.count_nonzero(latent))
            totals["core.reconstruction_mse"] += stats["reconstruction_mse"]
            totals["core.fvu"] += 1.0 - stats["fve"]
            totals["core.explained_variance"] += stats["fve"]
            n += 1
    if n == 0:
        return {}, 0
    return {k:v/n for k,v in totals.items()}, n
