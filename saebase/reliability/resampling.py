"""Prompt-group cluster resampling with frozen within-group variants."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
import numpy as np

from .schema import Observation

def group_samples(observations: Sequence[Observation]) -> dict[str,tuple[Observation,...]]:
    """A prompt's variants remain one resampling unit. No train/test mixing."""
    if not observations:
        raise ValueError("at least one observation required")
    keys={row.series_key for row in observations}
    if len(keys)!=1:
        raise ValueError("cannot mix SAE, metric, condition, split, dataset or seed in bootstrap")
    grouped=defaultdict(list)
    seen=set()
    for o in observations:
        if o.measurement_ref is not None:
            if o.measurement_ref in seen:
                raise ValueError("duplicate measurement reference")
            seen.add(o.measurement_ref)
        grouped[o.prompt_group_id].append(o)
    return {k:tuple(v) for k,v in sorted(grouped.items())}

def draw_group_indices(n_groups:int, n_draws:int, *, seed:int)->np.ndarray:
    if type(n_groups) is not int or n_groups<2:
        raise ValueError("at least two independent prompt groups required")
    if type(n_draws) is not int or n_draws<1:
        raise ValueError("n_draws must be positive")
    if type(seed) is not int or seed<0:
        raise ValueError("seed must be nonnegative integer")
    return np.random.default_rng(seed).integers(0,n_groups,size=(n_draws,n_groups))

def group_sums(observations:Sequence[Observation]):
    groups=group_samples(observations)
    names=tuple(groups)
    numerators=np.asarray([
        sum(o.value*o.unit_count for o in groups[key]) for key in names],dtype=np.float64)
    counts=np.asarray([
        sum(o.unit_count for o in groups[key]) for key in names],dtype=np.float64)
    if not np.isfinite(numerators).all() or np.any(counts<=0):
        raise ValueError("invalid group values or weights")
    return names,numerators,counts

def estimate_with_draws(numerators,counts,draws):
    """Computes the observation-unit-weighted estimator after whole-group draws."""
    n=np.asarray(numerators,dtype=np.float64)
    d=np.asarray(counts,dtype=np.float64)
    if n.ndim!=1 or d.shape!=n.shape or np.any(d<=0):
        raise ValueError("invalid group aggregation")
    ids=np.asarray(draws,dtype=np.int64)
    if ids.ndim!=2 or ids.shape[1]<1 or np.any(ids<0) or np.any(ids>=len(n)):
        raise ValueError("invalid resampling indices")
    return n[ids].sum(axis=1)/d[ids].sum(axis=1)
