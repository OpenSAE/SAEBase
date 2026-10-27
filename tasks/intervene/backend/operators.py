"""Feature-space operations with explicit reconstruction-residual semantics."""

from __future__ import annotations

import torch

from saebase.api import FeaturePatch, FeatureSnapshot, edit_features
from .specs import Condition


@torch.no_grad()
def apply_condition(sae: object, snapshot: FeatureSnapshot, condition: Condition) -> FeaturePatch:
    if not isinstance(condition, Condition):
        raise TypeError("condition must be a Condition")
    codes = snapshot.codes
    if any(i >= codes.numel() for i in (*condition.feature_ids, *condition.retained_ids)):
        raise IndexError("feature index exceeds SAE width")
    if condition.operation == "identity":
        return edit_features(sae, snapshot, replacements={})
    if condition.operation == "reconstruction":
        # Deliberately *not* the residual-preserving edit: used for loss recovered.
        return FeaturePatch(
            codes=codes.detach().clone(), activation=snapshot.reconstruction.detach().clone()
        )
    if condition.operation == "coalition":
        values = {i: 0.0 for i in condition.feature_ids if i not in condition.retained_ids}
    elif condition.operation == "ablate":
        values = {i: 0.0 for i in condition.feature_ids}
    elif condition.operation == "scale":
        values = {i: codes[i] * condition.value for i in condition.feature_ids}
    elif condition.operation == "clamp":
        values = {i: condition.value for i in condition.feature_ids}
    elif condition.operation == "add":
        values = {i: codes[i] + condition.value for i in condition.feature_ids}
    else:
        raise RuntimeError("unreachable intervention operation")
    return edit_features(sae, snapshot, replacements=values)
