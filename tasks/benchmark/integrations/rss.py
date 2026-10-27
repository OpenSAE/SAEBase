"""RSS feature response acquisition through SAEBase's public capture API.

Selection and held-out variants are disjoint before any metric is computed.
The only RSS metric implementations used are supplied by the original rss.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch

from saebase.api import ActivationSite, ModelAdapter, SAEIdentity, capture_features


@dataclass(frozen=True)
class RSSFeatures:
    clean_codes: np.ndarray
    selection_codes: np.ndarray
    held_out_codes: np.ndarray
    selection_changes: np.ndarray
    held_out_changes: np.ndarray
    selection_positions: tuple[int, ...]
    held_out_positions: tuple[int, ...]
    clean_position: int
    site: ActivationSite
    identity: SAEIdentity | None = None


@dataclass(frozen=True)
class RSSResult:
    support: np.ndarray
    direction: np.ndarray
    mean_rss: float
    held_out_rates: np.ndarray
    features: RSSFeatures

    def summary(self) -> dict[str, Any]:
        return {
            "support": self.support.tolist(),
            "direction": self.direction.tolist(),
            "mean_rss": self.mean_rss if np.isfinite(self.mean_rss) else None,
            "held_out_rates": self.held_out_rates.tolist(),
            "n_selection": int(self.features.selection_changes.shape[0]),
            "n_held_out": int(self.features.held_out_changes.shape[0]),
            "site": {
                "layer": self.features.site.layer,
                "hook_name": self.features.site.hook_name,
                "clean_position": self.features.clean_position,
                "selection_positions": list(self.features.selection_positions),
                "held_out_positions": list(self.features.held_out_positions),
            },
            "identity": self.features.identity.to_dict() if self.features.identity else None,
        }


def _prompts(clean: str, selection: Sequence[str], held_out: Sequence[str]) -> None:
    if not isinstance(clean, str) or not clean.strip():
        raise ValueError("clean prompt must be a nonempty string")
    for name, strings in (("selection", selection), ("held_out", held_out)):
        if (
            isinstance(strings, (str, bytes))
            or not isinstance(strings, (list, tuple))
            or not strings
        ):
            raise ValueError(f"{name} variants must be a nonempty sequence of strings")
        if any(not isinstance(s, str) or not s.strip() for s in strings):
            raise ValueError(f"{name} variants must contain nonempty strings")
    combined = [clean, *selection, *held_out]
    if len(set(combined)) != len(combined):
        raise ValueError("Clean, selection, and held-out prompts must be disjoint and unique")


def _positions(pos: Sequence[int] | None, size: int, default: int, name: str) -> tuple[int, ...]:
    if pos is None:
        return (default,) * size
    if (
        isinstance(pos, (str, bytes))
        or len(pos) != size
        or any(isinstance(i, bool) or not isinstance(i, int) for i in pos)
    ):
        raise ValueError(f"{name} positions must contain one integer token index per variant")
    return tuple(pos)


@torch.no_grad()
def collect_rss_features(
    model: ModelAdapter,
    sae: Any,
    *,
    site: ActivationSite,
    clean_prompt: str,
    selection_variants: Sequence[str],
    held_out_variants: Sequence[str],
    selection_positions: Sequence[int] | None = None,
    held_out_positions: Sequence[int] | None = None,
    identity: SAEIdentity | None = None,
) -> RSSFeatures:
    """Encode clean/selection/held-out prompts at explicitly chosen sites.

    Positions default to ``site.position`` for each prompt independently.
    A caller dealing with prefix/wrapper changes should pass appropriate
    variant positions or confirm that last-token refers to the same concept.
    Selection and held-out codes are returned in separate, immutable-by-convention
    copies; RSS selection must consume ONLY ``selection_changes``.
    """
    _prompts(clean_prompt, selection_variants, held_out_variants)
    sel_p = _positions(selection_positions, len(selection_variants), site.position, "selection")
    ho_p = _positions(held_out_positions, len(held_out_variants), site.position, "held_out")
    model.capabilities.require("capture")
    if identity is not None and (
        identity.hook_name != site.hook_name or identity.layer != site.layer
    ):
        raise ValueError("SAE identity site differs from requested activation site")

    def observe(prompt: str, position: int) -> tuple[np.ndarray, int]:
        selected_site = replace(site, position=position)
        ids = model.tokens(prompt=prompt)
        snap = capture_features(model, sae, tokens=ids, site=selected_site)
        if identity is not None and (
            snap.activation.numel() != identity.d_in or snap.codes.numel() != identity.d_sae
        ):
            raise ValueError("SAE identity dimensions differ from captured features")
        idx = int(position % ids.shape[1])
        return snap.codes.detach().cpu().numpy().astype(np.float64, copy=True), idx

    clean_codes, clean_position = observe(clean_prompt, site.position)
    selection_observations = [
        observe(p, position) for p, position in zip(selection_variants, sel_p)
    ]
    held_observations = [observe(p, position) for p, position in zip(held_out_variants, ho_p)]
    selection_codes = np.stack([x[0] for x in selection_observations], axis=0)
    held_codes = np.stack([x[0] for x in held_observations], axis=0)
    for name, arr in (("selection", selection_codes), ("held_out", held_codes)):
        if arr.shape[1] != clean_codes.shape[0] or not np.isfinite(arr).all():
            raise ValueError(f"{name} SAE feature responses are inconsistent or nonfinite")
    return RSSFeatures(
        clean_codes=clean_codes,
        selection_codes=selection_codes,
        held_out_codes=held_codes,
        selection_changes=selection_codes - clean_codes[None, :],
        held_out_changes=held_codes - clean_codes[None, :],
        selection_positions=tuple(int(x[1]) for x in selection_observations),
        held_out_positions=tuple(int(x[1]) for x in held_observations),
        clean_position=clean_position,
        site=site,
        identity=identity,
    )


def evaluate_rss(
    features: RSSFeatures,
    *,
    paper: Any,
    rho: float = 0.8,
    kappa: float = 0.8,
    tau: float = 1e-5,
    M: int = 400,
    K: int = 20,
) -> RSSResult:
    """Call the paper's rss() on selection and held_out_rate() on unseen variants."""
    for name in ("rss", "held_out_rate"):
        if not callable(getattr(paper, name, None)):
            raise TypeError(f"paper must expose callable {name}() from rss.py")
    support, direction, mean = paper.rss(
        features.selection_changes, rho=rho, kappa=kappa, tau=tau, M=M, K=K
    )
    rates = paper.held_out_rate(features.held_out_changes, support, direction, tau=tau)
    return RSSResult(
        support=np.asarray(support, dtype=int).copy(),
        direction=np.asarray(direction, dtype=int).copy(),
        mean_rss=float(mean),
        held_out_rates=np.asarray(rates, dtype=float).copy(),
        features=features,
    )
