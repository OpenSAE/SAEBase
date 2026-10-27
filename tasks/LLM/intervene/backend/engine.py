"""Bounded batched SAE intervention execution, validation and resume.

TransformerLens-compatible backends execute actual batch hook forwards. HFHooks
uses the original single-prompt API explicitly (no misleading batch claims).
Right padding is supported because all patches/readouts precede it in a causal
model. Left padding or noncausal models require a different adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import numpy as np
from time import perf_counter
from typing import Any, Sequence
import math

import torch
import torch.nn.functional as F

from saebase.api import ActivationSite, FeatureSnapshot, ModelAdapter
from saebase.api.features import (
    _decode,
    _layer_norm_stats,
    _sae_dimensions,
    _sae_input,
    _sae_uses_layer_norm,
)
from saebase.store import ShardedReader, VerifiedCheckpointSession, sha256_state_dict
from .operators import apply_condition
from .results import ResultWriter
from .specs import Condition, ExperimentIdentity, MeasurementSample, measurement_id


@dataclass(frozen=True)
class ExecutionStats:
    measured: int
    reused: int
    baseline_batches: int
    patch_batches: int
    microbatch_retries: int
    peak_microbatch: int
    elapsed_seconds: float
    backend: str

    @property
    def interventions_per_second(self) -> float:
        return self.measured / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


@dataclass(frozen=True)
class RunBudget:
    max_measurements: int = 100_000
    max_generated_tokens: int = 10_000
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        if type(self.max_measurements) is not int or self.max_measurements < 1:
            raise ValueError("max_measurements must be positive")
        if type(self.max_generated_tokens) is not int or self.max_generated_tokens < 0:
            raise ValueError("max_generated_tokens must be nonnegative")
        if self.max_seconds is not None and (
            not math.isfinite(self.max_seconds) or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be finite and positive")


def dry_run(
    samples: Sequence[MeasurementSample],
    conditions: Sequence[Condition],
    *,
    microbatch: int = 16,
    budget: RunBudget | None = None,
) -> dict[str, Any]:
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be positive")
    if not samples or not conditions:
        raise ValueError("nonempty samples and conditions required")
    if len({s.record.sample_id for s in samples}) != len(samples):
        raise ValueError("duplicate sample IDs")
    if len({c.name for c in conditions}) != len(conditions):
        raise ValueError("duplicate condition names")
    count = len(samples) * len(conditions)
    lim = budget or RunBudget()
    if count > lim.max_measurements:
        raise ValueError(f"{count} measurements exceed budget {lim.max_measurements}")
    return {
        "samples": len(samples),
        "conditions": len(conditions),
        "measurements": count,
        "max_token_length": max(s.valid_length for s in samples),
        "microbatch": microbatch,
        "minimum_patch_forwards_tl": math.ceil(count / microbatch),
        "estimated_runtime_seconds": None,
        "note": "Model/GPU-dependent run time and VRAM require an actual pilot run.",
    }


def _padded_tokens(
    samples: Sequence[MeasurementSample], *, device: torch.device, pad_token_id: int
) -> torch.Tensor:
    size = max(s.valid_length for s in samples)
    tokens = torch.full((len(samples), size), pad_token_id, device=device, dtype=torch.long)
    for i, s in enumerate(samples):
        tokens[i, : s.valid_length] = torch.tensor(s.tokens, dtype=torch.long, device=device)
    return tokens


def _is_oom(exc: BaseException) -> bool:
    """True for CUDA OOM and for runtimes that surface the same failure as text."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _metrics(
    baseline: torch.Tensor, patched: torch.Tensor, sample: MeasurementSample, delta_h: float
) -> dict[str, Any]:
    if baseline.shape != patched.shape or baseline.ndim != 1:
        raise ValueError("baseline/edited logits must have equal [vocab] shapes")
    b, p = baseline.float(), patched.float()
    finite = torch.isfinite(b).all() & torch.isfinite(p).all()
    diff = p - b
    # One host transfer for the reductions. KL stays the sum of
    # softmax(baseline) * (log softmax(baseline) - log softmax(edited)),
    # clamped at 0 because a float32 reduction of an identity can be -1e-8.
    pieces = [
        torch.linalg.vector_norm(diff),
        diff.abs().max(),
        F.kl_div(F.log_softmax(p, -1), F.softmax(b, -1), reduction="sum"),
        finite.to(dtype=b.dtype),
    ]
    target = sample.target_token_id
    reference = sample.reference_token_id
    if target is not None:
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target >= b.numel()
            or target < 0
        ):
            raise IndexError("target token ID exceeds vocabulary")
        if reference is not None and (
            isinstance(reference, bool)
            or not isinstance(reference, int)
            or reference >= b.numel()
            or reference < 0
        ):
            raise IndexError("reference token ID exceeds vocabulary")
        top_b = torch.argmax(b)
        top_p = torch.argmax(p)
        pieces.extend(
            [
                b[target],
                p[target],
                diff[target],
                (top_b == target).to(b.dtype),
                (top_p == target).to(b.dtype),
            ]
        )
        if reference is not None:
            pieces.extend([b[target] - b[reference], p[target] - p[reference]])
    values = torch.stack(pieces).detach().cpu().tolist()
    if values[3] == 0.0:
        raise ValueError("non-finite model output")
    out: dict[str, Any] = {
        "logit_delta_l2": float(values[0]),
        "logit_delta_max_abs": float(values[1]),
        "kl_baseline_to_edited": max(0.0, float(values[2])),
        "activation_delta_l2": delta_h,
    }
    if target is not None:
        out["baseline_target_logit"] = float(values[4])
        out["edited_target_logit"] = float(values[5])
        out["target_logit_delta"] = float(values[6])
        out["baseline_target_top1"] = int(values[7] == 1.0)
        out["edited_target_top1"] = int(values[8] == 1.0)
        if reference is not None:
            out["baseline_logit_difference"] = float(values[9])
            out["edited_logit_difference"] = float(values[10])
    return out


def _boundary_digests_for(
    model: Any, sae: Any, lease: VerifiedCheckpointSession | None
) -> tuple[str, str]:
    if lease is None:
        return sha256_state_dict(model), sha256_state_dict(sae)
    return lease.leased_sha256(model), lease.leased_sha256(sae)


class BatchEngine:
    """Execute cross-product samples x conditions with bounded patch microbatches.

    Optional M1 activation and code readers bypass the initial live capture and
    encoding. Their provenance must match the live model/SAE states. Model
    forwards for downstream logits are *still necessary* after patching.
    """

    def __init__(
        self,
        model: ModelAdapter,
        sae: Any,
        identity: ExperimentIdentity,
        *,
        activation_reader: ShardedReader | None = None,
        code_reader: ShardedReader | None = None,
        pad_token_id: int | None = None,
        enable_hf_masked_batch: bool = False,
        enable_suffix_replay: bool = False,
        identity_lease: VerifiedCheckpointSession | None = None,
        release_content_check: bool | None = None,
    ) -> None:
        self.model, self.sae, self.identity = model, sae, identity
        self.activation_reader, self.code_reader = activation_reader, code_reader
        self.pad_token_id = pad_token_id
        self.identity_lease = identity_lease
        if release_content_check is None:
            release_content_check = identity_lease is not None
        if not isinstance(release_content_check, bool):
            raise TypeError("release_content_check must be a bool")
        self.release_content_check = release_content_check
        self.counters = {
            "model_forwards": 0,
            "combined_capture_forwards": 0,
            "separate_baseline_forwards": 0,
            "sae_encodes": 0,
            "patch_forwards": 0,
            "suffix_prefix_forwards": 0,
            "suffix_replay_forwards": 0,
        }
        # Off by default. Opt-in run_batch uses the single-example kernel.
        # A fused padded GEMM changes Qwen3 bf16 TopK ids, so it is not used.
        # Leave this off until a fused path meets the frozen tolerance and 1.15x.
        if not isinstance(enable_hf_masked_batch, bool):
            raise TypeError("enable_hf_masked_batch must be a bool")
        if not isinstance(enable_suffix_replay, bool):
            raise TypeError("enable_suffix_replay must be a bool")
        if enable_suffix_replay and enable_hf_masked_batch:
            raise ValueError("suffix replay and the masked HF batch cannot be combined")
        self.enable_hf_masked_batch = enable_hf_masked_batch
        # Off by default until a real workload shows at least 1.15x and exact parity.
        self.enable_suffix_replay = enable_suffix_replay
        if enable_suffix_replay:
            probe = getattr(model.model, "supports_suffix_replay", None)
            if model.backend != "hf_hooks" or not callable(probe) or not probe():
                raise ValueError("suffix replay was requested but this model does not support it")
        model.capabilities.require("forward")
        model.capabilities.require("patch")
        if model.backend != identity.backend:
            raise ValueError("model backend differs from experiment identity")
        model_sha, sae_sha = _boundary_digests_for(model.model, sae, identity_lease)
        if model_sha != identity.model_state_sha256:
            raise ValueError("model state differs from experiment identity")
        if sae_sha != identity.sae_state_sha256:
            raise ValueError("SAE state differs from experiment identity")
        self.d_in, self.d_sae = _sae_dimensions(sae)
        if activation_reader is not None:
            m = activation_reader.manifest
            if activation_reader.kind != "activations" or activation_reader.width != self.d_in:
                raise ValueError("activation reader kind/width mismatch")
            provenance = m["identity"]
            for k, v in (
                ("model_state_sha256", identity.model_state_sha256),
                ("hook_name", identity.site.hook_name),
                ("layer", identity.site.layer),
            ):
                if provenance.get(k) != v:
                    raise ValueError(f"activation provenance mismatch: {k}")
            if (
                identity.tokenizer_sha256
                and provenance.get("tokenizer_sha256") != identity.tokenizer_sha256
            ):
                raise ValueError("tokenizer provenance mismatch")
        if code_reader is not None:
            if activation_reader is None:
                raise ValueError("code cache requires its source activation reader")
            meta = code_reader.manifest
            if code_reader.kind != "codes" or code_reader.width != self.d_sae:
                raise ValueError("code reader kind/width mismatch")
            if meta["identity"].get("activation_key") != activation_reader.manifest["identity_key"]:
                raise ValueError("code cache belongs to another activation store")
            if meta["identity"]["sae"].get("sae_state_sha256") != identity.sae_state_sha256:
                raise ValueError("code cache SAE checkpoint differs")

    def _read_cache(self, sample: MeasurementSample) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.activation_reader is None:
            return None
        record = sample.record
        a = self.activation_reader.get(record.sample_id)
        if a.record.key != record.key or a.positions != (sample.patch_position,):
            raise ValueError("cached activation sample identity or selected token position differs")
        device = self.model.tokens(tokens=torch.tensor([sample.tokens])).device
        h = torch.tensor(a.values[0].copy(), dtype=torch.float32, device=device)
        if self.code_reader is None:
            z = self.sae.encode(_sae_input(self.sae, h).unsqueeze(0))[0]
        else:
            c = self.code_reader.get(record.sample_id)
            # M1 code records deliberately include the SHA-256 digest of the
            # *source activation data* to detect stale codes after a repair.
            source = np.asarray(a.values, dtype=np.float32)
            source_hash = hashlib.sha256(source.tobytes(order="C")).hexdigest()
            expected = replace(record, source_activation_sha256=source_hash)
            if c.record.key != expected.key or c.positions != a.positions:
                raise ValueError("cached SAE code record or source activation differs")
            z = torch.tensor(
                c.values[0].copy(), dtype=torch.float32, device=_sae_input(self.sae, h).device
            )
        return h, z

    def _hf_batch_enabled(self) -> bool:
        return bool(
            self.enable_hf_masked_batch
            and self.model.backend == "hf_hooks"
            and callable(getattr(self.model.model, "run_batch", None))
        )

    def _suffix_replay_enabled(self) -> bool:
        probe = getattr(self.model.model, "supports_suffix_replay", None)
        return bool(
            self.enable_suffix_replay
            and self.model.backend == "hf_hooks"
            and callable(probe)
            and probe()
        )

    def _hf_batch(
        self, samples: Sequence[MeasurementSample], patches: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute a masked batch only when the caller explicitly opts in."""
        if self.pad_token_id is None and len({s.valid_length for s in samples}) > 1:
            raise ValueError("mixed token lengths require explicit pad_token_id")
        device = self.model.tokens(
            tokens=torch.tensor([samples[0].tokens], dtype=torch.long)
        ).device
        ids = _padded_tokens(
            samples,
            device=device,
            pad_token_id=self.pad_token_id if self.pad_token_id is not None else 0,
        )
        mask = torch.zeros_like(ids)
        for i, sample in enumerate(samples):
            mask[i, : sample.valid_length] = 1
        return self.model.model.run_batch(
            ids,
            layer=self.identity.site.layer,
            positions=[s.patch_position for s in samples],
            readout_positions=[s.readout_index for s in samples],
            attention_mask=mask,
            patches=patches,
        )

    @torch.no_grad()
    def _suffix_baselines(
        self, samples: Sequence[MeasurementSample]
    ) -> tuple[list[FeatureSnapshot], list[torch.Tensor]]:
        """Cache each prompt's block output once and read both the site row and logits."""
        from saebase.api.features import snapshot_from_activation

        snapshots: list[FeatureSnapshot] = []
        baselines: list[torch.Tensor] = []
        self._suffix_hidden = {}
        layer = self.identity.site.layer
        for sample in samples:
            ids = self.model.tokens(tokens=torch.tensor([sample.tokens], dtype=torch.long))
            hidden, logits = self.model.model.cache_site_with_logits(ids, layer)
            self._suffix_hidden[tuple(sample.tokens)] = hidden
            self.counters["suffix_prefix_forwards"] += 1
            self.counters["model_forwards"] += 1
            self.counters["combined_capture_forwards"] += 1
            site = ActivationSite(layer, self.identity.site.hook_name, sample.patch_position)
            activation = hidden[0, sample.patch_position]
            snapshot = snapshot_from_activation(
                self.sae, tokens=ids, site=site, activation=activation
            )
            self.counters["sae_encodes"] += 1
            snapshots.append(snapshot)
            baselines.append(logits[0, sample.readout_index].detach().clone())
        return snapshots, baselines

    @torch.no_grad()
    def _baselines(
        self, samples: Sequence[MeasurementSample]
    ) -> tuple[list[FeatureSnapshot], list[torch.Tensor]]:
        if self._suffix_replay_enabled():
            return self._suffix_baselines(samples)
        # Padding token never read; right-pad positions must be strictly later
        # than each prompt's readout/patch. For TL this is one forward per group.
        if self.model.backend == "transformer_lens":
            if self.pad_token_id is None and len({s.valid_length for s in samples}) > 1:
                raise ValueError("mixed token lengths require explicit pad_token_id")
            ids = self.model.tokens(tokens=torch.tensor([samples[0].tokens], dtype=torch.long))
            padded = _padded_tokens(
                samples,
                device=ids.device,
                pad_token_id=self.pad_token_id if self.pad_token_id is not None else 0,
            )
            needs_capture = any(self.activation_reader is None for _ in samples)
            if needs_capture:
                out, cache = self.model.model.run_with_cache(
                    padded, names_filter=self.identity.site.hook_name
                )
                h_all = cache[self.identity.site.hook_name]
                if h_all.ndim != 3 or tuple(h_all.shape[:2]) != tuple(padded.shape):
                    raise RuntimeError("batched activation hook mismatch")
            else:
                out = self.model.model(padded)
                h_all = None
            logits = getattr(out, "logits", out)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or logits.shape[:2] != padded.shape
            ):
                raise RuntimeError("model batch logits shape mismatch")
            ret = []
            bs = []
            for i, sample in enumerate(samples):
                baseline = logits[i, sample.readout_index].detach().clone()
                cached = self._read_cache(sample) if self.activation_reader is not None else None
                if cached is not None:
                    h, z = cached
                else:
                    h = h_all[i, sample.patch_position].detach().clone()
                    z = self.sae.encode(_sae_input(self.sae, h).unsqueeze(0))[0]
                ln_mu = ln_std = None
                if _sae_uses_layer_norm(self.sae):
                    h_in = _sae_input(self.sae, h).unsqueeze(0)
                    ln_mu, ln_std = _layer_norm_stats(h_in)
                    ln_mu, ln_std = ln_mu.detach().clone(), ln_std.detach().clone()
                reconstruction = _decode(
                    self.sae,
                    z,
                    self.d_in,
                    ln_mu=ln_mu,
                    ln_std=ln_std,
                    activation=h,
                ).to(device=h.device, dtype=h.dtype)
                ret.append(
                    FeatureSnapshot(
                        tokens=torch.tensor(
                            [sample.tokens], device=padded.device, dtype=torch.long
                        ),
                        site=ActivationSite(
                            self.identity.site.layer,
                            self.identity.site.hook_name,
                            sample.patch_position,
                        ),
                        activation=h.detach().clone(),
                        codes=z.detach().clone(),
                        reconstruction=reconstruction.detach().clone(),
                        ln_mu=ln_mu,
                        ln_std=ln_std,
                    )
                )
                bs.append(baseline)
            return ret, bs
        if self.model.backend == "hf_hooks" and self._hf_batch_enabled():
            activations, logits = self._hf_batch(samples)
            snapshots: list[FeatureSnapshot] = []
            baselines: list[torch.Tensor] = []
            for i, sample in enumerate(samples):
                cached = self._read_cache(sample) if self.activation_reader is not None else None
                if cached is None:
                    h = activations[i]
                    z = self.sae.encode(_sae_input(self.sae, h).unsqueeze(0))[0]
                else:
                    h, z = cached
                ln_mu = ln_std = None
                if _sae_uses_layer_norm(self.sae):
                    ln_mu, ln_std = _layer_norm_stats(_sae_input(self.sae, h).unsqueeze(0))
                    ln_mu, ln_std = ln_mu.detach().clone(), ln_std.detach().clone()
                reconstruction = _decode(
                    self.sae,
                    z,
                    self.d_in,
                    ln_mu=ln_mu,
                    ln_std=ln_std,
                    activation=h,
                ).to(device=h.device, dtype=h.dtype)
                snapshots.append(
                    FeatureSnapshot(
                        tokens=torch.tensor([sample.tokens], device=h.device, dtype=torch.long),
                        site=ActivationSite(
                            self.identity.site.layer,
                            self.identity.site.hook_name,
                            sample.patch_position,
                        ),
                        activation=h.detach().clone(),
                        codes=z.detach().clone(),
                        reconstruction=reconstruction.detach().clone(),
                        ln_mu=ln_mu,
                        ln_std=ln_std,
                    )
                )
                baselines.append(logits[i])
            return snapshots, baselines
        # One unpadded forward can return the site activation and the baseline
        # logits. Models without that primitive keep capture plus forward_logits.
        from saebase.api.features import capture_features, snapshot_from_activation

        ret, bs = [], []
        combined = callable(getattr(self.model, "capture_with_logits", None))
        for s in samples:
            ids = self.model.tokens(tokens=torch.tensor([s.tokens], dtype=torch.long))
            site = ActivationSite(
                self.identity.site.layer, self.identity.site.hook_name, s.patch_position
            )
            cached = self._read_cache(s) if self.activation_reader is not None else None
            baseline = None
            if cached is None and combined:
                activation, baseline = self.model.capture_with_logits(
                    tokens=ids, site=site, readout_position=s.readout_index
                )
                snapshot = snapshot_from_activation(
                    self.sae, tokens=ids, site=site, activation=activation
                )
                self.counters["combined_capture_forwards"] += 1
                self.counters["model_forwards"] += 1
                self.counters["sae_encodes"] += 1
            elif cached is None:
                snapshot = capture_features(self.model, self.sae, site=site, tokens=ids)
                self.counters["separate_baseline_forwards"] += 1
                self.counters["model_forwards"] += 1
                self.counters["sae_encodes"] += 1
            else:
                h, z = cached
                ln_mu = ln_std = None
                if _sae_uses_layer_norm(self.sae):
                    h_in = _sae_input(self.sae, h).unsqueeze(0)
                    ln_mu, ln_std = _layer_norm_stats(h_in)
                    ln_mu, ln_std = ln_mu.detach().clone(), ln_std.detach().clone()
                reconstructed = _decode(
                    self.sae,
                    z,
                    self.d_in,
                    ln_mu=ln_mu,
                    ln_std=ln_std,
                    activation=h,
                ).to(device=h.device, dtype=h.dtype)
                snapshot = FeatureSnapshot(
                    ids,
                    site,
                    h,
                    z,
                    reconstructed,
                    ln_mu=ln_mu,
                    ln_std=ln_std,
                )
            if baseline is None:
                baseline = self.model.forward_logits(tokens=ids, readout_position=s.readout_index)
                self.counters["model_forwards"] += 1
            ret.append(snapshot)
            bs.append(baseline)
        return ret, bs

    @torch.no_grad()
    def _suffix_patched_logits(
        self, jobs: Sequence[tuple[MeasurementSample, torch.Tensor]]
    ) -> list[torch.Tensor]:
        """Reuse one clean block output per prompt, then run only the later blocks."""
        grouped: dict[tuple[int, ...], list[int]] = {}
        order: list[tuple[int, ...]] = []
        for index, (sample, _activation) in enumerate(jobs):
            key = tuple(sample.tokens)
            if key not in grouped:
                order.append(key)
                grouped[key] = []
            grouped[key].append(index)
        produced: list[torch.Tensor | None] = [None] * len(jobs)
        layer = self.identity.site.layer
        cached = getattr(self, "_suffix_hidden", {})
        for key in order:
            ids = self.model.tokens(tokens=torch.tensor([key], dtype=torch.long))
            hidden = cached.get(key)
            if hidden is None:
                hidden = self.model.model.cache_site_hidden(ids, layer)
                cached[key] = hidden
                self.counters["suffix_prefix_forwards"] += 1
                self.counters["model_forwards"] += 1
            self._suffix_hidden = cached
            for index in grouped[key]:
                sample, activation = jobs[index]
                if activation.ndim != 1 or activation.shape[0] != hidden.shape[-1]:
                    raise ValueError(
                        "suffix replay activation width does not match the cached site"
                    )
                patched = hidden.clone()
                patched[0, sample.patch_position] = activation.to(
                    device=patched.device, dtype=patched.dtype
                )
                read_offset = int(ids.shape[1] - 1 - sample.readout_index)
                logits = self.model.model.logits_from_site_hidden(
                    ids, layer, patched, readout_offset=read_offset
                )
                self.counters["suffix_replay_forwards"] += 1
                self.counters["patch_forwards"] += 1
                self.counters["model_forwards"] += 1
                produced[index] = logits.detach().clone()
        if any(item is None for item in produced):
            raise RuntimeError("suffix replay did not return one logit vector per edit")
        return produced  # type: ignore[return-value]

    @torch.no_grad()
    def _patched_logits(
        self, jobs: Sequence[tuple[MeasurementSample, torch.Tensor]]
    ) -> list[torch.Tensor]:
        if self._suffix_replay_enabled():
            return self._suffix_patched_logits(jobs)
        if self.model.backend == "hf_hooks" and self._hf_batch_enabled():
            samples = [sample for sample, _ in jobs]
            patches = torch.stack([h for _, h in jobs])
            _, logits = self._hf_batch(samples, patches=patches)
            return [logits[i] for i in range(len(jobs))]
        if self.model.backend != "transformer_lens":
            logits = [
                self.model.patch_logits(
                    tokens=self.model.tokens(tokens=torch.tensor([s.tokens])),
                    site=ActivationSite(
                        self.identity.site.layer, self.identity.site.hook_name, s.patch_position
                    ),
                    activation=h,
                    readout_position=s.readout_index,
                )
                for s, h in jobs
            ]
            self.counters["patch_forwards"] += len(logits)
            self.counters["model_forwards"] += len(logits)
            return logits
        samples = [s for s, _ in jobs]
        device = self.model.tokens(tokens=torch.tensor([samples[0].tokens])).device
        padded = _padded_tokens(samples, device=device, pad_token_id=self.pad_token_id or 0)
        positions = [s.patch_position for s in samples]
        replacements = torch.stack([h.to(device=device) for _, h in jobs])

        def replace(h: torch.Tensor, hook: Any = None) -> torch.Tensor:
            if h.ndim != 3 or tuple(h.shape[:2]) != tuple(padded.shape) or h.shape[-1] != self.d_in:
                raise RuntimeError("batched patch hook shape or SAE width differs")
            changed = h.clone()
            rows = torch.arange(len(jobs), device=h.device)
            changed[rows, torch.tensor(positions, device=h.device)] = replacements.to(
                device=h.device, dtype=h.dtype
            )
            return changed

        out = self.model.model.run_with_hooks(
            padded, fwd_hooks=[(self.identity.site.hook_name, replace)]
        )
        logits = getattr(out, "logits", out)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != padded.shape
        ):
            raise RuntimeError("batched patch logits shape mismatch")
        return [logits[i, s.readout_index].detach().clone() for i, s in enumerate(samples)]

    def _baselines_split(
        self, active: Sequence[MeasurementSample]
    ) -> tuple[tuple[list[FeatureSnapshot], list[torch.Tensor]], int]:
        """Split a baseline/capture batch after OOM. One-example failure still raises."""
        try:
            return self._baselines(active), 0
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_oom(exc) or len(active) == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            middle = len(active) // 2
            left, left_retries = self._baselines_split(active[:middle])
            right, right_retries = self._baselines_split(active[middle:])
            return (left[0] + right[0], left[1] + right[1]), left_retries + right_retries + 1

    @torch.no_grad()
    def run(
        self,
        samples: Sequence[MeasurementSample],
        conditions: Sequence[Condition],
        writer: ResultWriter,
        *,
        sample_batch: int = 8,
        microbatch: int = 16,
        budget: RunBudget | None = None,
        verify_identity: bool = True,
        tolerance: float = 1e-4,
    ) -> ExecutionStats:
        dry_run(samples, conditions, microbatch=microbatch, budget=budget)
        if self.release_content_check:
            if self.identity_lease is not None:
                self.identity_lease.strict_recheck()
            else:
                model_sha = sha256_state_dict(self.model.model)
                sae_sha = sha256_state_dict(self.sae)
                if (
                    model_sha != self.identity.model_state_sha256
                    or sae_sha != self.identity.sae_state_sha256
                ):
                    raise RuntimeError("content checksum differs from the experiment identity")
        if writer.identity.key != self.identity.key:
            raise ValueError("result store experiment identity mismatch")
        if type(sample_batch) is not int or sample_batch < 1:
            raise ValueError("sample_batch must be positive")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        started = perf_counter()
        done, reused, baselines, patches, retries, peak = 0, 0, 0, 0, 0, 0
        safe_microbatch = microbatch
        for offset in range(0, len(samples), sample_batch):
            group = []
            for sample in samples[offset : offset + sample_batch]:
                needed = []
                for condition in conditions:
                    mid = measurement_id(self.identity, sample, condition)
                    if writer.has(mid):
                        reused += 1
                    else:
                        needed.append((condition, mid))
                if needed:
                    group.append((sample, needed))
            if not group:
                continue
            if (
                budget is not None
                and budget.max_seconds is not None
                and perf_counter() - started > budget.max_seconds
            ):
                raise TimeoutError("batch measurement elapsed time exceeds run budget")
            # These baseline examples include diverse lengths. The TL backend
            # runs one batch forward and one batch capture where needed.
            active = [x[0] for x in group]
            (snapshots, ref_logits), baseline_retries = self._baselines_split(active)
            retries += baseline_retries
            if baseline_retries:
                safe_microbatch = min(safe_microbatch, max(1, len(active) // 2))
            baselines += (
                1
                if self.model.backend == "transformer_lens" or self._hf_batch_enabled()
                else len(active)
            )
            if verify_identity:
                # Replay at the baseline forward's width. A narrower microbatch
                # changes the GEMM reduction and is not an identity-patch error.
                pending: list[
                    tuple[list[tuple[MeasurementSample, FeatureSnapshot]], list[torch.Tensor]]
                ] = [(list(zip(active, snapshots)), list(ref_logits))]
                while pending:
                    subset, expected_logits = pending.pop()
                    try:
                        replay = self._patched_logits([(s, snap.activation) for s, snap in subset])
                        patches += (
                            1
                            if self.model.backend == "transformer_lens" or self._hf_batch_enabled()
                            else len(subset)
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                        if not _is_oom(exc):
                            raise
                        if len(subset) == 1:
                            raise
                        middle = len(subset) // 2
                        retries += 1
                        safe_microbatch = min(safe_microbatch, middle)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        left, right = subset[:middle], subset[middle:]
                        for half in (right, left):
                            half_samples = [sample for sample, _ in half]
                            half_snaps, half_logits = self._baselines(half_samples)
                            pending.append((list(zip(half_samples, half_snaps)), list(half_logits)))
                        continue
                    for expected, actual in zip(expected_logits, replay):
                        if not torch.allclose(
                            expected.float(), actual.float(), rtol=1e-5, atol=tolerance
                        ):
                            error = float((expected.float() - actual.float()).abs().max().item())
                            raise RuntimeError(f"identity patch mismatch; max logits error {error}")
            all_jobs: list[
                tuple[
                    MeasurementSample, Condition, str, FeatureSnapshot, torch.Tensor, torch.Tensor
                ]
            ] = []
            for (sample, jobs), snapshot, baseline in zip(group, snapshots, ref_logits):
                for condition, mid in jobs:
                    patch = apply_condition(self.sae, snapshot, condition)
                    delta = patch.activation.to(snapshot.activation.device) - snapshot.activation
                    all_jobs.append((sample, condition, mid, snapshot, baseline, patch.activation))
                    if (
                        condition.operation == "identity"
                        and float(torch.linalg.vector_norm(delta).item()) > tolerance
                    ):
                        raise RuntimeError("identity SAE edit does not preserve activation")
            i = 0
            while i < len(all_jobs):
                width = min(safe_microbatch, len(all_jobs) - i)
                while True:
                    chunk = all_jobs[i : i + width]
                    try:
                        updated = self._patched_logits([(job[0], job[5]) for job in chunk])
                        patches += (
                            1
                            if self.model.backend == "transformer_lens" or self._hf_batch_enabled()
                            else len(chunk)
                        )
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                        if not _is_oom(exc):
                            raise
                        if width == 1:
                            raise
                        width = max(1, width // 2)
                        retries += 1
                        safe_microbatch = min(safe_microbatch, width)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                peak = max(peak, len(chunk))
                for job, edited in zip(chunk, updated):
                    sample, condition, mid, snapshot, baseline, activation = job
                    delta = activation.to(snapshot.activation.device) - snapshot.activation
                    same_width = len(chunk) == len(active)
                    if (
                        condition.operation == "identity"
                        and same_width
                        and not torch.allclose(
                            baseline.float(), edited.float(), rtol=1e-5, atol=tolerance
                        )
                    ):
                        raise RuntimeError("identity patch changes model logits")
                    result = {
                        "schema": "saebase.m2.measurement.v1",
                        "measurement_identity": {
                            "schema": "saebase.m2.measurement.v1",
                            "experiment": self.identity.key,
                            "sample": sample.to_dict(),
                            "condition": condition.to_dict(),
                        },
                        "measurement_id": mid,
                        "sample_id": sample.record.sample_id,
                        "condition_name": condition.name,
                        "operator": condition.operation,
                        "patch_position": sample.patch_position,
                        "readout_position": sample.readout_index,
                        "model_backend": self.model.backend,
                        "metrics": _metrics(
                            baseline,
                            edited,
                            sample,
                            float(torch.linalg.vector_norm(delta.float()).item()),
                        ),
                    }
                    if writer.append(result):
                        done += 1
                i += len(chunk)
            writer.flush()  # committed at least once per sample batch for resume.
        elapsed = perf_counter() - started
        return ExecutionStats(
            done, reused, baselines, patches, retries, peak, elapsed, self.model.backend
        )
