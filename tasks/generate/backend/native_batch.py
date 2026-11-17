"""Native batched full-prefix generation for independent SAE conditions.

Uses each trace's own prefix and original seed. No edited KV cache is shared.
The original scalar implementation remains the acceptance oracle in Precheck.
"""

import hashlib
import torch
from saebase.api import ActivationSite
from saebase.api.features import snapshot_from_activation
from saebase.api.replay import observe_batch, patch_batch
from tasks.intervene.backend.operators import apply_condition
from tasks.intervene.backend.steering import _sample, _distribution_shift


@torch.inference_mode()
def compare_many(pair, sample, conditions, spec, batch_size):
    original = list(sample.tokens)
    token_device = pair.model.tokens(tokens=torch.tensor([original], dtype=torch.long)).device
    native_limit = getattr(pair.model.model.cfg, "n_ctx", None)

    def rng():
        material = hashlib.sha256(f"{spec.seed}:{sample.record.key}".encode()).digest()
        return torch.Generator(device="cpu").manual_seed(
            int.from_bytes(material[:8], "little") % (2**63 - 1)
        )

    baseline = []
    prefix = original.copy()
    generator = rng()
    baseline_stop = "max_new_tokens"
    for _ in range(spec.max_new_tokens):
        if (
            spec.stop_at_context_limit
            and isinstance(native_limit, int)
            and len(prefix) > native_limit
        ):
            baseline_stop = "context_limit"
            break
        ids = pair.model.tokens(tokens=torch.tensor([prefix], dtype=torch.long))
        value = pair.model.forward_logits(tokens=ids, readout_position=-1)
        token = _sample(value, spec, rng=generator)
        baseline.append(token)
        prefix.append(token)
        if token in spec.eos_token_ids:
            baseline_stop = "eos"
            break
    results = []
    for start in range(0, len(conditions), batch_size):
        group = conditions[start : start + batch_size]
        prefixes = [original.copy() for _ in group]
        generated = [[] for _ in group]
        steps = [[] for _ in group]
        generators = [rng() for _ in group]
        stops = ["max_new_tokens" for _ in group]
        active = list(range(len(group)))
        for step in range(spec.max_new_tokens):
            if not active:
                break
            if (
                spec.stop_at_context_limit
                and isinstance(native_limit, int)
                and len(prefixes[active[0]]) > native_limit
            ):
                for index in active:
                    stops[index] = "context_limit"
                break
            ids = torch.tensor(
                [prefixes[index] for index in active],
                dtype=torch.long,
                device=token_device,
            )
            site = ActivationSite(pair.site.layer, pair.site.hook_name, ids.shape[1] - 1)
            hidden, clean = observe_batch(pair.model, ids, site)
            snapshots = [
                snapshot_from_activation(
                    pair.sae, tokens=ids[row : row + 1], site=site, activation=hidden[row]
                )
                for row in range(len(active))
            ]
            patched = torch.stack(
                [
                    apply_condition(pair.sae, snapshot, group[index]).activation
                    for snapshot, index in zip(snapshots, active)
                ]
            )
            logits = patch_batch(pair.model, ids, site, patched)
            continuing = []
            for row, index in enumerate(active):
                feature = group[index].feature_ids[0] if group[index].feature_ids else None
                activation = float(snapshots[row].codes[feature]) if feature is not None else None
                effect = _distribution_shift(
                    clean[row], logits[row], feature_id=feature, activation=activation
                )
                effect.update(step=step, position=len(prefixes[index]) - 1)
                steps[index].append(effect)
                token = _sample(logits[row], spec, rng=generators[index])
                generated[index].append(token)
                prefixes[index].append(token)
                if token in spec.eos_token_ids:
                    stops[index] = "eos"
                else:
                    continuing.append(index)
            active = continuing
        for index in range(len(group)):
            results.append(
                {
                    "baseline_generated_token_ids": baseline.copy(),
                    "steered_generated_token_ids": generated[index],
                    "baseline_stop_reason": baseline_stop,
                    "steered_stop_reason": stops[index],
                    "steered_steps": steps[index],
                    "same_continuation": baseline == generated[index],
                    "shared_prefix_tokens": len(original),
                    "first_different_generated_index": next(
                        (i for i, (a, b) in enumerate(zip(baseline, generated[index])) if a != b),
                        None,
                    ),
                }
            )
    return results
