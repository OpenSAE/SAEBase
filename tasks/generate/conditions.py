"""Generate and compare configured feature-edit conditions on one prompt.

Owner: generate. Reuses native Condition/GenerationSpec and the steering backend.
Automatic selection is explicit; the default generate Task keeps its configured feature.
"""

from __future__ import annotations
import torch
from saebase.api import ActivationSite
from tasks.intervene.backend.specs import Condition, MeasurementSample
from tasks.intervene.backend.steering import GenerationSpec, compare_steered_generation
from saebase.store.identity import SampleRecord, TokenSelection

SPECIAL = {"<bos>", "<eos>", "<pad>", "<unk>", "<start_of_turn>", "<end_of_turn>"}


def _sample(model, prompt: str, sample_id: str) -> tuple[MeasurementSample, list[str]]:
    token_ids = model.tokens(prompt=prompt)
    ids = tuple(int(value) for value in token_ids[0].tolist())
    tokenizer = model.model.tokenizer
    texts = tokenizer.convert_ids_to_tokens(list(ids))
    record = SampleRecord(
        sample_id=sample_id,
        token_ids=ids,
        selection=TokenSelection(mode="last_valid"),
        prompt_group_id=sample_id,
        split="discovery",
    )
    return MeasurementSample(record=record), [str(text) for text in texts]


def _active_feature(
    model, sae, site_hook: str, layer: int, sample: MeasurementSample, texts: list[str]
) -> tuple[int, int, float]:
    ids = torch.tensor([sample.tokens], dtype=torch.long)
    for position in range(len(sample.tokens) - 1, -1, -1):
        if texts[position] in SPECIAL:
            continue
        site = ActivationSite(layer, site_hook, position)
        from saebase.api import capture_features

        snapshot = capture_features(model, sae, site=site, tokens=ids)
        active = torch.nonzero(snapshot.codes, as_tuple=True)[0]
        if active.numel() == 0:
            continue
        values = snapshot.codes[active].abs()
        choice = int(active[int(torch.argmax(values).item())].item())
        return position, choice, float(snapshot.codes[choice].detach().float().item())
    raise RuntimeError("No active SAE feature at a content token")


def compare_prompt_conditions(
    pair,
    prompt: str,
    sample_id: str,
    max_new_tokens: int,
    scale: float = 2.0,
    *,
    feature_id=None,
    conditions=None,
    seed: int = 0,
    temperature: float = 0.0,
    stop_at_context_limit: bool = False,
    measured_runs=None,
    original_snapshot=None,
) -> dict:
    sample, texts = _sample(pair.model, prompt, sample_id)
    if feature_id is None:
        position, feature_id, activation = _active_feature(
            pair.model, pair.sae, pair.identity.hook_name, pair.identity.layer, sample, texts
        )
    else:
        from saebase.api import capture_features

        snapshot = original_snapshot or capture_features(
            pair.model, pair.sae, site=pair.site, prompt=prompt
        )
        position, activation = -1, float(snapshot.codes[feature_id].float().item())
    site = ActivationSite(pair.identity.layer, pair.identity.hook_name, position)
    spec = GenerationSpec(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        stop_at_context_limit=stop_at_context_limit,
    )
    conditions = (
        [
            Condition(
                name=row["name"],
                operation=row["operation"],
                feature_ids=tuple(
                    row.get(
                        "feature_ids",
                        () if row["operation"] in {"identity", "reconstruction"} else (feature_id,),
                    )
                ),
                value=float(row.get("value", 0.0)),
            )
            for row in conditions
        ]
        if conditions
        else [
            Condition(name="noop", operation="identity"),
            Condition(name="ablate", operation="ablate", feature_ids=(feature_id,)),
            Condition(name="scale", operation="scale", feature_ids=(feature_id,), value=scale),
        ]
    )
    decode = pair.model.model.tokenizer.decode
    runs = []
    for index, condition in enumerate(conditions):
        result = (
            measured_runs[index]
            if measured_runs is not None
            else compare_steered_generation(
                pair.model, pair.sae, site=site, sample=sample, condition=condition, spec=spec
            )
        )
        runs.append(
            {
                "condition": condition.to_dict(),
                "same_continuation": result["same_continuation"],
                "first_different_generated_index": result["first_different_generated_index"],
                "baseline_generated_token_ids": result["baseline_generated_token_ids"],
                "edited_generated_token_ids": result["steered_generated_token_ids"],
                "baseline_continuation": decode(
                    result["baseline_generated_token_ids"], skip_special_tokens=False
                ),
                "edited_continuation": decode(
                    result["steered_generated_token_ids"], skip_special_tokens=False
                ),
                "baseline_stop_reason": result["baseline_stop_reason"],
                "edited_stop_reason": result["steered_stop_reason"],
                "noop_equal_baseline": result["same_continuation"]
                if condition.operation == "identity"
                else None,
                "steps": result.get("steered_steps", []),
            }
        )
    return {
        "sample_id": sample_id,
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "site_id": f"{pair.identity.model_key}:{pair.identity.sae_key}:L{pair.identity.layer}",
        "hook_name": pair.identity.hook_name,
        "feature_position": position,
        "feature_id": feature_id,
        "feature_activation": activation,
        "protocol": "greedy_full_prefix_recompute",
        "max_new_tokens": max_new_tokens,
        "generated_text": True,
        "runs": runs,
    }
