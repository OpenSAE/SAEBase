"""Canonical teacher-forced answer readout shared by Capture behavior modes."""

import torch
import inspect


class ContextTailScorer:
    """Score the full sequence after a frozen layer, retaining native tensor shapes."""

    def __init__(self, model, hidden, answer_tokens, prompt_length, start_layer):
        self.model, self.hidden, self.answer = model, hidden, answer_tokens
        self.position, self.start_layer = prompt_length - 1, start_layer

    @torch.no_grad()
    def __call__(self, target):
        hidden = self.hidden.clone()
        hidden[0, self.position] = target
        logits = self.model(hidden, start_at_layer=self.start_layer)
        values = logits[0, self.position :].float().log_softmax(-1)
        values = values.gather(1, self.answer[0, :, None]).squeeze(1)
        return float(values.double().mean())


def answer_patch(original_activation, readout_activation, edited_activation):
    delta = edited_activation.float().to(
        readout_activation.device
    ) - original_activation.float().to(readout_activation.device)
    return (readout_activation.float() + delta).to(readout_activation.dtype)


@torch.no_grad()
def prepare_answer_readout(
    pair, snapshot, prompt, readout, *, feature_ids, answer_backend="full_sequence"
):
    from saebase.api import UnsupportedCapability
    import torch

    if snapshot.site.position not in (-1, snapshot.tokens.shape[1] - 1):
        raise ValueError("Answer SAR captures and patches the final prompt token")
    tokenizer = getattr(pair.model.model, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise UnsupportedCapability("Answer scoring requires the selected model tokenizer")
    reference = readout["reference"]
    original_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    observed = snapshot.tokens[0].detach().cpu().tolist()
    extra = len(observed) - len(original_prompt_ids)
    specials = set(getattr(tokenizer, "all_special_ids", []))
    if (
        extra < 0
        or observed[extra:] != original_prompt_ids
        or any(token not in specials for token in observed[:extra])
    ):
        raise UnsupportedCapability(
            "Canonical answer SAR requires the complete original prompt; tokenizer truncation or rewriting was observed"
        )
    answer_ids = tokenizer.encode(reference["prefix"] + reference["text"], add_special_tokens=False)
    if not answer_ids:
        raise ValueError("Reference answer encoded to zero tokens")
    answer_tokens = torch.tensor([answer_ids], dtype=torch.long, device=snapshot.tokens.device)
    from saebase.api import ActivationSite

    scoring_tokens = torch.cat((snapshot.tokens, answer_tokens[:, :-1]), dim=1)
    hidden = None
    if (
        answer_backend == "full_context_tail"
        and pair.model.backend == "transformer_lens"
        and snapshot.site.hook_name == f"blocks.{snapshot.site.layer}.hook_resid_post"
        and {"start_at_layer", "stop_at_layer"}.issubset(
            inspect.signature(pair.model.model.forward).parameters
        )
    ):
        hidden = pair.model.model(scoring_tokens, stop_at_layer=snapshot.site.layer + 1).detach()
    readout_activation = (
        hidden[0, snapshot.tokens.shape[1] - 1].detach()
        if hidden is not None
        else pair.model.capture(
            tokens=scoring_tokens,
            site=ActivationSite(
                snapshot.site.layer, snapshot.site.hook_name, snapshot.tokens.shape[1] - 1
            ),
        )
    )
    readout.update(
        answer_token_ids=answer_ids,
        answer_token_count=len(answer_ids),
        patch_position=snapshot.tokens.shape[1] - 1,
        tokenization="append separately tokenized canonical continuation; no BOS/EOS",
        accumulation="float32 log_softmax; float64 mean",
        patch_application="native feature-edit delta added to the unmodified answer-context activation",
    )

    def activation_readout(activation):
        target = None
        if activation is not None:
            target = answer_patch(snapshot.activation, readout_activation, activation)
        values = pair.model.answer_token_logprobs(
            prompt_tokens=snapshot.tokens,
            answer_tokens=answer_tokens,
            site=snapshot.site,
            activation=target,
        )
        return values.double().mean()

    readout["execution"] = "full_sequence"
    if answer_backend == "full_context_tail" and hidden is None:
        readout["cache_fallback"] = (
            "The selected backend/site does not expose complete-context tail execution"
        )
    if hidden is not None:
        from saebase.api import edit_features

        tail = ContextTailScorer(
            pair.model.model,
            hidden,
            answer_tokens.to(hidden.device),
            snapshot.tokens.shape[1],
            snapshot.site.layer + 1,
        )

        def cached(activation):
            return tail(
                answer_patch(
                    snapshot.activation,
                    readout_activation,
                    snapshot.activation if activation is None else activation,
                )
            )

        removed = edit_features(
            pair.sae, snapshot, replacements={int(i): 0.0 for i in feature_ids}
        ).activation
        error = max(
            abs(float(activation_readout(value)) - cached(value))
            for value in [snapshot.activation, removed]
        )
        if error > 1e-7:
            raise ValueError(f"Full-context tail differs from native full answer scoring: {error}")
        readout.update(execution="full_context_tail", cache_validation={"max_abs_error": error})
        return cached, answer_ids, readout_activation
    if answer_backend == "auto":
        from saebase.api.answer_cache import FrozenPrefixAnswerReadout
        from saebase.api import edit_features

        candidate = None
        try:
            candidate = FrozenPrefixAnswerReadout(
                pair.model,
                snapshot.tokens,
                answer_tokens,
                snapshot.site,
                reference_activation=snapshot.activation,
            )
            selected = feature_ids
            removed = edit_features(
                pair.sae, snapshot, replacements={i: 0.0 for i in selected}
            ).activation
            validation = candidate.verify(
                [snapshot.activation, removed], reference_readout=activation_readout
            )
        except UnsupportedCapability as exc:
            readout["cache_fallback"] = str(exc)
            del candidate
        else:
            activation_readout = candidate
            readout.update(execution="frozen_causal_prefix", cache_validation=validation)
    return activation_readout, answer_ids, readout_activation
