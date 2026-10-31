"""Frozen QA500 model input, answer scoring and generation.

All public metrics are produced by actual model logits. The gold answer is
never included in model tokens. Candidate label likelihood and generated
answer accuracy use different scorer IDs and are never mixed.
"""

from __future__ import annotations

import math
import re
import string
from decimal import Decimal, InvalidOperation
from typing import Any

import torch

from saebase.datasets.qa_standard import render_prompt

# Fixed, conservative default budgets. Large generations require a separate run.
MAX_GENERATION = {"number": 48, "math": 96, "exact_text": 48}
INSTRUCTION_MODELS = frozenset({"gemma3_4b_it"})


def _tokenizer(pair: Any) -> Any:
    tokenizer = getattr(pair.model.model, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise RuntimeError("A real, accessible tokenizer is required for QA scoring")
    return tokenizer


def make_input(pair: Any, record: dict[str, Any], model_key: str) -> tuple[Any, dict[str, Any]]:
    """Render model-appropriate prompt, return verified token IDs and provenance."""
    tokenizer = _tokenizer(pair)
    prompt = render_prompt(record)
    if model_key in INSTRUCTION_MODELS:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise RuntimeError("Instruction-tuned model has no chat template")
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if isinstance(encoded, list):
            encoded = torch.tensor([encoded], dtype=torch.long)
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2:
            raise ValueError("Chat template returned invalid token IDs")
        tokens = pair.model.tokens(tokens=encoded.long())
        mode = "tokenizer_chat_template"
    else:
        tokens = pair.model.tokens(prompt=prompt)
        mode = "base_plain_zeroshot"
    if tokens.ndim != 2 or tokens.size(0) != 1 or tokens.size(1) < 2:
        raise ValueError("Model input must contain a nonempty [1, seq] token tensor")
    if not bool((tokens >= 0).all()):
        raise ValueError("Negative input token IDs")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size and bool((tokens >= int(vocab_size)).any()):
        # Some tokenizers have additional special tokens beyond vocab_size.
        if not hasattr(tokenizer, "__len__") or bool((tokens >= len(tokenizer)).any()):
            raise ValueError("Input token ID exceeds tokenizer vocabulary")
    import hashlib

    digest = hashlib.sha256(
        tokens.detach().cpu().numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return tokens, {
        "input_mode": mode,
        "prompt_sha256": record["prompt_sha256"],
        "token_ids_sha256": digest,
        "token_count": int(tokens.size(1)),
    }


def choice_label_tokens(tokenizer: Any, label: str) -> list[int]:
    """Tokenize an explicit completion, never the gold reference as prompt."""
    ids = tokenizer.encode(" " + label, add_special_tokens=False)
    if not ids or not all(isinstance(k, int) and k >= 0 for k in ids):
        raise ValueError(f"Tokenization of candidate {label} failed")
    return list(ids)


@torch.no_grad()
def score_choices(
    pair: Any,
    tokens: torch.Tensor,
    labels: list[str],
    *,
    clean_logits: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score complete candidate letter sequences by conditional log likelihood.

    Single-token candidates share one clean next-token forward. Multi-token
    candidates use explicit teacher forcing, never a different denominator.
    """
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Expected nonempty unique choice labels")
    tok = _tokenizer(pair)
    encoded = {x: choice_label_tokens(tok, x) for x in labels}
    if clean_logits is None:
        clean_logits = pair.model.forward_logits(tokens=tokens)
    if not bool(torch.isfinite(clean_logits).all()):
        raise ValueError("Nonfinite baseline model logits")
    first = torch.log_softmax(clean_logits.float(), dim=-1)
    values: dict[str, float] = {}
    vocab = int(first.numel())
    for label, suffix in encoded.items():
        if any(x >= vocab for x in suffix):
            raise ValueError("Choice token ID outside model output vocabulary")
        value = float(first[suffix[0]].item())
        if len(suffix) > 1:
            # Each successive token is conditioned on the prior candidate tokens.
            for i in range(1, len(suffix)):
                prefix = torch.cat(
                    [tokens, torch.tensor([suffix[:i]], device=tokens.device, dtype=torch.long)],
                    dim=1,
                )
                next_logits = pair.model.forward_logits(tokens=prefix)
                log_probs = torch.log_softmax(next_logits.float(), dim=-1)
                value += float(log_probs[suffix[i]].item())
        if not math.isfinite(value):
            raise ValueError("Nonfinite QA choice likelihood")
        values[label] = value
    # Sum of log probabilities without length normalization is an explicit,
    # reproducible protocol; a longer answer should not be rewarded by average.
    predicted = sorted(values, key=lambda label: (-values[label], label))[0]
    return {
        "prediction": predicted,
        "choice_logprobs": values,
        "scorer_id": "candidate_sequence_logprob_v1",
    }


@torch.no_grad()
def greedy_answer(
    pair: Any,
    tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> dict[str, Any]:
    """Deterministic, backend-neutral greedy autoregressive generation."""
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("Unapproved QA generation budget")
    tokenizer = _tokenizer(pair)
    prefix = tokens
    generated: list[int] = []
    reason = "max_new_tokens"
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = pair.model.forward_logits(tokens=prefix)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite logits during QA generation")
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            if eos_token_id is not None and next_id == int(eos_token_id):
                reason = "eos"
                break
            new_token = torch.tensor([[next_id]], dtype=torch.long, device=prefix.device)
            prefix = torch.cat([prefix, new_token], dim=1)
    response = tokenizer.decode(generated, skip_special_tokens=True)
    if not isinstance(response, str):
        raise ValueError("Tokenizer.decode returned a non-string answer")
    return {
        "prediction": response.strip(),
        "generated_token_ids": generated,
        "stop_reason": reason,
        "scorer_id": "greedy_text_v1",
    }


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _parse_number(text: str) -> Decimal | None:
    raw = text.strip().replace(",", "")
    boxed = re.search(r"\\boxed\s*\{([^{}]+)\}", raw)
    if boxed:
        raw = boxed.group(1)
    if "####" in raw:
        raw = raw.rsplit("####", 1)[1].strip()
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:/\d+)?", raw)
    if not match:
        return None
    s = match.group(0)
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return Decimal(a) / Decimal(b)
        return Decimal(s)
    except (InvalidOperation, ZeroDivisionError):
        return None


def grade_answer(answer_type: str, predicted: str, gold: str) -> dict[str, Any]:
    """Basic QA grading. Math is deliberately provisional without math-verify."""
    if answer_type in {"choice", "boolean"}:
        valid = _normalize_text(predicted)
        expected = _normalize_text(gold)
        correct = valid == expected
        method = "strict_label_v1"
    elif answer_type == "number":
        actual, expected = _parse_number(predicted), _parse_number(gold)
        correct = actual is not None and expected is not None and actual == expected
        method = "numeric_final_v1"
    elif answer_type == "math":
        # A mathematical equivalence scorer must be separately installed and
        # validated. Do not call string equality a standard MATH500 accuracy.
        return {
            "correct": None,
            "grade_status": "needs_math_equivalence",
            "grader_id": "math_equivalence_unavailable",
        }
    else:
        correct = _normalize_text(predicted) == _normalize_text(gold)
        method = "normalized_exact_match_v1"
    return {"correct": bool(correct), "grade_status": "scored", "grader_id": method}


def evaluate_qa(
    pair: Any,
    tokens: torch.Tensor,
    record: dict[str, Any],
    *,
    clean_logits: torch.Tensor | None = None,
) -> dict[str, Any]:
    kind = record["answer_type"]
    if kind in {"choice", "boolean"}:
        labels = (
            [item["label"] for item in record["choices"]] if kind == "choice" else ["Yes", "No"]
        )
        pred = score_choices(pair, tokens, labels, clean_logits=clean_logits)
    else:
        pred = greedy_answer(
            pair,
            tokens,
            max_new_tokens=MAX_GENERATION[kind],
            eos_token_id=getattr(_tokenizer(pair), "eos_token_id", None),
        )
    scored = grade_answer(kind, pred["prediction"], record["answer"])
    return {**pred, **scored, "answer_type": kind}
