"""Shared tokenizer-aware QA input construction, independent of experiments."""

from __future__ import annotations
import hashlib
from typing import Any
import torch
from saebase.data.qa_records import render_prompt


def _prompt_format(model_key: str) -> str:
    """Read the model implementation's own prompt contract, never a QA500 list."""
    import importlib

    if not model_key or not model_key.replace("_", "").isalnum():
        raise ValueError("Invalid model registry key")
    model_module = importlib.import_module(f"models.{model_key}.{model_key}")
    spec = model_module.get_spec()
    if str(spec.key) != model_key:
        raise ValueError("Model prompt profile identity does not match requested model")
    mode = str(spec.prompt_format)
    if mode not in {"plain", "chat_template"}:
        raise ValueError("Unknown model prompt-format policy")
    return mode


def _tokenizer(pair: Any) -> Any:
    tokenizer = getattr(pair.model.model, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise RuntimeError("A real, accessible tokenizer is required for QA scoring")
    return tokenizer


def make_input(pair: Any, record: dict[str, Any], model_key: str) -> tuple[Any, dict[str, Any]]:
    """Render model-appropriate prompt, return verified token IDs and provenance."""
    tokenizer = _tokenizer(pair)
    prompt = render_prompt(record)
    if _prompt_format(model_key) == "chat_template":
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

    digest = hashlib.sha256(
        tokens.detach().cpu().numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return tokens, {
        "input_mode": mode,
        "prompt_sha256": record["prompt_sha256"],
        "token_ids_sha256": digest,
        "token_count": int(tokens.size(1)),
    }
