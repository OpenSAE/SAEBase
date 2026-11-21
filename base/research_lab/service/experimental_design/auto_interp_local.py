"""Provide local Lab execution and its explicit command-line operation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tasks.explain.interpretation import propose_interpretation


class LocalAutoInterpreter:
    """Turns a loaded local autoregressive instruct model into real callbacks."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_name: str,
        model_sha256: str,
        device: str,
        max_input_tokens: int = 2048,
    ) -> None:
        if not callable(getattr(model, "generate", None)):
            raise TypeError("A real locally loaded causal LM with generate() is required")
        if not callable(tokenizer) or not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("A matching source tokenizer is required")
        if (
            not model_name
            or len(model_sha256) != 64
            or any(x not in "0123456789abcdef" for x in model_sha256)
        ):
            raise ValueError("Local generation must have full model source identity")
        if not 64 <= max_input_tokens <= 4096:
            raise ValueError("Explicit local input budget required")
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_sha256 = model_sha256
        self.device = device
        self.max_input_tokens = max_input_tokens

    @classmethod
    def from_local_snapshot(
        cls,
        path: Path,
        *,
        device: str = "cpu",
        max_input_tokens: int = 2048,
    ) -> "LocalAutoInterpreter":
        """No Hub download; read existing snapshot using HF safe tensors only."""
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError("An already-downloaded local HF checkpoint is required")
        if device not in {"cpu", "cuda"}:
            raise ValueError("Only explicit local CPU/GPU devices are supported")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Selected local CUDA device is unavailable")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from saebase.store import sha256_state_dict

        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                torch_dtype=torch.float32,
            )
            .to(device)
            .eval()
        )
        digest = sha256_state_dict(model)
        return cls(
            model,
            tokenizer,
            model_name=str(path.resolve()),
            model_sha256=digest,
            device=device,
            max_input_tokens=max_input_tokens,
        )

    @torch.no_grad()
    def _ask(self, text: str, *, max_new_tokens: int) -> str:
        if not 1 <= max_new_tokens <= 100 or len(text) > 18000:
            raise ValueError("Interpretation prompt exceeds its local budget")
        chat = getattr(self.tokenizer, "apply_chat_template", None)
        encoded = None
        if callable(chat):
            templated = chat(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(templated, torch.Tensor):
                encoded = {"input_ids": templated}
            elif isinstance(templated, Mapping) and "input_ids" in templated:
                encoded = templated
        if encoded is None:
            encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise TypeError("Original tokenizer did not return input_ids")
        ids = encoded["input_ids"]
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Expected exact [1,T] locally tokenized context")
        if ids.shape[1] > self.max_input_tokens:
            raise ValueError("Interpretation prompt exceeds max_input_tokens")
        inputs = {
            key: tensor.to(self.device)
            for key, tensor in encoded.items()
            if isinstance(tensor, torch.Tensor)
        }
        eos = getattr(self.tokenizer, "eos_token_id", None)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=eos,
        )
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
            raise RuntimeError("Original local generator did not return token IDs")
        if output.shape[1] <= ids.shape[1]:
            raise RuntimeError("Local language model generated no response")
        return self.tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True).strip()

    def _candidate(self, contexts: list[str]) -> str:
        if not 1 <= len(contexts) <= 32:
            raise ValueError("Need 1–32 genuine activating contexts")
        content = "\n".join(f"{i + 1}. {value[:1200]}" for i, value in enumerate(contexts))
        question = (
            "You are interpreting the observed activity of one sparse autoencoder "
            "feature. The following real text contexts activate that feature. "
            "Identify a short, concrete shared pattern. Do not infer unsupported "
            "causal claims. Return only one sentence describing the pattern.\n"
            "Observed contexts:\n" + content + "\nDescription:"
        )
        answer = self._ask(question, max_new_tokens=72)
        if not answer or len(answer) > 900:
            raise ValueError("No bounded feature hypothesis produced")
        return answer

    def _predictor(self, description: str, context: str) -> bool:
        question = (
            "Does this text clearly exhibit the described pattern? "
            "Return exactly YES or NO, and nothing else.\n"
            "Pattern: " + description[:900] + "\nText: " + context[:1400] + "\nAnswer:"
        )
        response = self._ask(question, max_new_tokens=6)
        word = "".join(ch for ch in response.upper() if ch.isalpha() or ch.isspace()).split()
        label = word[0] if word else ""
        if label not in {"YES", "NO"}:
            raise ValueError("Held-out predictor returned neither YES nor NO: " + response[:80])
        return label == "YES"

    def interpret(
        self,
        *,
        site_id: str,
        feature_id: int,
        discovery: list[dict[str, Any]],
        heldout: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Real candidate + held-out balanced accuracy, but never auto-approve."""
        result = propose_interpretation(
            site_id=site_id,
            feature_id=feature_id,
            discovery=discovery,
            heldout=heldout,
            generator=self._candidate,
            generator_model=self.model_name,
            generator_method="local_greedy_context_pattern_v1",
            authorized_budget_usd=0.0,
            estimated_cost_usd=0.0,
            predictor=self._predictor,
        )
        result["generator_state_sha256"] = self.model_sha256
        result["compute_mode"] = "local_original_model_forward"
        result["approval_required"] = True
        result["private_unpublished"] = True
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-model", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    from models.loading_policy import resolve_device

    args.device = resolve_device(args.device)
    if args.output.exists():
        parser.error("Never overwrite a previously generated explanation candidate")
    if not args.examples.is_file() or args.examples.stat().st_size > 1_000_000:
        parser.error("A bounded local reviewed source examples JSON is required")
    data = json.loads(args.examples.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {
        "site_id",
        "feature_id",
        "discovery",
        "heldout",
    }:
        parser.error("Require exact source site, feature, discovery and heldout fields")

    worker = LocalAutoInterpreter.from_local_snapshot(args.local_model, device=args.device)
    result = worker.interpret(
        site_id=data["site_id"],
        feature_id=data["feature_id"],
        discovery=data["discovery"],
        heldout=data["heldout"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.chmod(args.output, 0o600)
    print(
        json.dumps(
            {
                "status": result["status"],
                "approved": False,
                "heldout_score": result.get("heldout_score"),
                "generator_state_sha256": result["generator_state_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
