"""Private, local-only SAE feature interpretation and held-out scoring.

No external model provider or paid API is invoked. The operator selects an
already-downloaded instruct checkpoint. All examples come from real
source-identified SAE activations, not synthetic published feature labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .auto_interp import propose_interpretation


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
        encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
        if not isinstance(encoded, dict) or "input_ids" not in encoded:
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
        label = response.upper().strip().rstrip(".")
        if label not in {"YES", "NO"}:
            raise ValueError("Held-out predictor returned neither YES nor NO")
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
