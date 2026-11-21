"""Expose SAEBase model families and shared language-model adapters."""

from pathlib import Path as _Path

# Keep the established models.<module> imports after grouping shared adapters.
__path__.append(str(_Path(__file__).resolve().parent / "language_model"))

from saebase.import_aliases import register_aliases as _register_aliases

_register_aliases({
    "models.gpt2": "models.gpt_family",
    "models.gpt2.gpt2": "models.gpt_family.gpt2_series",
    "models.gpt2.gpt2.specs": "models.gpt_family.gpt2_series.specs",
    "models.mistral.mistral7b": "models.mistral.mistral_series_7b",
    "models.mistral.mistral7b.specs": "models.mistral.mistral_series_7b.specs",
    "models.numerics": "models.numerical_policy",
    "models.qwen": "models.qwen_family",
    "models.qwen.qwen25": "models.qwen_family.qwen_series_25",
    "models.qwen.qwen25.specs": "models.qwen_family.qwen_series_25.specs",
    "models.qwen.qwen3": "models.qwen_family.qwen_series_3",
    "models.qwen.qwen3.specs": "models.qwen_family.qwen_series_3.specs",
    "models.qwen.qwen35": "models.qwen_family.qwen_series_35",
    "models.qwen.qwen35.specs": "models.qwen_family.qwen_series_35.specs",
    "models.qwen.r1_distill": "models.qwen_family.r1_distill",
    "models.qwen.r1_distill.specs": "models.qwen_family.r1_distill.specs",
    "models.transformerlens_backend": "models.transformer_lens_backend",
})
