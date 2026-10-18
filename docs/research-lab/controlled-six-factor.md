# Six-factor controlled input designs

This CPU-only component prepares the **two complete 64-state** controlled input populations required by Work Plan 04. They are distinct from the 10 public source datasets and **are not part of the 500 ordinary prompts**.

- **synthetic_known_factor_control** defines six abstract independent toggles and a six-feature mathematical truth table. Its values are synthetic, not learned SAE activations.
- **real_model_six_factor_design** defines six explicit factors of a fictional event-writing request. Its 64 prompts are project-authored, with no model forward or checkpoint observations recorded yet.

Every state mask from 0 to 63 occurs once, in integer bitmask order. Factor zero is the least significant bit. All 64 states share **one independent prompt_group_id**; resampling them as 64 unrelated prompts is statistically invalid.

## Generate on CPU

Install SAEBase in the repository checkout (no GPU necessary):

    python -m pip install -e ".[dev]"
    cd apps/research-lab
    python -m backend.controlled_designs --output-dir /tmp/opensae-six-factor-v1

Both JSON files are immutable. They include project-authored prompt text, factor assignments, bitmask order, SHA-256 prompt fingerprints and manifest digests. The builder refuses to overwrite a pre-existing file.

The operator may import make_candidate and to_saebase_design from backend.controlled_designs, then use the original SAEBase ControlledInputDesign. The Research Lab does not introduce another model hook, cache identity system or localization estimator.

The synthetic_reference_codes function is available only for the abstract synthetic control. It must never be passed off as real checkpoint SAE codes.

## GPU-stage boundary

For real-model input design, the GPU operator must still choose and pin a model/SAE/hook, tokenize all 64 prompts, verify distinct token sequences and selected patch positions, capture actual SAE activations, preserve all source hashes, and call original Localize on the **full** grid.

These JSON files prove only that input designs are complete, not causal independence, meaningful learned features, or localization conclusions. Do not replace these with 64 arbitrary natural-language examples. The grid is one complete controlled experiment.
