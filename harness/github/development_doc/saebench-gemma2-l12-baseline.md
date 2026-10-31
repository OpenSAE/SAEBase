# SAEBench Gemma-2-2B Layer 12 TopK baseline

This is an **optional baseline candidate**, not a claim of real-weight acceptance.
Use the existing `gemma2_2b_base` model and the new
`gemma2_2b_saebench_topk_l12_k80` SAE profile. Existing model defaults,
research outputs, historical evidence and benchmark metrics remain unchanged.

## Original source and identity

- SAE Bench: https://github.com/adamkarvonen/SAEBench
- Original HF repository: https://huggingface.co/canrager/saebench_gemma-2-2b_width-2pow14_date-0107
- Frozen repository revision: `18190d53776b0710543f8e7e8b78a2cb8f12e773`.
- Checkpoint directory: `gemma-2-2b_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_2`.
- Original `TopKTrainer`/`AutoEncoderTopK` at
  `blocks.12.hook_resid_post`, width 16,384, input 2,304, training/inference k=80.
- Original loader in `sae_bench.custom_saes.topk_sae` defines the TopK
  mathematics and checks decoder normalization. SAEBase loads tensors with
  `torch.load(weights_only=True)`, validates exact shape/source and uses that
  same upstream `TopKSAE` class. No custom TopK semantics are introduced.

The large checkpoint is **not** included in the GitHub source and is never
downloaded during a normal import or offline CI. Install a compatible pinned
upstream SAEBench checkout as needed for native acceptance. Dependencies,
checkpoint availability and license requirements must be verified on the
intended execution machine before any real benchmark.

## Validation sequence

1. Offline check (no downloads):
   `python -m pytest -q tests/workplan/saebench_topk/test_baseline.py`
2. Configuration and plugin identity: validate `model=gemma2_2b_base`,
   `sae=gemma2_2b_saebench_topk_l12_k80`, layer 12 and
   `blocks.12.hook_resid_post`; reject alternative layers and incompatible
   models.
3. In a separately approved network/GPU environment, call
   `load_sae_for_profile(..., layer=12)`; record exact base model identity,
   pinned checkpoint revision, downloaded weight SHA256, original config,
   actual hook/dimensions, and lossless activation tests.
4. Run SAEBase's already-existing
   `tasks/benchmark/native/custom_saebench.py` Core small acceptance through
   its `run_custom` wrapper with the frozen original SAEBench commit, followed
   by Sparse Probing only when approved. Preserve the original protocol,
   dataset, and output, and independently compare with upstream.
5. Before merging as fully verified, independently check the upstream source
   directory and every expected file at the pinned revision, the full real
   load, output evidence and negative controls.

Status: offline implementation candidate. No real checkpoint, native
evaluator or scientific result has been executed by this PR author.
