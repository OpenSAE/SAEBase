# SAEBase performance and reliability

SAEBase reduces repeated computation during feature evaluation and supports
generation with KV caching.

## Execution optimizations

- Cached activation encoding now coalesces uncached records within a bounded
  reader batch into one SAE invocation, splitting codes back by their exact
  sample boundaries. Existing row identity and source SHA-256 checks remain.
- The real Hugging Face hooks wrapper has an explicit masked batch forward
  capable of capturing original residual activations and applying per-row
  patches. Only right-padded causal inputs with valid patch/readout positions
  are accepted.
- Hugging Face batch execution is optional; single-sample forwards remain the default.
- OOM microbatch reduction is retained for subsequent patch chunks instead
  of retrying the original unsafe size repeatedly.
- A deterministic offline encoding microbenchmark and CPU batch-parity tests
  are included.

## Measure, do not infer speedups

Run `python saebase/tools/benchmark_encoding.py` with a chosen CPU/GPU device.
The output records the device, tensor sizes and median sample throughput
for selected batch sizes. This isolates SAE encoding overhead and is **not**
an end-to-end benchmark or a verified acceleration factor.

For complete workloads, capture wall time and CUDA peak allocated/reserved
memory, including model load separately from warmed inference. Compare:
activation collection, feature interventions, SAR coalitions, RSS variants,
cached SAE encoding and steered token generation at matched inputs/precision.

A change is accepted as a *scientifically equivalent acceleration* only
after identity patches, per-position logits, exact feature code provenance,
sample grouping, original SAR/RSS estimators, and appropriate numerical
tolerances pass on each targeted real model and checkpoint.

## Backend acceptance

A faster backend remains opt-in until equivalence checks have passed for each relevant model, checkpoint, token position and edit condition. Verify right-padding parity, identity edits, exact feature-code provenance, resource limits, interruption and resume, and recorded peak CUDA memory. Do not infer throughput, scientific validity or compatibility from a standalone microbenchmark.

For active benchmark and intervention interfaces, use the [Python API](python-api.md) and [Feature Interventions](feature-interventions.md). Hardware-specific timings belong to the measured run receipts, not the permanent implementation guide.
