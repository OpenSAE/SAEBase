# SAEBase performance and reliability

SAEBase optimizes its evaluation execution while keeping scientific identities,
SAE estimators, feature operations and results unchanged.

## Implemented in this branch

- Cached activation encoding now coalesces uncached records within a bounded
  reader batch into one SAE invocation, splitting codes back by their exact
  sample boundaries. Existing row identity and source SHA-256 checks remain.
- The real Hugging Face hooks wrapper has an explicit masked batch forward
  capable of capturing original residual activations and applying per-row
  patches. Only right-padded causal inputs with valid patch/readout positions
  are accepted.
- The intervention engine uses this new optional batch API when present,
  while preserving the legacy single-sample fallback for other adapters.
- OOM microbatch reduction is retained for subsequent patch chunks instead
  of retrying the original unsafe size repeatedly.
- A deterministic offline encoding microbenchmark and CPU batch-parity tests
  are included.

## Measure, do not infer speedups

Run `python scripts/benchmark_encoding.py` with a chosen CPU/GPU device.
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

## Not yet verified or shipped

- Exact GPU speedups as a new default. The opt-in block-suffix replay
  stays off unless `BatchEngine(..., enable_suffix_replay=True)` is set.
  On the full Qwen3-1.7B Base layer-7 workload (25 rows, 7 edits, identity
  replay included) the RTX 5070 median is 1.888 s versus 2.409 s for the
  original hooks, 1.276x, metric max abs 0. Untested families keep the
  original hook. See [workplan11_results.md](performance/workplan11_results.md).
- 48-site live-model parity and cross-backend equivalence.
- KV-cached steering. The reference still recomputes a clean prefix at each
  step. An incremental clean-history prototype matches the prefix itself,
  then leaves the frozen bf16 bar on the first new token (layer 7 activation
  max abs 0.25). It is not enabled.
- interp-engine 1.14.0 / vLLM 0.28 SAE capture. The write tap applies, and
  the residual, logits, and TopK support do not match HF. Dynamic and static
  taps stay off the scientific default.
- Multi-GPU scheduling, dynamic model residency and production hosted GPU
  serving; Research Lab's public endpoint remains read-only.

A production release should additionally test right-padding parity on each
architecture, CUDA OOM injection, worker interruption/resume, pinned backend
dependencies and clean installation. Do not enable a backend automatically
until its advertised capabilities have passed the associated tests.
