# M2 — Batch feature interventions and reference steering

M2 adds a bounded measurement engine on top of the **M0 verified real model**
and **M1 Activation Store**. It is the second implementation milestone of
SAEBase v0.2, *not* a unified benchmark suite or a claim of causal explanations
from a single intervention.

## Quickstart (offline, no downloads)

```bash
python -m pip install -e '.[dev]'
python examples/11_batch_interventions_offline.py
python -m pytest -q tests/test_m2_interventions.py
```

The demo uses the deterministic toy TransformerLens-like backend and a toy SAE.
Its outputs are engineering examples, not scientific findings.

## JSON experiment CLI

```bash
# Validate IDs, operators, measurements and budgets, without loading weights.
saebase-intervene --config conf/interventions/m2_pythia_example.json --dry-run

# Explicitly authorize public Pythia/SAE checkpoint download or loading.
saebase-intervene --config conf/interventions/m2_pythia_example.json \
  --output experiment/m2-pythia --real

# Re-running reuses completed measurements. Refuse reusing an existing run:
saebase-intervene --config conf/interventions/m2_pythia_example.json \
  --output experiment/new-run --real --no-resume
```

The provided real-model configuration includes one frozen M0 example token
sequence as an *illustration of syntax*. New experiments must provide their own
complete token IDs, prompt identity and reproducible protocol. The default CLI
runs on CPU. A separate Python API allows configuring CUDA with an appropriately
loaded model and SAE.

`--dry-run` reports sample count, condition count, total measurements, maximum
length and an optimistic patch-forward lower bound. Time, VRAM and device
performance are deliberately `null` until a pilot run measures them.

## Operator semantics

| Operation | Code replacement | Residual |
|---|---|---|
| `identity` | Unmodified `z` | Preserved |
| `ablate` | Selected `z_i -> 0` | Preserved |
| `scale` | Selected `z_i -> alpha * z_i` | Preserved |
| `clamp` | Selected `z_i -> c` | Preserved |
| `add` | Selected `z_i -> z_i + c` | Preserved |
| `coalition` | Zero selected IDs not in `retained_ids` | Preserved |
| `reconstruction` | Entire `h -> decode(encode(h))` | **Not preserved** |

For all feature-only operators, SAEBase recomputes

```
h_new = h_original + decode(z_edited) - decode(z_original)
```

This preserves content not reconstructed by the SAE. The `reconstruction`
control is intentionally different and can later support Loss Recovered.
Feature indices are **absolute SAE dictionary indices**; coalition retained
indices must be a subset of the selected set. The decoder is invoked through
its loaded implementation so biases and normalization are applied consistently.

## Python API

```python
from saebase.interventions import (
    BatchEngine, Condition, ExperimentIdentity, MeasurementSample,
    ResultWriter, RunBudget, dry_run, read_results,
    GenerationSpec, run_steering,
)
from saebase.store import SampleRecord, TokenSelection, sha256_state_dict
from saebase.api import ActivationSite

# Assume `model` is a ModelAdapter, `sae` is loaded, and both are in eval mode.
site = ActivationSite(layer=3, hook_name='blocks.3.hook_resid_post')
identity = ExperimentIdentity(
    model_state_sha256=sha256_state_dict(model.model),
    sae_state_sha256=sha256_state_dict(sae),
    site=site, backend=model.backend,
)
record = SampleRecord('sample-001', tuple(token_ids), selection=TokenSelection('last_valid'))
samples = [MeasurementSample(record, target_token_id=target_id, reference_token_id=other_id)]
conditions = [Condition('none', 'identity'), Condition('remove', 'ablate', (feature_id,))]
plan = dry_run(samples, conditions, microbatch=8, budget=RunBudget(max_measurements=1000))
with ResultWriter('experiment/m2/measurements', identity) as results:
    stats = BatchEngine(model, sae, identity).run(
        samples, conditions, results, sample_batch=4, microbatch=8,
    )
```

A run records versioned `measurement_id`, `model/SAE` SHA-256 identities,
position metadata, operation, logit difference, KL(base || patched), L2 logit
change, actual `||delta h||`, target logits and top-1 correctness when a target
is supplied. `read_results()` verifies committed JSONL shard hashes.

## Batching and backends

- For TransformerLens-compatible models, **different prompts and intervention
  conditions** can be processed in a single padded forward/hook call. The model
  must use causal attention; only unpadded/right-padded inputs are supported.
  An explicit `pad_token_id` is required for mixed prompt lengths.
- `sample_batch` bounds the number of unique prompts in each baseline pass;
  `microbatch` bounds patch forwards. A CUDA out-of-memory during patching
  halves the microbatch and retries without skipping conditions. If a
  single intervention cannot fit, the error is propagated.
- HFHooksModel uses the already validated sequential API and the stats report
  its actual call count; M2 does **not** claim this backend is vectorized.
- An M1 raw activation reader can bypass repeated activation capture. An M1
  SAE code reader additionally bypasses encoding. Source-activation digests,
  checkpoint SHA-256 and sample/position IDs must all match. **New logits
  forwards are still required for causal patch experiments**; M1 caching
  does not eliminate the downstream network computation.
- For CPU real-checkpoint batched GEMMs, floating-point kernel order may
  produce small differences from a single-example forward. Report the maximum
  absolute **individual logit** error and specify a tested absolute tolerance
  (the Pythia-70M M2 smoke uses 3e-3 absolute and 1e-5 relative tolerance); do not
  confuse this bound with the Euclidean norm across the entire vocabulary.
  In particular, a stored singleton activation patched into a multi-prompt
  forward can have a small nonzero output delta even for an identity edit.
  We retain the exact checkpoint/cache identity checks and explicitly record
  the peak per-logit identity discrepancy; this tolerance applies only to
  the batch replay path, not to M0 activation-probe acceptance.
- The current public interface requires exactly one selected patch position
  per measurement sample. M1 itself supports multiple positions; M2 can make
  distinct samples for each site as an explicit future extension.

## Crash recovery, resource budgets and measurements

- Results are written in bounded JSONL shards and committed by atomically
  replacing `manifest.json`; uncommitted shard files are not authoritative.
- Resume verifies shard hashes and sample/condition identities and skips
  already committed work **without invoking the model**. A mismatched model,
  SAE or experiment identity cannot be silently reused. One writer per local
  results directory is protected by an advisory filesystem lock.
- The default maximum is 100,000 logical measurements. A separate generation
  budget limits the aggregate number of token predictions. No claim is made
  about wall-clock runtime before a real device pilot.
- Statistics include measured/reused rows, baseline forward batches, patch
  batches, OOM retries, peak microbatch and measured interventions/second.
  They are meaningful for their actual backend and device only.
- Random/matched-norm controls are **not silently invented**; users supply
  actual feature IDs and explicit strength conditions. Higher-level control
  *selection* and automatic statistical comparisons belong in M3/M4.

## Reference short-text steering

M2 implements an intentionally straightforward generation loop using
`compare_steered_generation` / `run_steering`. Every generated token causes a
**full-prefix forward** and a patch at the current last valid token of the
steered continuation. A separate unmodified continuation supplies a baseline.

- Greedy decoding (`temperature=0`) and independently seeded categorical
  sampling (`temperature>0`, optional `top_k`) are supported.
- Common random numbers start from the same sample-specific seed, with
  separate, independent RNG state for the two continuations.
- EOS stops and max new-token limits are recorded. `max_context_tokens` can
  prevent an unsupported generation length.
- KV-cache optimization is intentionally **not** claimed; cached decoding
  changes intervention semantics if old prefix states are stale.
- Both baseline and steered token ID sequences, stopping reasons and exact
  generation specs are written to the result store for reproducibility.

## Real checkpoint acceptance

The opt-in GitHub Actions workflow `.github/workflows/m2-real.yml` downloads
public Pythia-70M and the `pythia70m_resid_sm` SAE and verifies:

1. Exact loaded model and SAE full state digests from the [M0 real fixture](../tests/fixtures/m0_reference_public.json).
2. M0 activation probes and SAE activation sparsity.
3. M1 activation/code caches and feature IDs in a real run.
4. All seven M2 conditions in a **real multi-prompt batch**.
5. Exact numerical comparison between the M2 batch result and the existing
   single-prompt `evaluate_edit` reference.
6. Zero model forward calls on a fully committed result resume.
7. Actual greedy 2-token continuation with and without feature ablation.

The workflow uploads a JSON artifact with results, identity checks, numerical
differences and batch performance. It is not considered passed until the run
and artifact are inspected. A synthetic test passing is not proof that public
Pythia/Gemma weights work.

## M2 completion gates

- [ ] Existing M0/M1 tests and M2 contracts pass in Python 3.11 and 3.12 CI.
- [ ] Clean wheel + source distribution includes Python, CLI and JSON config.
- [ ] Real Pythia/SAE run completes with M0/M1 numerical comparisons.
- [ ] Real multi-prompt batch output equals sequential reference to tolerance.
- [ ] Identity, recovery, OOM, mixed-length and steering tests pass.
- [ ] Real JSON artifact is archived and linked from here after inspection.
