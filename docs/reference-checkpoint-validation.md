# Reference Checkpoint Acceptance

For the main research workflow, see [Documentation Home](index.md). The original checkpoint and experimental schema identifiers remain stable for compatibility.


This is the foundation gate for SAEBase v0.2. **reference validation does not implement the initial release
activation store or the batch intervention engine.** It restores the clean
installation contract and introduces a real-model numerical reference for
future activation storage/interventions comparisons.

## 1. Dependency and package boundaries

| Environment | Install in a fresh Python 3.11–3.12 virtual environment | Acceptance |
|---|---|---|
| Core | `python -m pip install -e .` | `saebase-check-env --mode core`, CLI, Python API imports |
| Development | `python -m pip install -e ".[dev]"` | Core checks, `python -m pytest -q`, wheel install test |
| Analysis | `python -m pip install -e ".[analysis]"` | `saebase-check-env --mode analysis`, check `sae_lens`, `transformer_lens`, `huggingface_hub`, `safetensors` imports |
| Generation | `python -m pip install -e ".[generation]"` | Experimental GPU backend; not an reference validation acceptance requirement |

`huggingface_hub` and `safetensors` are imported only when the loader actually
requires the corresponding remote checkpoint. Core use and offline tests no
longer require them. `required_packages('core')` lists all four core runtime
requirements (`numpy`, `torch`, `hydra-core`, `omegaconf`); the analysis check
adds the libraries defined in `pyproject.toml`'s `analysis` extra.

The GitHub CI has **separate fresh installations** for development, analysis,
and wheel packaging on supported Python versions. A missing dependency failure
is not counted as a passed model check.

## 2. Real checkpoint end-to-end acceptance

Do **not** label a run as real until its actual weights were downloaded or
loaded from a verified local cache and the test below succeeded:

```bash
# In an environment with network access to public Hugging Face weights:
python -m pip install -e '.[analysis]'
DEVICE=cpu bash scripts/m0_real_smoke.sh
```

The default run executes the following work on the **published**
Pythia-70M-deduped model + `pythia70m_resid_sm` SAE at layer 3:

1. Load both checkpoints using the registered production loaders.
2. Prepare versioned SimpleCopying prompts, including tokenizer identity and
   model-based logit filtering; retain a validated manifest.
3. Verify prepared token IDs against live tokenization for each reference.
4. Capture the actual residual-stream activation and SAE codes.
5. Check original activation patch identity against the model logits.
6. Ablate the strongest active feature while preserving SAE reconstruction
   residual; record the output logit change.
7. Hash the complete model and SAE parameter/state tensors and persist
   activated features, token IDs, sampled activation/logit values, package
   versions, selected site, tolerances and error measures.
8. Run the established `saebase --task sae` activation-health CLI against the
   same prompt directory, require all tested prompts succeed, and cross-check
   the selected hook/layer and sample count.

Success produces `m0-artifacts/m0_reference.json` and
`m0-artifacts/activation_health/sae_activation_summary.json`. No model weights,
private prompts or large activation caches are committed to Git.

To reproduce and compare the same reference in a second environment:

```bash
python scripts/m0_reference_smoke.py \
  --device cpu --seed 0 --n-generate 120 --min-kept 5 --n-prompts 3 \
  --data-dir m0-artifacts/data \
  --output m0-artifacts/rechecked.json \
  --reuse-prompts --compare-to m0-artifacts/m0_reference.json
```

The comparison requires matching model/SAE weight hashes, selected feature
indices, prepared token IDs and sample identities. Floating numerical probes
are compared under the stored tolerance. An unresolved upstream revision is
not invented: exact weight digests are recorded even if a revision is unknown.

**The structured report distinguishes `real_checkpoint_validated: true`
from synthetic offline tests.** The latter must never be presented as live
checkpoint acceptance.

## 3. GitHub CI gate

- `.github/workflows/ci.yml` installs development and analysis extras in
  separate clean environments; it executes offline tests and packaging checks.
- `.github/workflows/m0-real.yml` has a manual Run Workflow trigger. A main
  commit containing the explicit `[m0-real]` marker runs the CPU checkpoint
  acceptance once. Ordinary pushes do not download weights.
- Inspect the final Actions job status and produced artifact before claiming
  real reference validation acceptance. Pythia/SAE availability is external to the source tree.

## 4. reference validation exit criteria

- [x] Python 3.11/3.12 development install, CLI and offline CI green.
- [x] Analysis extra clean installation and import probes green.
- [x] Wheel and source distribution build and inspect successfully.
- [x] One actual Pythia-70M+SAE activation-health and identity-patch run succeeds.
- [x] Numerical reference includes full model and SAE weight hashes, tokens,
      activation/logit probes, and actual edit effects.
- [x] Rerunning against the same cached weights/data passes `--compare-to`.

Check these boxes only after seeing evidence from the corresponding run.

## 5. Verified October 14, 2026 acceptance evidence

- Implementation merged in [PR #1](https://github.com/OpenSAE/SAEBase/pull/1), commit `2daa855`.
- Clean environment and package CI: [successful run 38634415319](https://github.com/OpenSAE/SAEBase/actions/runs/38634415319), with Python 3.11, Python 3.12, analysis installation, and package jobs all successful.
- **Real pretrained checkpoint**: [successful CPU GitHub Actions run 38634372619](https://github.com/OpenSAE/SAEBase/actions/runs/38634372619), artifact [saebase-m0-reference](https://github.com/OpenSAE/SAEBase/actions/runs/38634372619/artifacts/11927288030).
- Real model profile: `pythia70m`, SAE `pythia70m_resid_sm`, `blocks.3.hook_resid_post`, input width 512, SAE width 32768.
- Real reference: `schema=saebase.m0.reference.v1`, `status=passed`, `real_checkpoint_validated=true`, 3 evaluated prompts, identity-patch maximum absolute logit error **0.0**.
- Tested per-sample SAE L0 counts: 172, 66, and 53. Ablating each prompt's highest absolute active feature resulted in a nonzero logit delta (L2 values approximately 195.95, 76.99, 82.20).
- Existing activation-health evaluation ran on the same prepared dataset: `n_ok=3`, `n_fail=0`, median L0=66, median reconstruction MSE≈0.09825, median FVE≈0.80736.
- A **second real model/SAE load** completed with the cached prompts and `--compare-to`, producing `m0_reference_replay.json` with `reference_comparison=passed`, matching full model and SAE state SHA-256 digests.
- The original real-run failures revealed previously undetected scalar-tensor hashing and TransformerLens keyword-hook issues. Both were fixed in `main` and covered by new regression tests.

reference validation acceptance is complete. These small-model measurements are **engineering acceptance evidence**, not a claim about SAE generalization or statistical quality.
