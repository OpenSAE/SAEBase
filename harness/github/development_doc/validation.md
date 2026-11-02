# Validation and Reproducibility

SAEBase distinguishes **what can be configured**, **what has run with real public weights**, and **what its scientific results establish**. A successful registry lookup or a synthetic smoke test is not proof of a real checkpoint evaluation.

## Current verification

- Validate source and contracts with the [CI workflow](../../../.github/workflows/ci.yml) for a focused change.
- Before software release, run [Release Validation](../../../.github/workflows/release-validation.yml) on the exact approved source SHA.
- For a real checkpoint experiment, record the actual model, SAE, hook, tokenizer, input population, parameters, output hashes, resource use and independent verification status.
- Treat configured, offline-tested, real-checkpoint-tested, independently reviewed and publicly released as separate evidence levels.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
python saebase/tools/verify_distribution.py dist
```

The model/SAE source registry, task-specific evaluation protocols and released run identities are authoritative for their respective scope. A successful synthetic check does not prove scientific model behavior.

## Reproducibility constraints

- Compare equivalent inputs, checkpoints, metrics and test splits.
- Group related prompt variants correctly for uncertainty estimates.
- Preserve original estimator and checkpoint identities.
- Refuse incompatible caches or a mismatched replay rather than silently recomputing results under different conditions.
- Record unexecuted, blocked and failed checks explicitly.

For the current software release procedure, see [Release Process](release-process.md). Historical release artifacts remain recoverable from their immutable Git tags without being required inputs to current verification.
