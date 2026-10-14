# SAEBase 0.1.0 release checklist

## Reproducible package checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/verify_distribution.py dist
python scripts/check_installed_wheel.py dist
```

Run the installed CLI from a temporary directory outside the checkout:

```bash
saebase --list
saebase-check-env --mode core
python -m saebase.sae.verify_registry --list
```

For published Pythia-70M/SAE validation on a model-enabled machine:

```bash
python -m pip install -e ".[analysis]"
bash scripts/phase_a_smoke.sh
```

For the second pairing and the paper integration examples:

```bash
python examples/09_second_model_smoke.py --model gemma270m
# SAR_SOURCE and RSS_SOURCE should point to their original checkouts.
SAR_SOURCE=/path/to/SAR RSS_SOURCE=/path/to/RSS bash scripts/phase_c_smoke.sh
```

## Approval gates

- [ ] Offline tests and installed wheel smoke pass on Python 3.11 and 3.12.
- [ ] `saebase --list` works outside the checkout; package config is present.
- [ ] Pythia-70M and the corresponding public SAE run in a real environment.
- [ ] Gemma-3 270M pairing is run and results are reviewed.
- [ ] vLLM and NNsight capabilities are labeled according to live GPU testing.
- [ ] Source ownership, Apache-2.0 licensing and third-party attribution are confirmed.
- [ ] Maintainers approve documentation, version, and contact information.

## Publish after approvals

1. Create or choose the OpenSAE/SAEBase repository.
2. Run the `ci.yml` workflow on its default branch.
3. Confirm the release metadata (including `CITATION.cff`) and push the annotated `v0.1.0` tag after all gates pass.
4. Upload a verified wheel, sdist and source archive to a GitHub Release.
5. Confirm installation with the release artifacts from a new environment.

The CI workflow builds the artifacts; a tag does not automatically publish to
PyPI or deploy external model weights.
