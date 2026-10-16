# Troubleshooting

These checks resolve common installation, setup and first-experiment
problems without requiring a pretrained model download.

## Installation and Python version

SAEBase supports Python 3.11 and 3.12. If packaging rejects your
interpreter, check it first:

~~~bash
python --version
python -m pip install -e .
python -m pip check
saebase-check-env --mode core
~~~

The public `v0.2.0` wheel is an older release. Current
development features such as `saebase.extensions` require installing
from the repository checkout until a newer version is published.

## Missing optional libraries

`ModuleNotFoundError` for TransformerLens, SAE Lens, Transformers
or Hugging Face integration libraries often means you installed only
the small core package.

~~~bash
python -m pip install -e ".[analysis]"
saebase-check-env --mode analysis
~~~

The offline synthetic tutorial works with the core installation and
does not require these optional packages.

## First experiment cannot import SAEBase

Check that installation used the active Python interpreter and that
you run the example from the checked-out repository root:

~~~bash
python -m pip show opensae-saebase
python examples/19_first_experiment_offline.py --dry-run
~~~

For an existing editable checkout, reinstall with `python -m pip
install -e .` after switching environments.

## Benchmark dry-run versus execution

`saebase-benchmark --dry-run` validates the plan without loading
weights. Running it with `--real` actually executes the configured
model/SAE providers, which may download checkpoints and need GPU/RAM.

For a first report with no downloads, use
[the synthetic experiment](first-experiment.md) instead of a
real checkpoint configuration.

## Mismatched hook, layer, dimension or dtype

SAEBase checks that the selected `ActivationSite` matches the
registered model layer and SAE's input dimension. A failed
identity-patch check can signal an incompatible hook, backend or
non-deterministic model configuration.

Check the exact checkpoint and hook in the
[verified support matrix](verified-model-matrix.md) or the
[external extension contract](extensions.md). Do not force a
different hook solely because the embedding widths match.

## Results cannot resume

A source SHA, model/SAE hash, tokenizer, token selection, split or
frozen benchmark option change can require a new run ID.
An altered `metrics.jsonl` or reliability audit manifest correctly fails
integrity verification.

Keep existing results, start the changed experiment in a separate
output directory, and inspect `resolved_config.json` and
`provenance.json`. Do not edit result files to bypass integrity
checks.

## Report a reproducible problem

Search [existing issues](https://github.com/OpenSAE/SAEBase/issues),
then use the [Bug Report form](https://github.com/OpenSAE/SAEBase/issues/new/choose).
Include the Python/PyTorch versions, SAEBase source revision, command,
minimal input, exact model/SAE identifiers if relevant, and traceback.
Remove access tokens or private dataset content from logs.

[First Experiment](first-experiment.md) ·
[Getting Started](getting-started.md) ·
[Contributor Guide](contributor-guide.md)
