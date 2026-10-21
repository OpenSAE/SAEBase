# Your First SAEBase Experiment

This walkthrough produces a complete benchmark and reliability report on
your own computer, without pretrained checkpoints or GPU access.

It uses one tiny deterministic model, two distinct toy SAE weights, eight
independent input groups, and three existing SAEBase metrics. All values are
synthetic demonstrations.

## Install

SAEBase supports Python 3.11 and 3.12.

~~~bash
git clone https://github.com/OpenSAE/SAEBase.git
cd SAEBase
python -m pip install -e .
~~~

The checkout provides the examples and full documentation. The base install
does not require the optional Hugging Face/SAE Lens analysis dependencies.

## Run the experiment

First, inspect the bounded experiment plan:

~~~bash
python examples/quickstart/first_experiment.py --dry-run
~~~

This prints its deterministic run ID, two model–SAE evaluations, eight
independent prompt groups and the planned metric/measurement workload.
It does not write files or download model weights.

Run the same experiment and keep the results:

~~~bash
python examples/quickstart/first_experiment.py --output runs/first-experiment
~~~

The command prints JSON with the exact paths. Open the `report_html`
path in a browser, or inspect the CSV file named by `metrics_csv`.
The run ID is derived from the complete frozen benchmark configuration.

## What the program does

1. Builds one tiny model and two SAEs with different weights.
2. Captures model activations once and reuses the identity-checked
   activation store across the two SAEs.
3. Computes selected-token feature count (`core.l0`), reconstruction
   MSE and downstream logit change under identity and feature-ablation
   interventions.
4. Writes metric rows, source/weight identities, measurement references,
   a CSV summary and a portable HTML report.
5. Reads the original cached activations and intervention measurements
   to create a grouped reliability audit. Independent input groups, rather
   than individual feature activations, are the resampling units.

The `real=True` argument inside the example enables execution of the
benchmark engine. The example still uses **synthetic, locally constructed**
model/SAE weights, not real pretrained checkpoints.

## Explore the output

The directory structure is:

~~~text
runs/first-experiment/
  <run_id>/
    resolved_config.json
    provenance.json
    plan.json
    metrics.jsonl
    metrics_manifest.json
    summary.csv
    report.html
    reliability.json
    reliability_manifest.json
    reliability_summary.csv
    reliability.html
  _cache/      (SAEBase-managed activation and intervention stores)
~~~

Use `summary.csv` to compare metric rows. Metrics retain their
original units, conditions, observation counts and SAE identifiers;
they are **not combined into one quality score**.

Use `reliability.json` to examine audit status, grouped estimates
and paired comparisons. Eight synthetic groups exercise the reporting
and source-consistency machinery; they do not establish statistical
evidence about pretrained representations.

Run the same command again with the same output path. SAEBase verifies
and reuses the committed result rather than recalculating its
measurements. The integrity manifest binds the result to its run ID.
Different metrics, inputs or checkpoints must have distinct identities.

## Continue with your research

- For a real public model/SAE checkpoint, follow [Getting Started](getting-started.md)
  and review the [verified layer matrix](verified-model-matrix.md).
- To inspect activation cache files and feature edit mechanics, see
  [Activation Store](activation-store.md) and
  [Feature Interventions](feature-interventions.md).
- For original SAR, RSS or Input Localization estimators, see
  [Research Integrations](research-integrations.md).
- To connect a new checkpoint or estimator, use
  [Third-Party Extensions](extensions.md).
- To contribute a bug fix, test, example or documentation improvement,
  follow the [Contributor Guide](contributor-guide.md).

[Examples Index](../examples/README.md) ·
[Troubleshooting](troubleshooting.md) ·
[Python API](python-api.md)
