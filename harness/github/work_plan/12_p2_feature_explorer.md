# WP12 P2 — Feature exploration, recorded prompt traces and explanations

Status: **P2 implementation branch. Routine CPU tests in progress; broad real-data coverage and final verification belong to P8.**
Branch: `feat/wp12-p2-feature-explorer`, stacked on P1.
References: [Neuronpedia dashboard](https://docs.neuronpedia.org/sparse-autoencoders/feature-dashboards), [three search modes](https://docs.neuronpedia.org/sparse-autoencoders/search), [SAE-Vis](https://github.com/callummcdougall/sae_vis).
Parent: [WP12 roadmap](12_unified_interpretability_master_blueprint.md).

## Purpose

One actual SAE feature identity should be discoverable from labels, a previously
recorded prompt or a selected token. Opening it gives source-identified
reviewed examples, hypothesis labels, activity counts and a portable offline
feature page. No synthetic outputs are displayed as real. A recorded prompt
lookup **does not run a new model**.

The public CPU Lab is deliberately read-only. It can browse exactly the data
the operator published, without needing a resident GPU. New input calculations
are an explicit local, opt-in Python task. P2 leaves actual hosted GPU inference
to P7; it does not require re-running the historical 48-site matrix.

## Shipped source routes

| Capability | Actual implementation | Present boundary |
| --- | --- | --- |
| F01 feature dashboard | `GET /api/explore/features/{site_id}/{feature_id}`; stable `#explore?site=...&feature=...` location | Shows only source-backed cards from real SQLite / compact indexes; population density/logits absent without source evidence |
| F02 search | `POST /api/explore/search` with separate `explanation`, `inference`, `token` modes | Prompt mode only retrieves *exact* published frozen input; genuinely new prompt gives 503 until a GPU worker is available |
| F03 auto-interpretation | `backend.auto_interp.propose_interpretation` pluggable model generator, explicit budget, held-out boolean-activation classifier, negative examples | No default external model/API; hypotheses require human approval and declared method; score is not semantic ground truth |
| F15 feature analysis | Existing checked encoder outputs + new whole-sequence `saebase.explore.capture_prompt_trace` | Model/SAE must be local and correctly matched; no unbounded model sweeps |
| F31 feature-centric view | Exact site+index card, inspected contexts, hypotheses, navigation and source file | Examples are reviewed subset; cannot claim globally highest corpus contexts if coverage is limited |
| F32 prompt-centric view | `GET /api/explore/traces/{trace_id}` and local real capture CLI | Only opt-in published tokens; unknown user inputs aren't simulated |
| F33 token inspection | `POST /api/explore/search mode=token` sorted real complete sparse codes | Exact tokenizer IDs, selected position and fixed prompt identity |
| F34 correlations | `GET /api/explore/features/{site_id}/{feature_id}/correlations` using complete sparse code per independent prompt group | Requires at least three independent groups and nonzero variance; absence in *complete* sparse rows only can be treated as zero; not causal identity |
| F35 portable visualization | `GET /api/explore/features/{site_id}/{feature_id}/export.html` | Self-contained HTML with escaped approved text, no CDN, model run or unlicensed media |

Existing `backend.feature_store` is now the active source for the UI and API
when its immutable SQLite manifest exists. The compact JSON feature index
remains supported. `feature_by_id` performs an indexed lookup with preceding
and following feature ID, rather than loading a dictionary of 65k entries.
No old Research Lab or paper estimator data structure is overwritten.

## Source contracts

Published exploration data sit in `$OPENSAE_LAB_DATA/explore`:

```text
explore/
  manifest.json
  records.json
```

The release contains at most 32 verified-site prompt traces and 1,024 approved
candidate explanations, with an 8 MiB total limit. Every complete recorded
trace is tied to the existing verified site and full model/SAE state hashes,
explicit tokenizer SHA256, prompt text SHA256, independent prompt-group ID,
discovery/evaluation split, real nonzero code values at every token and
positive explicit rights review. Nonzero codes are **not truncated** to
top-k. Publication requires a separate explicit `public_consent=True` call.
Both files become immutable and checksummed. Tampering fails closed.

A `real_checkpoint` source flag alone is not a scientific guarantee: the
operator must inspect original GPU/CPU run artifacts and the source/rights
record before approving any release. All paths are empty by default, without
placeholder metrics or LLM-generated fake explanations.

## Producing your own genuine prompt trace

In an isolated analysis environment with the exact registered weights
already accessible:

```bash
python -m pip install -e ".[analysis,dev]"
python saebase/tools/wp12_p2_capture.py \
  --sae-key gemma270m_scope_resid --layer 5 --device cuda \
  --prompt-file /private-workspace/input.txt \
  --sample-id local:sample1 --group-id local:group1 \
  --split discovery \
  --output /private-workspace/trace1.json
```

The capture uses an actual model forward and native SAE encoding for every
prompt token, checks that each sparse row is complete and bounded, and outputs
real feature indices and their values. The output has
`public_text_reviewed=False` and `rights_reviewed=False`. It cannot be
published to the CPU Lab until an operator independently reviews actual
inputs, model/source digests and redistribution rights, then explicitly
produces the reviewed release. Do not publish user-supplied private text by
simply changing boolean flags.

Publish a reviewed SQLite feature index:

```bash
PYTHONPATH=saebase/research_lab python -m backend.feature_store \
  --input /private-workspace/reviewed_feature_observations.jsonl \
  --output /private-workspace/research-lab/features
```

To publish an approved prompt/interpretation release:

```python
from pathlib import Path
from backend.explore import publish_exploration

# traces and explanations must be source-checked and human-reviewed inputs.
manifest = publish_exploration(
    traces, explanations,
    output=Path("/private-workspace/research-lab/explore"),
    public_consent=True,
)
print(manifest)
```

**Do not** upload the raw model weights or private user inputs to GitHub.
No original Hugging Face dataset text or Neuronpedia feature labels have been
mirrored into this PR. For third-party data, use official authorized exports
with exact source identifiers, not page scraping.

## Running basic tests

```bash
python -m pip install -e ".[dev]"
python -m pip install -r saebase/research_lab/requirements.txt
python -m pytest -q saebase/research_lab/tests/test_p2_explorer.py
python -m pytest -q saebase/research_lab/tests/test_platform.py
node --check saebase/research_lab/frontend/app.js
```

Development tests use synthetic small numeric fixtures isolated under
`tmp_path`. They do not establish that ten real features are publicly
released. The Research Lab will show accurate empty states until source
approved assets exist. A C1 prompt-capture pilot for a published checkpoint is
optional for development and not a reason to delay P3 implementation.

The accepted P2 source footprint preserves all prior work. Final large
dictionary performance, independent interpretations, cross-model
coverage, native SAE-Vis parity and at least ten authentic user demonstrations
remain explicit P8 checks. P3 can start when the routine P2 code checks pass.
