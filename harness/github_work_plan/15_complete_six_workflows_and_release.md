# Work Plan 15 — Complete the Six Core Product Workflows and Release

**Date:** 2026-10-22  
**Status:** Active engineering directive; implementation and runtime results are NOT yet claimed  
**Repository:** OpenSAE/SAEBase  
**Starting point:** `harness/github_development_doc/wp13-real-world-integration` after WP14 receipt review  
**Execution runbook:** [15_gpu_agent_completion_runbook.md](15_gpu_agent_completion_runbook.md)  
**Current task board:** [15_release_completion_matrix.csv](15_release_completion_matrix.csv)

## Executive decision

WP14 was overly willing to hide features when an upstream checkout, small checkpoint, or data snapshot was missing. Such missing assets are **engineering prerequisites**, not product reasons to give up. **Finish G1, G2, G3, G4, G5 and G6, integrate all six into a usable local API/CLI and public recorded-results Research Lab, and then prepare release.**

This plan **supersedes WP14's SHIP/HIDE/DEFER decisions and the WP14 release scope as instructions for what to finish next**. Preserve WP14 records as historical observations; do not rewrite the old GPU receipt, past source hashes, WP12 scientific ledgers, Stage D results, SAR, RSS, or Input Localization estimators.

We are not restarting a 50-function checklist, requiring 50 P8 independent certifications, implementing remote NDIF, or requiring multi-GPU distributed training. The scope change is precisely the **six G1–G6 product workflows** plus ordinary public deployment, documentation, and usability.

**Mandatory distinction:** running a local GPU job and browsing its real recorded output in a public CPU Space are different experiences. Finish both where applicable: users must be able to execute the model-dependent operation through a real local Python/CLI path, and visitors must be able to inspect a real saved result in the Research Lab. Never describe a CPU Space as live GPU serving.

## 1. Mandatory product capability contract

| ID | Public Research Lab journey | Local new-input capability | Minimum completed result |
| --- | --- | --- | --- |
| G1 Explore + Intervene | Original prompt → token → actual SAE feature → source-consistent no-op/ablation/scale | Load Gemma-3-270M L5 SAE, inspect a fresh prompt, apply actual SAE intervention; private operator queue inspect/edit if achievable with existing code | Reusable command, real sparse codes, measured logits, populated interlinked UI |
| G2 Generate + Steer | One genuine recorded baseline/no-op/edited continuation with token sequence, model/site and per-step effects | Run real token-by-token (reference or correctly cached) feature steering on user-supplied input | Actual baseline and edited generation, measured next-token probability/logit shift, no-op parity; divergent decoded text desirable but not fabricated |
| G3 Benchmark | Genuine original-upstream Core and Sparse Probing result pages, clear model/SAE/metric IDs | Run native SAEBench evaluator, import real original result and open it from SAEBase | One bounded real original Core + Sparse Probing execution, readable numbers and saved provenance |
| G4 Circuits | Real native graph with node/edge/error inspection, filtering, pinning, source details and export | Compute one genuine circuit-tracer graph from supported weights, import to SAEBase; attempt a graph-linked native intervention where compatible | Native graph computation + converted graph + working UI + executable example |
| G5 Training | Actual small training run record, losses/checkpoint and a load/reuse demonstration | Train a real SAE with original SAELens or Sparsify on an allowed real model/dataset; save and reload in SAEBase | Native run, genuine checkpoint reload, real encoding/reconstruction and user-facing instructions |
| G6 Auto Interpretation | Real candidate explanation labeled as model-generated with positive/negative activation evidence and held-out score | Use local installed instruct model to infer a candidate from genuine SAE examples and score separate held-out groups | Genuine local model generation + scored candidate + a human-approved, non-misleading published example |

All six must have a **real runnable backend**, a **documented example**, **one real stored result**, and a **meaningful available UI path** before saying the complete feature suite is released. Do not satisfy this with synthetic fixtures, mock pages, filenames alone or unfounded labels. Keep GPU operations private/local; public pages display reviewed recorded evidence and instructions for reproducing it.

## 2. Engineering decisions for each missing prerequisite

### G1 — Close the integration loop, not rerun GPU-00

Use G1's existing RTX 5070 Gemma-3-270M L5 F1627 authentic capture and edit. The reported 13-token prompt has 972 nonzero codes and a no-op 0 logit difference; ablation changes the teacher-forced edited-position argmax 1607 → 7549. Keep this **G1** example source-bound; do not attach it to the different **LIVE-03** F506/F95 examples as if from the same run.

- Implement and test fresh `capture_prompt_trace` + `compare_sae_edits` on a clean documented environment.
- Private queue previously proved two **inspect** jobs and a bad bearer token rejection; explicitly test **edit** jobs too before saying queue-based intervention works.
- For public browsing, curate and approve one original prompt from existing LIVE-02 traces, publish the exact complete sparse record in the approved local staging bundle, and show a feature page. Stage authentic LIVE-03 intervention cards separately, with their actual source identities.
- Acceptance: a new researcher can follow one tutorial, while a public visitor can navigate Prompt → Token → Feature → Recorded Intervention without confusing model, feature or prompt.

### G2 — Finish generation steering as a functional tool

A four-token greedy continuation remained identical under two ablations. That is **not** sufficient to diagnose steering. Use the existing reference generation path `saebase.interventions.steering.compare_steered_generation` and relevant faster cached engine when it has a compatible model/site; choose features active at the **generation-step last token**, not just at an unrelated earlier prompt token.

- Execute baseline, no-op and 2–3 bounded edit strengths (ablate / scale, optional additive) on two deliberately distinct original prompts, initially 16–32 generated tokens. Test the actual generation-time intervention at each step.
- Report per-step edited/logit or probability shifts, feature activity, generated token IDs and decoded text; record no-op equality, RNG/decoding settings, time and GPU use. Varying the decoding seed or overly large edit must **not** be misrepresented as a stable causal effect.
- If greedy text remains identical, a genuine distribution shift with exact working generation and user-controlled edit still counts as a working steering mechanism; choose a truthful UI with per-token probability/rank differences. Seek a clear text-difference example with one carefully selected alternative feature or prompt, not a never-ending search.
- Acceptance: local prompt-in → two actual continuations and per-step effects out. The public recorded demo is a real saved execution, with the direction/effect shown rather than an invented success narrative.

### G3 — Obtain native SAEBench, then run the provided code

The current `saebase/wp12/benchmarks.py` already expects exact upstream commit **`8042bb3828c6340da8d12062324e92b2077c571c`** at `<worktree>/upstream`, and `saebase/tools/m3_native_saebench_smoke.py` directly calls official Core and Sparse Probing on Pythia-70M L3 (upstream gated SAE) and AG News. **Missing upstream is a straightforward setup task.**

- Clone [SAEBench](https://github.com/adamkarvonen/SAEBench) into the expected path; checkout and verify the exact commit. In an isolated environment install its checkout and applicable commit-specific constraints, then `python -c 'import sae_bench'`. Do not replace a pinned old runner with a newer incompatible API silently.
- Fetch the exact original selected Pythia/SAE/dataset assets through the original libraries in the private environment. Start with the existing bounded original CPU evaluator; use GPU or a smaller batch if that same native protocol supports it.
- Run `NativePilot(...)`, retain actual Core L0/MSE and Sparse Probing accuracy (or debug an upstream API mismatch in a focused PR), then import through existing native adapter and show result cards. A sourced comparison between different SAE checkpoints is an enhancement, not a requirement to prove execution.
- Acceptance: real official evaluator runs, unaltered metrics import, result page opens. An empty upstream checkout is not a DEFER outcome.

### G4 — Install original circuit-tracer and obtain compatible model/representation

Use [decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer), which supports real native tracing and includes Gemma-2-2B and Llama-3.2-1B demos. It is **not** the same as Gemma-3-270M L5 SAE used by G1. Avoid mismatched model or transcoder identities.

- Install source in its own environment and verify the CLI with `circuit-tracer --help`; inspect original tutorial/demo for supported model and asset IDs. Prefer a compatible small officially supported model and transcoder, not an arbitrary SAE.
- Try one short prompt with exact upstream model/transcoder source. On 12 GB RTX 5070 use supported CPU offload/batch reduction first; if genuine native tracing requires more VRAM, migrate **only this native task** to an already available 24–32 GB GPU, if one exists. Never claim it ran on 5070 if it did not. Paid cloud allocation requires owner approval.
- Use `run_native_circuit_tracer` for actual generation, then locate the frontend v1 `{metadata,nodes,links}` JSON. If the original upstream emits a different format, implement a source-specific converter rather than inventing native graph data. Keep original .pt files in private GPU storage; public service reads only approved converted JSON.
- Render nodes, signed edges and error nodes; implement click, zoom, filter, pin/annotation and export; show source identity. Execute a real node intervention only where original source supports it, otherwise label that particular action unsupported while still completing native tracing and graph browsing.
- Acceptance: one generated native graph, actual converter output, a working public graph viewer and a local reproducible command. Absence of an installed package is not a scientific blocker.

### G5 — Create the missing small data asset, then train

Sparsify imports on the agent host but no snapshot was supplied. The upstream [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify) supports on-the-fly activation extraction and original model/data training; SAEBase's `TrainPlan`, `run_training` and `register_checkpoint` already provide integration.

- Prefer an ungated small **Pythia-70M** or officially compatible small model with a real named hook, not a 7B job. Materialize a **small operator-authored real-text corpus** (e.g. 200–1000 short distinct examples; this is an engineering training demo, NOT a quality benchmark) as a versioned local Hugging Face dataset that the installed native trainer actually loads. Freeze tokenizer, model and dataset revisions/hashes. Do not pass an incompatible `save_to_disk` folder to `load_dataset` unexamined.
- Set modest width/k, batch/context, no mixed-up model hook, and start a 50–200-step native **Sparsify or SAELens** run on one real CUDA GPU; if a native API changed, repair the wrapper against the installed pinned original, not a home-built estimator.
- Save native loss log and real native checkpoint; register the checkpoint, **load it through SAEBase**, encode/decode a real activation, and record reconstruction/sparsity. For a restart/resume badge, actually resume from native optimizer state. `--finetune` is not equivalent to `--resume`.
- Acceptance: documented data build → original trainer → saved checkpoint → reloaded SAE → real feature codes, with output in a Training page. A training config or importable `sparsify` alone is NOT completion.

### G6 — Download a real small local instruct model and compute an explanation

A 20 KB code stub is not a model checkpoint. Use an ungated compact *instruct* model with real `config.json`, tokenizer and safetensors, for example **Qwen2.5-0.5B-Instruct** as initial low-memory candidate. Pin its actual HF revision; if the local loader's default fp32 does not fit, either run it sequentially when other models are unloaded or add a correctly tested dtype option. Do not use a fake local folder.

- From genuine complete SAE traces, choose a feature with several activating and non-activating **independent prompt groups**. If the 20 original traces do not contain sufficient diverse examples, generate a small *new* original text batch and capture additional traces, keeping groups separated between discovery and held-out data.
- Write exact input JSON expected by `saebase/tools/wp12_autointerpret.py`: `site_id`, `feature_id`, `discovery`, `heldout`; each row includes actual source group, text and measured activation according to `backend.auto_interp`. Verify expected schema from current tests before running.
- Run `LocalAutoInterpreter.from_local_snapshot` on real weights; generate a candidate and separately score held-out activation predictions. Inspect failure cases and present score and limits. Require human review for the first public description; don't auto-approve an obviously meaningless hypothesis or call F1627 a geography concept from one Paris token.
- Acceptance: at least one original model-generated candidate from genuine examples, actual held-out score, reviewed display on a real feature card, and a reproducible local CLI. If one candidate is poor, improve examples or select a better-supported feature before declaring the whole interface impossible.

## 3. Shared environment and execution design

**Do not destabilize the working G1 environment to install unrelated tools.** Use separate Python virtual environments for G1/G2, SAEBench, circuit-tracer, native trainer, and local interpretation. Record `python --version`, `pip freeze`, CUDA, VRAM, upstream commit and disk. Install packages using official upstream sources and verify their actual CLI/API before constructing commands.

The 5070 is the primary real GPU. Run CPU-only tasks concurrently while preserving GPU model memory; sequence GPU training, circuit tracing and explanation when weights would overlap. Check `nvidia-smi` and `df -h` and select actual free storage before downloading. Keep small real datasets, experiment configs and lightweight nonprivate browser artifacts separately from model weights, raw traces, and prompt text.

Recommended private layout, adjust actual root to the GPU host:
```text
/workspace/wp15/
  venvs/{core,saebench,circuit,sparsify,interp}/
  sources/{saebench,circuit-tracer,sparsify}/
  saebase/assets/{models,saes,transcoders,datasets,authored-prompts}/
  runs/G1/ ... runs/G6/
  integration/public_candidates/
  receipts/{G1,...,G6,release_manifest}.json
```

Asset policy: prefer official ungated sources, cached weights and self-authored texts. A gated upstream asset must not be bypassed; try a correctly compatible openly available supported alternative or report the *exact* external access requirement. No purchase, paid API key or new cloud GPU rental without express permission. This is **not** an excuse to skip installing normal dependencies or creating a tiny permitted dataset.

## 4. Work order, parallelism and recovery

1. **WP15-A (CPU setup, immediately):** freeze source branch and six precise command contracts; install missing upstream in isolated environments, prepare allowed model/transcoder/data snapshots, implement deterministic asset discovery and straightforward source compatibility fixes. Start public UI integration of known LIVE-02/03 evidence in parallel.
2. **WP15-B (GPU):** finish G2 next because its existing code and cached G1 model are closest to ready; then G3 in its native environment (often CPU), and G6 small local instruction model. Do not wait for Circuit hardware to finish the rest.
3. **WP15-C (GPU):** do G5 short native training and G4 native circuit on their own compatible model/asset pair. If 5070 genuinely cannot support official tracing, raise a precise alternative-GPU requirement while shipping all other validated workflows.
4. **WP15-D (Integration):** show all six *real recorded* examples in the Research Lab, link actual local commands, correct labels and build a clean installation path. Polish visual hierarchy and remove only genuinely irrelevant disabled controls.
5. **WP15-E (Release candidate):** deploy isolated staging with approved small assets; test public browser (all six journeys) and local CLI examples, restart/data durability, auth and permissions, then merge the stacked PRs and update production Space with a retained rollback revision.

An error response must contain the failing command, observed traceback, source version, attempted fix, and next executable fallback. A missing module or missing `config.json` alone is **not an acceptable terminal task status**. Finish the repair or present a genuine external access/hardware blocker with reproducible proof.

## 5. Lightweight release gate

This is **product acceptance**, not 50 scientific certifications.

- G1: fresh capture + edit works, one approved published prompt supports end-to-end navigation.
- G2: true generated-token baseline/no-op/edit executions work and show source-linked per-step effects; do not require a fabricated text change.
- G3: original upstream native evaluation produced metrics displayed by SAEBase.
- G4: a real compatible upstream tracing job produced a graph; approved graph is interactive in the Lab.
- G5: a fresh original native SAE checkpoint has actually reloaded and encoded.
- G6: genuine local model produced an interpretable candidate with genuine disjoint held-out scoring, published only after review.
- Production: clean install/CLI, populated browsing, stable on restart, private services inaccessible to anonymous visitors, reasonable failure messages and documented GPU requirements.
- Each item has a working command, an artifact digest or source binding, one meaningful measured result, and one browser/API demonstration. One full system smoke is sufficient; **no historical 48-site rerun and no 50-item P8 audit**.

## 6. Delivery format and responsibility

**GPU Agent owns execution and missing-dependency repairs.** It must modify scripts and adapters as needed, commit real code on a feature branch based on WP13, and make a PR rather than only writing a status memo. **Integration Agent owns browser/data import, UI and deployment.** If the same Agent can do both, it should hand over a populated local staging UI and its exact reproducer; it must not self-certify an unchanged empty Space.

After each of G2–G6, update [task matrix](15_release_completion_matrix.csv) and a compact receipt under `/workspace/wp15/receipts/`. Send one consolidated handoff with per-feature command, actual result, source version, runtime, peak VRAM where applicable, small JSON and screenshots. Keep full weights/logits/raw traces private; small reviewed source-linked records may go in the PR when permitted.

Reviewers decide **working / fix-targeted / genuinely externally blocked** by trying the advertised operation and browsing its displayed result. They do **not** use a missing P8 signoff as a release objection.

**Go decision:** release after six actual workflows and staging work. If an unavoidable genuine hardware/license block survives compatible alternatives, disclose exactly that item and seek the owner's explicit scope decision; the Agent is not authorized to silently downgrade the goal.

### Useful references

- [WP14 agent receipt review](14_gpu_receipt_review_20261022.md)
- [WP13 original end-to-end handoff](13_gpu_agent_real_environment_handoff.md)
- [SAEBench upstream](https://github.com/adamkarvonen/SAEBench)
- [Circuit Tracer upstream](https://github.com/decoderesearch/circuit-tracer)
- [Sparsify upstream](https://github.com/EleutherAI/sparsify)
