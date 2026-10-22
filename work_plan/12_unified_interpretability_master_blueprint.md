# Work Plan 12 — OpenSAE Unified Interpretability Research Infrastructure

Date: 2026-10-21  
Status: **Strategic master blueprint. Not an implementation or GPU-completion claim.**  
Repository: https://github.com/OpenSAE/SAEBase  
Scope: Integrate, expose, validate, and maintain **all 50 named capabilities** drawn from 10 established SAE / mechanistic-interpretability projects.  
Execution companion: [GPU-agent handoff](12_gpu_agent_handoff_protocol.md)  
Machine-readable ledger: [50-capability acceptance matrix](12_capability_acceptance_matrix.csv)  
First step source audit: [WP12-00](12_p0_source_inventory_20261021.md)  
First GPU test specification: [GPU-00 P0 parity](12_gpu00_p0_parity_execution.md) · [frozen prompt pairs](12_gpu00_prompt_pairs.json)  
GPU agent's legacy-run outcome, **pending independent verification**: [GPU-00 reported results](12_gpu00_agent_report_20261022.md)

> **Final outcome.** A researcher can bring a model, an existing or newly trained representation, and a prompt; explore features; perform genuine interventions; compare SAEs; trace candidate circuits; analyze SAR/RSS/Input Localization; and publish a provenance-linked, independently reproducible experiment. A visitor without a GPU can explore the same real saved evidence. **All 50 features stay in scope; we phase their implementation, not their existence.**

This is the **next master plan**, not a replacement for previous scientific evidence, released artifacts, or Work Plans 01–11. Preserve their fixed protocols and historical records. Earlier Work Plans 04–07 contain valuable Research Lab / corpus / UI requirements; Work Plans 09–11 contain performance and GPU parity requirements. Reconcile them rather than blindly rewriting them.

## 0. Executive decision, authority, and definitions

1. **Single product:** OpenSAE is the organization; **SAEBase** is the unified research API and execution/provenance engine; **SAEBase Research Lab** is its browser-facing application; **SAEBase-Evaluation** is the public, rights-reviewed experimental-data entrance. Retain the independent SAR, RSS, and Input Localization research repositories with their original estimators.
2. **No artificial competition with upstream:** Use TransformerLens for model hooks, SAELens / Sparsify for training, NNsight and pyvene for additional intervention capabilities, the original circuit-tracer for graph computation, and native SAEBench evaluations. OpenSAE owns the integration, identities, user experience, evidence lifecycle, API stability, and verification.
3. **One capability can reuse another implementation:** e.g., Neuronpedia-style feature detail and SAE-Vis-style feature page must share an underlying feature record. Separate user-visible claims still receive separate acceptance tests.
4. **Scientific differences remain explicit:** SAE reconstruction vs residual-preserving edits; SAE vs transcoder / crosscoder; correlation vs causal intervention; attribution graph vs verified circuit; original SAR vs decoder-mediated experiments; prompt independence vs repeated variations.
5. **A claim must be earned:** “Code exists”, “button exists”, “external JSON imported”, “saved result can be replayed”, and “new user input can be computed” are different statuses. A marketing card must not collapse them.
6. **Never re-run the whole matrix as a default:** Preserve the earlier reported 18 configurations / 48 sites and 500 frozen prompt-group cohort; reuse exact source-verified archives where available and allowed. Historical measurement evidence is not proof of new-backend compatibility or an online inference service.
7. **Build one complete user journey before scaling to every model:** select actual prompt → inspect actual feature → edit / steer → compare actual outputs → inspect held-out evidence → export / rerun.
8. **Success is adoption and reliability, not a feature-count screenshot:** measure completion by an external researcher, reproducibility, native-tool numerical parity, meaningful usage, and upstream compatibility.

### 0.1 Audit baseline: what is established, and what is not

The current repository README documents model/SAE registration, cached activations and feature codes, reconstruction and feature edits, behavioral readouts, reliability, optional suffix replay, generation experiments, original SAR/RSS/Input Localization adapters, historical validated sites, and a CPU Research Lab. This is valuable infrastructure, not a blank slate.

* [SAEBase README](../README.md), [API](../docs/python-api.md), [verified model matrix](../docs/verified-model-matrix.md), [performance](../docs/performance.md), [research integrations](../docs/research-integrations.md).
* [Research Lab README](../apps/research-lab/README.md) states that the public CPU endpoint is read-only and new GPU jobs are disabled; recent 500-prompt raw archives are not included in its public Docker image. Do not claim universal live inference or publicly downloadable raw records.
* [v0.4.0 notes](../docs/releases/v0.4.0.md) distinguish software release, private GPU measurements, public data rights, and future hosted GPU service.
* Native SAEBench **selected result import** exists, but it is not yet a unified command that executes the entire native upstream benchmark suite.
* Existing loader code [models/hooked_lm_common.py](../models/hooked_lm_common.py) calls legacy TransformerLens \`HookedTransformer\`, and current unconstrained dependencies in [pyproject.toml](../pyproject.toml) allow major-version drift. **TransformerLens 4.0 removed legacy \`Hooked*\` classes on 2026-09-21.** Audit and repair version compatibility first. For an exact frozen legacy environment, use a pinned supported prior version; provide a tested bridge adapter for the current interface. Do not silently switch weight-processing conventions.
* Work Plan 11 describes execution on an RTX 5070 (~12 GiB). The user has also mentioned access to an RTX 5090 (32 GiB) for another workload. **Do not assume these are the same host or that either is available**. GPU agent must record the hardware actually allocated to each run.

Everything in this plan marked “target” is **not** claimed to be implemented. WP12-00 has now completed an initial **source-only** inventory on branch `feat/wp12-p0-foundation`: 24 PARTIAL, 24 NOT_STARTED and 2 BLOCKED, pending independent review. No fresh CPU test, GPU parity or live deployment verification was done in that inventory, so the P0 gate is still **open**. The [GPU-00 runbook](12_gpu00_p0_parity_execution.md) gives the GPU agent a fixed C1 handoff without changing scientific results. New verified statuses require source-linked execution evidence and separate reviewer signoff. **Update after GPU-00:** an agent reports 4/4 site-level parity PASS for legacy TransformerLens 3.9 / SAELens 6.54 fp32 on Pythia-70M L3/L5 and Gemma-3-270M L5/L9. The raw evidence is not yet accessible to the independent reviewer, TransformerLens 4 was not tested, and P0 remains open. See [the dated report](12_gpu00_agent_report_20261022.md).

## 1. Upstream product study: match useful behavior, not branding

These ten upstream projects form the reference experience. Their interfaces and documented semantics are a specification baseline, **not a license to copy their names, assets, services, model weights, or private data**.

| Reference | Exact interaction to emulate in OpenSAE | Reuse strategy | Documentation |
| --- | --- | --- | --- |
| **Neuronpedia** | Feature route by model / source / index, top active tokens and explanations; search explanations, search by live inference, top features by token; split baseline/steered response; candidate graph with pin/group/steer | Official API / approved bulk exports, existing public feature data with exact source identity; OpenSAE inference and reports for private/local runs. Bulk use exports instead of calling API once per feature; respect rate limits | [feature dashboards](https://docs.neuronpedia.org/sparse-autoencoders/feature-dashboards), [search](https://docs.neuronpedia.org/sparse-autoencoders/search), [steering](https://docs.neuronpedia.org/steering), [API](https://docs.neuronpedia.org/api-and-data/api) |
| **TransformerLens** | Select real hook locations, record/cache intermediate states, patch clean/corrupt runs, visualize per-component contribution | \`TransformerBridge\` for new supported 4.x runs; pinned legacy compatibility adapter; no reimplementation of Transformer internals | [4.0 release](https://transformerlensorg.github.io/TransformerLens/content/news/release-4.0.html), [hooks](https://github.com/TransformerLensOrg/TransformerLens/blob/main/docs/source/content/hook_system.md) |
| **SAELens** | Load pretrained SAE, set training configuration, select SAE family, run/train/resume, analyze feature activations | Native \`sae-lens\` model and training APIs pinned in isolated environment; reuse public checkpoints and existing SAEBase adapter | [training](https://github.com/decoderesearch/SAELens/blob/main/docs/training_saes.md), [releases](https://github.com/decoderesearch/SAELens/releases) |
| **NNsight** | \`trace\` intermediate values, mutate tensors, invoke multiple inputs, edit during generation, optionally run on NDIF | Official nnsight execution adapter; NDIF remote only for deployed models with user authorization, documented quotas and failures | [trace](https://github.com/ndif-team/nnsight/blob/main/docs/usage/trace.md), [NDIF](https://github.com/ndif-team/nnsight/blob/main/docs/remote/index.md) |
| **pyvene** | Configure locations / token steps / neuron subsets, compose interventions, save/reload complete configurations | Native \`pyvene\` interventions behind a semantic adapter, not a universal claim that all operations are equivalent | [repo](https://github.com/stanfordnlp/pyvene) |
| **circuit-tracer** | From compatible model / transcoders and target output generate candidate graph, prune, inspect connections, annotate and pin supernodes, intervene | Use original attribution engine and graph outputs; implement a consistent OpenSAE graph viewer and experiment links | [repo and demos](https://github.com/decoderesearch/circuit-tracer), [graph exploration](https://docs.neuronpedia.org/circuit-tracing/explore-graphs) |
| **SAE-Vis** | Feature-centric and prompt-centric token displays, correlation pane, self-contained visualization/export | Reproduce the useful presentation interactions over OpenSAE data; upstream project is not actively maintained, so do not depend on its server long-term | [repo](https://github.com/callummcdougall/sae_vis) |
| **SAEBench** | Run native evaluation suites, load custom \`encode/decode\` SAEs, compare across matched sparsities, browse native plots | Execute original code in pinned evaluation worker; import original JSON without redefining metrics; isolate credentialed / licensed tasks | [repo](https://github.com/adamkarvonen/SAEBench) |
| **Gemma Scope 2** | Browse asset catalogue by exact model, layer, site, representation type, width and sparsity; load SAEs / transcoders / cross-layer assets | Register official Hugging Face manifests and weights with precise license, tokenizer, scaling and revision; do not train a duplicate of the entire suite | [official model guide](https://huggingface.co/google/gemma-scope-2-270m-pt), [DeepMind](https://deepmind.google/models/gemma/gemma-scope/) |
| **Sparsify** | Select TopK, data, hookpoints, training mode; stream activations; resume; run distributed job | Native EleutherAI trainer and \`torchrun\`, separate pinned environment; use checkpoint adapter and training metadata | [repo](https://github.com/EleutherAI/sparsify) |

**Mandatory site-reference checks:** Before implementing each UI, capture the current official documentation URL, the exact user journey, its input/output types, authentication / capability restrictions, the upstream version or commit, and the OpenSAE equivalent behavior. Preserve credits and do an independent UI design; do not copy logos or protected site assets.

### 1.1 Target Research Lab information architecture

Use a consistent navigation organized around *what users do*, not upstream library names:

* **Explore:** Choose model + exact SAE + layer/site; search by explanation, prompt inference or top-k-per-token. Feature detail page shows real top examples, highlighted tokens, logits/contributions where valid, explanation provenance and score, frequency distribution, correlation, linked effects, support/negative examples. Stable canonical feature URL.
* **Intervene:** Clean vs edited output side by side; feature selector; ablate / scale / clamp / add / coalition / reconstruction; select token step and site where supported; logit changes separated from actual generated text; a clear “Saved experiment” vs “Run on my GPU / authorized worker” indicator.
* **Circuits:** Select a target quantity, compatible transcoder source, prompt; generate graph if compute exists; pan/zoom, node/edge details, token / error nodes, pruning thresholds, pin and group supernodes, annotate, export JSON, launch an intervention from a pinned node. Never render a static demonstration as a newly computed graph.
* **Compare:** Choose two compatible SAEs, datasets, protocols and sparsities; reconstruction/sparsity plots, native SAEBench outcomes, SAR/RSS/Localization panels and CI. Incomparable pairs show *why* and disable unsupported comparisons.
* **Train:** Choose available base model, training backend (SAELens or Sparsify), site, architecture, width, sparsity, dataset and budget; run preflight, inspect progress, resume, register resulting assets, then “Open this SAE in Explore”.
* **Datasets & Reports:** Rights-reviewed manifests; source versions; experiment registry; immutable report view, download and reproduce. No public export of private prompts without explicit consent.
* **Help:** A 5-minute “first real feature”, a small-model runnable notebook, API reference, capability matrix, known restrictions, and a reproducibility guide.

Do not duplicate the older Research Lab's five-module visual system for the sake of naming; preserve its best validated screens and update navigation through one design system.

## 2. Architecture: connect existing backends without losing their meaning

### 2.1 Minimum shared entities and invariants

**ModelIdentity**: upstream model ID, commit/revision, base vs instruction, tokenizer ID/revision, architecture, dtype/quantization, adapter, model-processing mode.

**RepresentationIdentity**: \`sae | transcoder | crosscoder | cross_layer_transcoder\`, checkpoint hash, source repo/revision, layer(s), input hook(s), target hook(s), encoder/decoder shapes, normalization, skip connection, decoder constraints, feature vocabulary/indexing, native backend.

**InputIdentity**: original source/dataset revision/rights, prompt group ID, rendered text and hash, token IDs, BOS / chat template, sampling and selection split, token/position policy.

**ActivationCapture**: model+site+input+backend+precision+software identity; full/tapped tensors only where capability allows; sparse feature codes by exact representation; keep source references and verify checksum. Do not treat equal string model IDs as cache-safe.

**InterventionSpec**: typed operation; baseline state and target, feature IDs/sets; token position(s), decoding phase, strength units, residual-preserving vs full reconstruction, source & target sites, random seed; valid only when the selected backend advertises that operation.

**EvaluationSpec**: exact upstream estimator and commit, metric definition, data split, target selection, aggregation unit, confidence interval method and baselines. Different FVU, KL or behavior scores are not interchangeable.

**GraphRecord**: exact input/model/transcoder, attribution target, node types (including error nodes), edge scores, pruning settings, source method, graph completeness diagnostics where supported, provenance and human annotations separate from computed data.

**ExperimentRecord**: immutable content-addressed run ID, upstream package commits, device/backend, configuration, artifacts, raw measurement hashes, controls, warnings, timing/peak VRAM, numerical parity result, sharability and rights, created-by/source.

Implementation rule: adapt these to existing SAEBase registry / manifests / reports where possible. Do **not** perform a mass refactor that invalidates archived references. Add compatibility fields, schema migrations and regression tests only as needed.

### 2.2 Backend selection is capability-aware

Expose operations, not fictional “all backends are equivalent” checkboxes. A capability matrix must report for each model / representation / backend:

\`load\`, \`capture\`, \`gradients\`, \`cache\`, \`patch\`, \`generate_intervene\`, \`train\`, \`graph_attribute\`, \`graph_intervene\`, \`distributed\`, \`remote\`.

The TransformerLens 4.0 local transformers Driver supports fuller hook/gradient access; its vLLM Driver exposes a different set of declarative interventions without autograd. Gradient-based circuit attribution cannot be routed to the vLLM path. NNsight NDIF only applies to models actually deployed there; network service access is not inferred from package installation. pyvene operation semantics depend on architecture and intervention class.

A backend adapter **must fail clearly** when an operation is unsupported. Automatic fallback requires a tested equivalent protocol; otherwise ask for an explicitly different protocol with a new run ID.

### 2.3 Canonical research sequence

\`model + SAE/transcoder + input\` → identity verification → activation capture → feature/circuit discovery → controlled edit/attribution → native metrics + SAR/RSS/Localization → reliability, controls, provenance → immutable report → optional reviewed publication.

Always distinguish (a) offline saved results, (b) local direct execution, (c) authorized remote execution. No synthetic interpolated outputs or fake model responses.

### 2.4 Environment separation

One minimal core install; optional pinned **model**, **training**, **benchmark**, **circuit**, and **GPU serving** worker environments. Share serialized experiment contracts and checkpoint identities, not a fragile global \`pip install everything\`. Document versions and lock hashes. Preserve Python 3.11/3.12 compatibility where actually verified. Old TL model-loading compatibility may require a frozen legacy worker; new TL 4.x uses \`TransformerBridge\` and must prove parity before promotion.

## 3. Nine phases and the exact 50-capability allocation

**P0 is an audit and compatibility gate, P1–P7 assign all 50 primary capabilities once, P8 is independent release and adoption.** Later phases extend earlier ones, but do not relabel earlier capabilities as newly done.

| Phase | Main output | Primary feature IDs | Count | CPU/local vs GPU strategy |
| --- | --- | --- | ---: | --- |
| **P0 — inventory, freezes, compatibility** | Baseline ledger, reference environments, data rights, 10 official UI reference journeys, frozen small-model evidence | None (precondition) | 0 | CPU-led, tiny GPU parity smoke only |
| **P1 — common model and representation engine** | Cross-library loading, activation cache/patching, inference trace, scope-checked Gemma assets | 06–11, 14, 16–18, 41–42, 45 | 13 | CPU contract + tiny model GPU; no 48-site sweep |
| **P2 — feature explorer and interpretation** | Feature dashboard/search/auto-interpretation; token and correlation visualizations | 01–03, 15, 31–35 | 9 | Use verified existing measurements; GPU only for live prompt/search pilot and targeted missing examples |
| **P3 — general interventions and generation** | Real steering, token-step interventions, pyvene composition & saved configs | 04, 19, 21–25 | 7 | Tiny GPU correctness, bounded 2–3 model/site confirmations |
| **P4 — native benchmarking and comparison** | Execute actual SAEBench, custom SAE, fair comparison and SAR/RSS/Localization links | 36–40 | 5 | CPU parsing and charts; GPU pilot then justified full evaluation |
| **P5 — training and new asset lifecycle** | SAELens/Sparsify training, architecture/config selection, resume, registry, multi-GPU | 12–13, 46–50 | 7 | Small end-to-end trainer first; large/distributed is conditional and separately authorized |
| **P6 — circuit and cross-layer research** | True candidate graph, visual editing, node interventions, transcoders/cross-layer assets | 05, 26–30, 43–44 | 8 | CPU graph viewer with verified upstream examples; autograd GPU pilot for graph generation |
| **P7 — remote execution and operational scale** | NDIF and other authorized worker route; durable queues and resource policies | 20 | 1 | Remote smoke with actual access; no default paid deployment |
| **P8 — independent acceptance and broad release** | 50-item source-bound verification; external users; documentation, release and adoption telemetry | None (final gate) | 0 | GPU only for failed coverage/parity gaps; CPU docs, CI, users |
| **Total** | | | **50** | |

Dependencies: **P0 → P1 → P2 → P3** for the first complete user path. P4 evaluation can start after P1 and may overlap P2/P3 when protocols are frozen. P5 training can overlap P4 after asset identity / result formats stabilize. P6 relies on P1 and a reliable P3 intervention engine. P7 uses stabilized outputs from P1/P3/P6. P8 tests every stage.

No timeline is a promise: estimate calendar only after P0 inventory, upstream environment locking, and the first GPU rate measurement. For scheduling, prioritize **P1/P2/P3 for a genuinely usable first research loop**, then run P4/P5 in parallel, then P6/P7/P8.

## 4. Detailed execution stage cards

### P0 — evidence-based inventory and compatibility (mandatory first)

**Goal:** Know which of the 50 abilities actually exist, which merely have stored examples, which require new model runs, and which have no working path.

**Local CPU/strategy agent responsibilities**
1. Check current \`main\`, the WP11 stacked PR state, CI and release tags. Do not rewrite accepted WP11 experiments or claim experimental branches are merged. Inventory existing routes and engines, classify every feature as \`NOT_STARTED | PARTIAL | CPU_VERIFIED | GPU_VERIFIED | RELEASED | BLOCKED\`; attach exact commit, test path and artifact link. Mark all unknown as unverified, not passed.
2. Freeze 10 upstream reference workflows from §1, with current versions, menu semantics and screenshot/checklist *for internal comparison*. Record each upstream license, model/data license, API rate limits, allowed redistribution and login/compute constraints.
3. Run core install, unit tests, public Python API and CPU Lab in a clean environment. Inspect broken pages and offline execution paths; no new feature implementation before repeatability is known.
4. Produce a two-track TransformerLens compatibility decision: frozen legacy worker vs new \`TransformerBridge\` adapter with weight-processing mode and parity tests. SAELens v6 and NNsight/vLLM versions require their own pinned worker lockfiles.
5. Define immutable v1 identities/JSON schema, version migration, capability registration, report artifact paths and comparison eligibility. Integrate with existing registries rather than inventing a parallel data system.
6. Verify rights and hashes for the existing 500-group cohort and feature cards; if raw files are unavailable, mark \`REQUIRES_AUTHORIZED_ARCHIVE\` and use legitimate smaller datasets—not made-up data.

**GPU agent:** first follow [GPU-00](12_gpu00_p0_parity_execution.md): two small model families (Pythia-70M and Gemma-3-270M), two registered SAE sites each, eight original clean/corrupt prompt groups. Run its C0 host/environment/tokenization preflight and exact native-versus-SAEBase C1 checks before requesting broader compute. Pin actual model/SAE hashes, logits, code and patch evidence, and record wall time/VRAM. This initial run measures the supported legacy backend: do **not** claim TransformerLens 4 bridge parity without separately executing a tested new bridge.

**Gate P0:** clean CPU install+CI; exact dependency lock; baseline registry of 50 with evidence links and honest states; one simple real-model capture/edit result; no silent compatibility changes. **Do not queue a large run yet.**

### P1 — foundations: models, activations, hooks, pretrained assets

**IDs:** F06/F07/F08/F09/F10 (TransformerLens), F11/F14 (SAELens), F16/F17/F18 (NNsight), F41/F42/F45 (Gemma Scope 2).

**How:** Keep existing SAEBase activation caching, identity checks, token alignment, patching and decoder operations. Add \`transformerlens_bridge\`, \`nnsight_local\`, \`saelens_pretrained\`, \`gemma_scope_2\` adapters behind the same model/representation records. Support *actual* module/hook discovery and source/target site validation. Implement patching with clean/corrupt controls; caching filters to avoid unnecessary tensor dumps. Provide reusable component/readout analyses through the verified local-capable backend. Expose original backend version and supported operation flags.

**User-visible parity:** \`Explore → Select model/SAE/site\`; a hook/activation inspector akin to TransformerLens tutorials; compare clean/corrupt activation patch effects; a Gemma Scope asset catalogue with model/site/width/L0 selection; “Open feature” link from a selected asset.

**Local CPU tasks:** Adapter contracts, registry UI, model asset index, data rights, cache schemas, synthetic and recorded-data regression, token UI, capability-matrix rejection tests.

**GPU agent tasks:** A very small Pythia/Gemma-270M load+capture+reconstruction+patch test; one Gemma Scope 2 actual SAE and one compatible non-SAE representation *only if official asset fits the card and provides needed semantics*. Pair inputs with frozen tokenization. Confirm reference backends independently, and validate candidate adapter numerics before enabling it.

**Scale:** **C1 smoke mandatory. C2 2–3 verified sites conditional. C3 broad site sweep forbidden.** Import pretrained official asset metadata CPU-only; download weights lazily and verify checksums; do not mirror model families.

**Gate P1:** all 13 capabilities each have at least one real validated supported route, unsupported combinations fail clearly, legacy/new TL behavior is explicitly versioned, and clean/corrupt intervention results are faithful to the declared protocol.

### P2 — Neuronpedia-style Explore + SAE-Vis-style interactive evidence

**IDs:** F01/F02/F03 (Neuronpedia), F15 (SAELens analysis), F31–F35 (SAE-Vis).

**How to match actual websites:** Reuse the current Feature Atlas + Research Lab layout. Feature route contains a stable exact \`model/representation/site/feature\` identity, next/previous feature, top activation contexts with token highlights and absolute activation values, baseline and frequency distributions, explained logits only where valid, explanations with source/model/method/score and counterexamples. The Search screen has *separate* tabs: **by explanation**, **by inference on this prompt**, **top features for selected token**. Do not pretend explanation-text search is a forward pass. Prompt view supports per-token feature rankings. Correlation panel labels corpus and sampling and does not equate correlation with identity.

**Auto-interpretation:** Generate candidate descriptions from discovery examples; score with held-out examples and negatives; display provenance, unsupported/uncertain, human edits as separate labels. Require explicit spend limit for external model APIs. Do not generate confident descriptions for unobserved features. Import Neuronpedia-approved public data only with exact source identity and rights; use bulk exports for large ingestion rather than many API calls.

**Local CPU tasks:** Feature indexing, full-text/embedding search over published labels, fast page queries, front-end, accessibility, private-vs-public controls, exportable static HTML, correlation computing from already captured codes, explanatory copy, 10 curated real examples and negative examples.

**GPU agent tasks:** 8–32 custom prompts for the live-inference search reference; targeted activation capture when a selected card lacks samples; 20–50-feature pilot explanation score validation (or external API separately authorized). Reuse 500-prompt caches if same identities and rights; no wholesale recapture.

**Scale:** **C0 for most features; C1 for live prompt lookup; C2 only to fill audited gaps.** Avoid computing dense all-feature × all-token displays. Store sparse top indices and paginated feature records; never assume absent from sample means globally dead.

**Gate P2:** ten authentic demonstrations; user finds a feature via all three routes; tokens and scores match exact raw artifacts; exports open offline; explanations can be traced to generating model/data and are visibly hypotheses; feature index remains responsive on large dictionaries.

### P3 — real interventions, generation and composability

**IDs:** F04 (Neuronpedia steering), F19 (NNsight generation edits), F21–F25 (pyvene).

**How to match websites:** User selects a real feature, token step(s), site and operation; baseline left / edited right. Show actual baseline and edited logits/text, fixed decoding protocol, effect charts and negative control. Support feature selection from Explore. Advanced panel provides pyvene-style multi-location, neuron subset, parallel/sequential composition, generation-step targeting and serializable configurations.

**Engineering:** Wrap native pyvene and NNsight operations with typed \`InterventionSpec\`. Keep existing SAEBase identity, ablation/scale/clamp/add/coalition/reconstruction and cached paths. **No silent replacement of SAE-code manipulation with a decoder-direction add.** Annotate whether an operation is residual-preserving; \`x' = x + decode(z') - decode(z)\` is different from \`x' = decode(z')\`. Non-SAE interventions may operate on a different site/representation and require different validation.

**Local CPU tasks:** Interface, config import/export, invalid-composition detection, guardrails, queue preflight, saved comparison cards, reproducibility manifest, controlled failure displays, cross-architecture **synthetic/PyTorch CPU** tests.

**GPU agent tasks:** Verify identity/no-op, zero/scale/clamp, one composed intervention, token-step generation and corresponding original tool output on 2–3 chosen model/site pairs. Run deterministic teacher-forced readout parity separately from stochastic text samples. Test KV-cache vs full-prefix reference when backend claims equivalence. Test at least one genuinely non-Transformer architecture under pyvene where upstream supports it, CPU if possible; do not claim universal support based on one Transformer.

**Scale:** **C1 mandatory; C2 bounded. C3 not justified for shipping UI.** Interventions may be repeated on 20–100 well-designed inputs with paired controls; only expand when effect reliability needs it.

**Gate P3:** source-backed end-to-end prompt→feature→edit→actual output→rerun succeeds. Exact operation recorded; no-op parity; unsupported modes visibly blocked; configs reload and reproduce within declared numerical tolerance.

### P4 — native SAEBench, fair SAE comparisons and scientific integrations

**IDs:** F36–F40.

**How:** Run **native** SAEBench entrypoints in pinned workers and separately import their untouched result records. Support appropriate Core, sparse probing, AutoInterp, feature absorption, RAVEL, SCR, TPP, unlearning and other presently supported benchmark modes, **subject to data/credential rights and upstream capabilities**. Do not claim all are ready if WMDP/unlearning access or AutoInterp API key is absent. Provide exact metric definitions, source commit, used examples, checkpoints and native outputs. Native result import is not execution.

**Website parity:** Benchmark page with evaluation selection, runnable configuration, target SAE, compatibility preflight and task status; comparison view with matched model/site/dataset/sparsity and Pareto-style multi-metric plots. “Not comparable” must be first-class. SAEBench recommend comparisons across a range of L0 settings: preserve sparsity axes.

**OpenSAE research integration:** Original SAR measures coalition non-additivity; RSS selection/evaluation identities remain separate; Input Localization requires complete controlled input designs and exact factor ordering. These are independently labeled analyses, not replacement SAEBench metrics. Preserve optional decoder-mediated intervention as a separate protocol from original SAR assumptions. Keep prompt-group resampling and negative controls.

**Local CPU tasks:** Native worker wrappers/containers, sample and checkpoint eligibility, result importer and source verification, exact plot implementation, provenance and fairness constraints, CI replay from saved native outputs.

**GPU agent tasks:** Start with Core + Sparse Probing on one compatible small checkpoint; compare numerical output against running upstream directly in the exact same environment. Then run one additional held-out downstream benchmark for two matched SAEs or matched sparsity points. Only after this pilot choose the full native suite and budget. Reuse upstream setup/cache when semantics allow.

**Scale:** **C1 native smoke; C2 benchmark core subset; C3 entire benchmark suite only if explicitly funded and prerequisites pass.** Upstream SAEBench documentation reports ~177 minutes shared setup plus ~115 minutes per SAE on one historical RTX 3090 / Gemma-2-2B protocol, **not a forecast for our GPU**. Cache may use ~100 GB; evaluate disk before launch.

**Gate P4:** at least two native modes executed from OpenSAE and independently matched to upstream outputs; same-model/site dual-SAE fair comparison, confidence intervals and exclusions; blocked restricted evaluations correctly marked; one shareable complete report.

### P5 — training, resuming, and automatic asset registration

**IDs:** F12/F13 (SAELens), F46–F50 (Sparsify).

**How to match trainers:** User selects \`train SAE / train transcoder\`, model & hookpoint, dataset, architecture and sparsity, expansion/width, training steps and checkpoint plan. Show resource estimate, loss, L0, reconstruction, dead features where defined, recovery and link to resulting registered SAE. Reuse SAELens training runner and Sparsify TopK/on-the-fly activation + hookpoint selection. Different training backend outputs must map into typed representation identities; **do not claim all architectures support every trainer**.

**Local CPU tasks:** Native config validator, training manifest, dataset identity, checkpoint-atomic write/read, artifact registry import/export, log UI, resume metadata validator, fake-worker fault-injection tests, unit-tested scheduler. Keep separate envs where conflicts arise.

**GPU agent tasks:** (1) tiny real-model short training using SAELens with verified restart; (2) tiny Sparsify TopK on-the-fly training and transcoder mode; (3) compare imported encodings/reconstruction to native results; (4) one scientifically useful moderate run **only after** pilot proves resources. Run 2-GPU distributed via actual \`torchrun\` only on a verified multi-GPU allocation; a single 5090 or 5070 may test configuration syntax and single-rank mechanics but does **not** satisfy F50 multi-GPU acceptance.

**Scale:** **C1 functionality pilot mandatory, C2 moderate evidence if needed, C3 extensive token training / distributed only with approved dataset, budget, checkpoint policy and hardware.** Pretrained Gemma Scope 2 models supply broad asset coverage without re-training them.

**Gate P5:** training result is not a toy placeholder: real checkpoint can load, encode and decode, resume reproduces declared training state, native and OpenSAE readouts agree, and “Open in Explore” works. Actual 2+-GPU run and a successful interrupted/resumed distributed checkpoint required for F50 VERIFIED; otherwise mark BLOCKED pending resources.

### P6 — genuine circuit tracing and cross-layer feature study

**IDs:** F05 (Neuronpedia-style graph exploration), F26–F30 (circuit-tracer), F43/F44 (Gemma Scope 2 transcoder / cross-layer).

**How to match websites:** Choose a documented model + supported transcoder set + target logit / output. Generate graph using original circuit-tracer on a gradient-capable backend; display tokens, features, edges, attribution signs/weights and **error nodes**; filter/prune, zoom/pan, click to inspect feature examples and score, pin candidate nodes, group supernodes, annotate, save/share/download JSON. “Intervene” from selected node calls the validated P3 execution backend with appropriate transcoder semantics. Reproduce a published upstream example before designing a new one.

**Representation engineering:** Explicitly support normal SAE, MLP transcoder, skip-transcoder, partial crosscoder and cross-layer transcoder as **different typed assets**. Read official Gemma Scope 2 manifest coverage: across Gemma 3 270M–27B SAEs/transcoders; cross-layer transcoders only for models actually published (not blanket 27B claims). Missing asset → clear unsupported status, not generated imaginary circuit. Start with a small compatible released circuit-tracer dataset/example.

**Local CPU tasks:** Asset catalog, graph schema and provenance, graph importer/exporter, interactive viewer, annotation persistence with author/source, small precomputed graph fixtures, URL sharing and data permissions, heuristic-free rendering.

**GPU agent tasks:** First reproduce one upstream attribution graph on a compatible small model (e.g., Gemma-2-2B using original tools if the card fits). Then test one newly specified prompt/target, graph generation, pruning invariants and at least one feature/supernode intervention; record raw and pruned graph separately. Test cross-layer assets with official checkpoints only after feasibility preflight. Attribution performance requires autograd and may have high RAM/VRAM pressure; vLLM-based generation is **not** a substitute for gradient graph computation.

**Scale:** **C0 CPU visualization; C1 small graph+intervention; C2 a small set of new graphs. C3 large batch of graphs or very large models only after explicit scientific need.** No requirement to regenerate every upstream graph.

**Gate P6:** one reference graph reproduced using native upstream outputs; new graph created for a valid input; annotations/groups survive save/load; graph node identities and error contributions remain intact; perturbing a pinned feature yields actual model readout and does not imply proof of a unique circuit.

### P7 — remote NNsight and operational execution at scale

**ID:** F20 plus hardening of capabilities previously validated locally.

**How:** User can choose “Run locally”, “Run on my connected worker”, or “NDIF (supported models only)”. The same ExperimentRecord follows all modes while runtime-specific semantics remain visible. Implement auth-scoped job submission, idempotency, leases, cancel, retry, queue resource tiers, ephemeral secrets, result upload, private defaults, audit and quotas. Distinguish NDIF deployed model coverage from generic remote server or own GPU worker.

**Local CPU tasks:** Agent protocol, queue state machine, auth and permission tests, export/import, timeout/OOM simulation, backend-capability preview and cost/rights controls. Public read-only Lab must continue working during GPU outages.

**GPU/remote agent tasks:** One real local-worker remote request and one real NDIF trace **if user credentials and hosted model are available**, comparing supported readouts with a reference. Try cancel/worker crash/retry recovery and unauthorized request denial. Do not buy cloud GPUs, upload private prompts, or activate a public GPU endpoint without explicit budget/permission. If NDIF is unavailable, leave F20 labeled BLOCKED for that route while own-worker route may still function.

**Scale:** **C1 network/GPU smoke, C2 load test if a service is actually being launched, C3 multi-user hosted service only after a separate approved operations budget.**

**Gate P7:** genuine authorized remote trace executes and returns source-linked outputs; authorization/quotas and durability tests pass; public outage degrades gracefully to saved-data browsing.

### P8 — independent 50/50 acceptance, docs and external adoption

**Goal:** Earn—not assert—a stable release that demonstrates each capability in its stated supported scope.

1. Another agent/reviewer reruns tests from fresh pinned environments and checks unmodified upstream reference outputs. Every feature row must contain: upstream behavior link, OpenSAE entrypoint, exact model/asset/backend scope, CPU test, if appropriate actual GPU evidence, provenance/report and limitations.
2. At least one external researcher reproduces a complete prompt→feature→intervention→report experiment; target an early cohort of 12 external users with 10 completing a small-model task without modifying library source or needing synchronous maintainer guidance. This is a **proposed target, not a current achievement**.
3. Cross-backend contract tests on each *declared* supported combination: tokenizer alignment, no-op, clean logits, captured activations, code shape, source version, loss semantics, network failure and sample group integrity.
4. Publish docs in user-task order; create 5–7-page guides for first real feature, comparative evaluation, training-to-Explore and graph-to-intervention. Main README remains short and useful, with technical caveats in linked docs.
5. A product smoke checks actual deployed HF Space routes, mobile/desktop legibility, required login and expiry, downloadable results, rollback and immutable release checksums. “Public live GPU” claim requires an actual durable GPU service.
6. Get source/data redistribution approval before publishing corpora or features from external services. Preserve earlier releases, original paper estimators and checkpoint rights.
7. Compare end-to-end tasks against the *appropriate upstream combination* in matched scope: success rate, setup burden, wall time, peak VRAM, result parity, recovery, reproducibility and UX comprehension. Do not imply beating an upstream library at a task it never intended to support.

**Gate P8:** 50 individually signed acceptance rows with scope and artifacts, no known blocking integrity defects for the released paths, clean installation, real external user task success, full rollback path. Capabilities that require inaccessible proprietary/credentialed services or unavailable multi-GPU hardware remain explicitly blocked—not magically marked complete. The **50-feature target remains unchanged**.

## 5. CPU and GPU separation: decide on computational necessity, not the feature name

### 5.1 Workload classes

| Class | Examples | What runs where | Expansion policy |
| --- | --- | --- | --- |
| **C0 — zero new model compute** | Website UX, docs, indexing, feature descriptions already licensed, charts, exports, graph viewing, schema and unit tests | Local strategy/code agent; CPU CI; existing authentic records | Default first action |
| **C1 — micro GPU** | Load/encode/reconstruct, live prompt lookup, identity edits, one graph, one short training/resume | One GPU agent; 8–32 frozen prompts/representative sites; small model first | Mandatory for claims involving real inference; stop on mismatch |
| **C2 — bounded evaluation** | 2–6 audited sites × up to 100–500 groups, paired interventions, native benchmark subset, 2 matched SAEs | GPU agent with preflight manifest, checkpoints, quotas; CPU aggregation | Only when C1 passed and pilot indicates scientific value |
| **C3 — large / distributed** | 48 sites × 500 groups re-capture, exhaustive coalitions, long SAE training, full SAEBench, many attribution graphs, multi-GPU | Dedicated scheduled GPU agents/authorized rented resources; **not automatic** | Require written need, existing-artifact reuse analysis, time/VRAM/disk prediction, cost approval, checkpoint/recovery and stop rules |

**C1 and C2 sizes are proposed starting scopes, not performance promises.** Agent uses measured rates, not fabricated throughput.

### 5.2 Calculating workload before execution

Every GPU preflight emits:
* counts of **unique model forward passes** by model/revision and by activation site; cached-vs-uncached breakdown,
* prompt groups × token lengths, target site(s) and microbatch size,
* SAE width/nonzero representation, dtype and estimated active activation bytes,
* number of interventions and generated tokens under the *actual* protocol; count coalition requests separately,
* shared setup vs per-checkpoint benchmark costs, estimated peak VRAM/RAM/disk and throughput range from C1 pilot,
* source checksums, rights, required credentials, expected outputs and failure/retry budget.

For standard SAR, one possible protocol at K16, M384 on 3 sites × 500 groups would require **3 × 500 × 384 = 576,000 coalition measurements** before controls/retries. A 48-site analogue is **9,216,000**. Neither is necessary merely to prove that the SAR function exists; reuse the original validated 500-prompt experiments with exact scope where possible. Distinguish unique base prompts from derived coalitions and variants.

Activation memory is driven by batch × sequence × hidden size × sites × bytes and retained graph/gradient states; disk is driven by whether dense activations or sparse codes are stored. Avoid retaining unnecessary dense feature × token matrices.

### 5.3 Where to draw the local-vs-GPU-agent line

**Strategy/local CPU agent must complete before any GPU launch:** source research, architecture/interface work, page designs, synthetic regressions, sample manifest, scope/rights checks, exact configs, preflight, no-op expected behavior, release and report format.

**GPU agent owns:** real checkpoint and source validation, actual CUDA forward/gradient/backward/training, numerical comparisons to untouched upstream, throughput & VRAM measurements, model-specific compatibility and large/medium job orchestration. It must write raw measurements and manifests, not rewrite scientific conclusions to match expectations.

**Local strategy agent after GPU:** inspect output coverage and mismatches, independently calculate report summaries from emitted records, update capability ledger, decide pass/fail, create small follow-up jobs only for uncovered cases. GPU agent cannot self-certify full phase completion solely on its own reported summary.

### 5.4 Hardware and storage policy

At start of each run, record \`nvidia-smi\`, \`torch.cuda.get_device_properties\`, \`df -h /workspace /\`, CPU/RAM, driver/CUDA, process list, GPU free memory. WP11 previously targeted a 12 GiB RTX 5070; a separately reported RTX 5090 has 32 GiB; **actual host always wins**.

* **≤16 GiB available VRAM:** small models, short sequences, one site, CPU offload where supported, aggressively bounded transcoders/graphs; no promise of 4B+ gradient graph/training.
* **~24–32 GiB available:** moderately sized models / SAEs, controlled microbatches and 1-site interventions; memory-measure before large circuit computations; 32 GiB does not guarantee any specific CLT fits.
* **2+ real GPUs:** only here test F50 distributed acceptance. One GPU plus \`torchrun --nproc_per_node=1\` is a smoke, not multi-GPU verification.
* Keep checkpoint archives, intermediate activation caches and published reports separate; avoid writing model weights/caches on a tight root filesystem. Leave a configurable minimum free-disk reserve (recommended **20% of scratch volume** at job submission); this is an operational target, not a measured guarantee.
* No destructive \`rm -rf\` cache purge or instance shutdown without checking ownership and retention; use resumable checkpoints and garbage-collection manifests.
* No local/private data pushed to HF/GitHub merely to simplify agent handoff; credentials never appear in reports or logs.

## 6. Feature-level release gate: one row for each of the 50

Use the companion CSV as the **execution ledger**; it is populated \`TO_AUDIT\` rather than guessed to be implemented. Each row is independent even if implementation shares code with other rows. Required columns: ID, upstream, capability, owning phase, user-visible action, preferred reuse, compute class, native parity test, OpenSAE end-to-end acceptance, evidence link, live status, declared supported scope, blocking reason, reviewer.

The IDs below are stable; **these are not a list of promises already fulfilled**.

| ID | Capability | Phase | Verifiable acceptance action |
| --- | --- | --- | --- |
| F01 | Neuronpedia feature dashboard | P2 | Exact feature ID opens real token examples, logits where valid, provenance and explanations |
| F02 | Feature search | P2 | Search label / live inference / per-token TopK lead to matching genuine feature IDs |
| F03 | Automatic feature interpretation | P2 | Generator/model/method/splits/scores/negatives recorded; uncertain label shown as uncertain |
| F04 | Feature steering | P3 | Clean vs edited real outputs; strength, placement and protocol reproduce |
| F05 | Circuit exploration | P6 | Compatible graph can be inspected and a pinned node launched into real intervention |
| F06 | TransformerLens model loading | P1 | One current TL bridge model boots with frozen identity and reference logits |
| F07 | Activation caching | P1 | Selected hooks captured, checksummed, reused only with exact matching identity |
| F08 | Activation patching | P1 | Clean/corrupt and no-op patch follow verified target positions and metric |
| F09 | Hook-based interventions | P1 | Named hook edit changes intended activation and is removed after scoped run |
| F10 | Component analysis | P1 | A specified component contribution matches upstream analysis within defined tolerance |
| F11 | Pretrained SAE loading | P1 | Public checkpoint loads, metadata validated; encode/decode aligns with native |
| F12 | SAELens training | P5 | A real SAE trains, checkpoints, resumes and is reopenable |
| F13 | Multiple SAE architectures | P5 | At least two supported SAE families roundtrip with distinct metadata/metrics |
| F14 | Model/SAE integration | P1 | Hook/site dimensions, normalization and tensor layout are enforced |
| F15 | Feature analysis / visualization | P2 | Native feature codes drive correct top contexts and output analyses |
| F16 | NNsight internal-state access | P1 | Saved intermediate and gradient values on a supported local trace match native |
| F17 | NNsight activation editing | P1 | Intermediate tensor change affects downstream output, with no-op control |
| F18 | NNsight multi-input tracing | P1 | Two invocations remain correctly associated with original input tokens/IDs |
| F19 | Generation-time interventions | P3 | Edited decoding step and later state are real, logged and compared with oracle |
| F20 | Remote execution / NDIF | P7 | Authorized deployed model actually executes trace; status/failure/outputs audited |
| F21 | pyvene activation interventions | P3 | A native intervention reproduces through SAEBase typed contract |
| F22 | Composable interventions | P3 | Parallel or sequential configuration runs with order and site preserved |
| F23 | Fine-grained control | P3 | Neuron and token-position targets verified against actual edits |
| F24 | Cross-architecture interventions | P3 | Non-Transformer PyTorch model receives native and OpenSAE comparable edit |
| F25 | Serializable interventions | P3 | Exported intervention can be reloaded in a new process with same spec/output |
| F26 | Feature-level circuit discovery | P6 | Original compatible transcoder engine computes candidates for real input |
| F27 | Attribution graph construction | P6 | Native graph nodes/edges/targets/error nodes match preserved raw output |
| F28 | Interactive graph visualization | P6 | Pan/filter/click/inspect function on large and small real graph fixtures |
| F29 | Feature grouping/annotation | P6 | Pin/group/label are editable, persist after save/reopen, provenance separate |
| F30 | Circuit interventions | P6 | Pinned node edit actually re-runs a compatible model with measurement |
| F31 | SAE-Vis feature view | P2 | Real highest-activation contexts/values shown from source-aligned codes |
| F32 | SAE-Vis prompt view | P2 | Custom or recorded prompt displays correct token-level top features |
| F33 | Interactive token inspection | P2 | Token selection updates aligned feature activations and scores |
| F34 | Feature correlations | P2 | Independently recomputed correlations match displayed values and sample base |
| F35 | Reusable visual outputs | P2 | Standalone generated visualization opens and identifies source run offline |
| F36 | Standardized native SAEBench | P4 | At least selected official evals executed, not only imported JSON |
| F37 | Reconstruction/sparsity metrics | P4 | Native core vs SAEBase scores compared under identical definitions |
| F38 | Downstream behavioral evaluation | P4 | One real downstream task runs with original evaluator and preserved split |
| F39 | Custom SAE integration | P4 | New \`encode/decode\` SAE can run a native evaluation without rewrites |
| F40 | Benchmark comparison UI | P4 | Paired comparable axes/sparsity and source-linked plots; mismatches blocked |
| F41 | Gemma Scope 2 pretrained collection | P1 | Official asset manifests enumerated, downloaded lazily and checksum-bound |
| F42 | Multiple activation sites | P1 | Residual/attention/MLP or actually published hook types correctly mapped |
| F43 | Transcoder support | P6 | Input→output hook type, skip option and native reconstruction/forward validate |
| F44 | Cross-layer representations | P6 | At least one official crosscoder/CLT valid checkpoint loads with correct sites |
| F45 | Width/sparsity variants | P1 | Exact configurations kept distinct; user can select and compare valid pair |
| F46 | Sparsify TopK training | P5 | Real trainer produces TopK SAE and nonzero counts verify native behavior |
| F47 | On-the-fly activation extraction | P5 | Native streaming mode trains without presaved dense activation corpus |
| F48 | SAE/transcoder training | P5 | Two distinct native training modes produce usable, typed checkpoints |
| F49 | Custom hookpoints | P5 | Nondefault user-selected hookpoint trains; mismatch rejected |
| F50 | Distributed training/checkpoint reuse | P5 | Real 2+-GPU run, recovery and native equivalence; otherwise explicitly blocked |

### 6.1 A capability passes only after five checks

1. **Original behavior:** linked upstream doc and an unmodified upstream reference run or authorized evidence.
2. **Working OpenSAE entrypoint:** exact Python / CLI / browser route and concrete inputs.
3. **Numerical or behavioral evidence:** unit/CI + actual GPU for genuine model claims; controls and native parity with preregistered precision tolerance.
4. **Complete provenance:** pinned model, representation, input, intervention, metric, run, checksum and privacy rights.
5. **Independent rerun:** fresh process/environment, reader-readable errors, declared support/limitations and accessible artifact.

Use release flags \`not_available / recorded_only / local_ready / worker_ready / remote_ready\` for access, *separate* from engineering acceptance. This prevents a valid offline measurement from becoming a false “Run now” button.

## 7. Data, licensing, privacy and publishing

* **Sources:** Before reusing Neuronpedia explanations, exports or feature metadata, check API rules and attributed origin. Use the supported bulk export rather than scraping pages. Preserve public external permalinks and model/source identities; never treat imported labels as verified ground truth.
* **Models/checkpoints:** Never mirror gated or third-party pretrained weights into OpenSAE without rights. Prefer manifests and user-authorized local download. Publish reproducible model IDs/revisions/checksums, not access tokens.
* **Datasets:** Preserve source revisions, group identities and redistribution rights. The existing ten-source 500-group corpus has unresolved public raw-data eligibility; indexed findings can be publicly described only within already approved publication boundaries.
* **Human prompts:** New user input stays private by default. Public sharing is opt-in and records precisely which text, activations and outputs become public. Add deletion/redaction and retention rules for hosted workers.
* **API services:** Remote NDIF, AutoInterp providers and hosted compute are optional user-authenticated integrations; public Explore must work without them.
* **Code licenses:** Each upstream adapter has a SPDX/source/notice ledger. An MIT-like circuit-tracer implementation does not grant rights to unrelated site graphics, Hugging Face model weights or hosted data. If code copying is ever necessary, preserve original notices and pass license review.
* **Scientific release:** Never publish a derived figure as “paper-verified” without input, estimator, raw array/provenance, negative controls and source commit. Old paper results are immutable.

## 8. Common handoff protocol and collaboration rules

**Branching:** \`feat/wp12-p0-foundation\`, \`feat/wp12-p1-model-backends\`, etc., small PRs and independent review. Do not push unfinished application code directly to \`main\`. Creating this strategic work plan does not authorize downloading datasets, purchasing compute, rewriting published paper estimators or changing historical evidence.

**Each feature PR must include:** ID(s), upstream URL+commit, original behavior reproduced, exact changed paths, dependency lock, tests, one screen capture if UX, at least one evidence manifest when computation required, runtime/VRAM/disk figures, limitations and rollback. CI and source/rights checks before merge.

**GPU work order:** local CPU agent freezes inputs/schema → GPU agent preflight without heavy download → approved C1 → numerical review → C2 only for explicit gaps → immutable artifacts + manifest → local CPU independent validation → optional C3 approval.

**GPU agent outputs under a per-run folder (outside source-control for large/private records):**

\`\`\`text
runs/wp12/<phase>/<run_id>/
  manifest.json                 # model/SAE/hash/backend/dataset/splits/operation
  preflight.json                # real hardware, files, rights, estimated resources
  native_reference/             # original upstream unmodified outputs or IDs
  opensae_results/              # raw measurements, not just rounded averages
  parity.json                   # metric-by-metric match/mismatch/tolerance
  resource_usage.json           # wall time, peak VRAM/RAM/disk, effective throughput
  failures.json                 # OOM, unsupported, missing rights, retry outcomes
  checksums.sha256
  summary.md                    # concise interpretation; does not replace raw evidence
\`\`\`

Do not check large raw artifacts, model weights, secrets or private prompts into Git. Stable public summaries are published only through approved report paths.

**Merge rules:** P0 must pass before P1; failures stay visible; no “success” because the run was skipped; no performance claim without matched protocol; unreviewed scientific estimator changes are excluded from infrastructure PRs. Any blocking library/API changes receive a compatibility decision documented in the ledger.

## 9. Recommended first actual execution queue

Run these in order (subsequent agents may parallelize independent CPU documentation/UI work after contracts freeze):

1. **WP12-00 (CPU):** inventory 50 features and preserve existing Work Plans / PR statuses; inspect actual Research Lab; commit the ledger with honest statuses.
2. **WP12-01 (CPU):** freeze upstream APIs, package versions, representation identities, operation capability matrix, native-vs-imported definitions and minimal dependencies.
3. **WP12-02 (GPU-C1 / GPU-00):** execute the [fixed GPU-00 P0 parity protocol](12_gpu00_p0_parity_execution.md) using its [eight prompt pairs](12_gpu00_prompt_pairs.json), after C0 environment and tokenization preflight. Preserve exact native-vs-SAEBase evidence, status per model/site and failures. A legacy-only pass does not certify TransformerLens 4.
4. **WP12-03 (CPU):** build a feature detail + three search modes on existing authentic public measurements and exact identity links.
5. **WP12-04 (GPU-C1):** add real custom-prompt inference and feature steering for one supported model/SAE/site; preserve clean vs edited text and logits.
6. **WP12-05 (CPU):** integrate SAR/RSS/Localization evidence links, correct reporting, export and comparison baseline; share one complete report.
7. **WP12-06 (GPU-C1 then C2):** native SAEBench Core + Sparse Probing small case, verify native parity, compare two compatible SAEs.
8. **WP12-07 (GPU-C1):** one real short train → checkpoint resume → load → Explore; do Sparsify and SAELens separately.
9. **WP12-08 (CPU + GPU-C1):** load official graph fixture → implement viewer → compute one actual circuit → intervene; support official cross-layer assets where available.
10. **WP12-09 (remote + user testing):** authenticated worker smoke, verify constraints, and conduct fresh-user completion tests.
11. **WP12-10 (conditional C3):** only after evidence of scientific demand, expand large-site benchmarking/training/distributed resources.
12. **WP12-11 (independent acceptance):** sign F01–F50 within their declared support scope, publish release evidence, and archive actual blockers.

**Stop rules:** If C1 mismatches the original model/SAE, stop C2/C3 for that route and debug; if an official asset does not fit the GPU, use a smaller published compatible asset instead of pretending success; if dataset rights or credentials are missing, do not circumvent them; if multi-GPU is unavailable, F50 remains open.

## 10. Risks and explicit mitigations

| Risk | Mechanism | Mitigation / gate |
| --- | --- | --- |
| API churn | TL 4 removed Hooked classes; SAELens/NNsight change | Pinned isolated workers, bridge adapter, contract and parity tests |
| False unification | Different SAE/transcoder protocols hidden behind same UI | Typed representation, distinct operation names, unsupported-mode rejection |
| UI-only demos | Feature pages show precomputed results as fresh inference | Explicit recorded/local/remote execution labels and artifact IDs |
| Compute explosion | Coalitions, full 48-site sweep, large SAE training | C0/C1/C2/C3 gate, mandatory cache reuse, pilot-derived budget |
| Invalid scientific claims | Correlation or attribution taken as causal proof | Interventions, controls, uncertainty and original estimator references |
| Invalid benchmark ranking | Incomparable SAE hook/model/sparsity or metric variants | Comparable-key checks, multiple L0 points and metric-native definition |
| Distribution promises | Single GPU cannot test 2+ GPU | Keep F50 blocked until real multi-GPU job |
| Upstream licensing/data | Mirror third-party weights/prompts; bulk API scraping | Manifest/rights review, remote asset pointers and official exports |
| Maintenance load | Ten evolving dependencies | Adapter isolation, source-commit pinning, upstream smoke schedule |
| Resource exhaustion | Model duplicates / dense tensors / root disk | Disk/VRAM preflight, sparse storage, chunked jobs, retention policy |
| “Best” unsupported | Claimed superiority without independent task study | Matched multi-task benchmark, outside users, published limitations |

## 11. Definitions of final strategic success

**Functional:** Each of F01–F50 has a clearly scoped executable and verified path; no feature gets credit simply because another application has it.

**Scientific:** Source-bound original-estimator comparison, negative controls, separate discovery/evaluation, exact representation semantics, immutable results and honest unsupported states.

**Product:** From a newly supplied real prompt to a shared reproducible result is an understandable short path; no mandatory GPU for browsing; interactive work remains accessible to specialists through Python.

**Engineering:** Clean install, pinned workers, cross-version maintenance, import/export stability, healthy CI, recovery and publish/rollback.

**Community:** Researchers cite/reuse actual public datasets, reports and APIs, contribute external examples and can complete tasks without the core author's hands-on help. Stars and visits are secondary.

**Important boundary:** The long-term ambition is full 50/50 delivery. The release claim “fully supports all 50” is allowed **only** when the complete evidence ledger actually passes with declared scope and conditions. Until then publish the current verified subset, unresolved external restrictions and a visible roadmap.

## 12. Upstream and internal references

**Internal:** [Repo README](../README.md) · [Work Plan archive](README.md) · [WP04](04_research_platform_execution_plan.md) · [WP07](07_research_lab_final_product_redesign.md) · [WP11](11_high_throughput_engine_and_final_acceptance.md) · [v0.4 release notes](../docs/releases/v0.4.0.md) · [Research Lab acceptance](../docs/research-lab/release-acceptance.md).

**Upstream documentation:** [Neuronpedia](https://docs.neuronpedia.org/) · [TransformerLens 4.0](https://transformerlensorg.github.io/TransformerLens/content/news/release-4.0.html) · [SAELens](https://github.com/decoderesearch/SAELens) · [NNsight](https://nnsight.net/documentation/) · [pyvene](https://github.com/stanfordnlp/pyvene) · [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) · [SAE-Vis](https://github.com/callummcdougall/sae_vis) · [SAEBench](https://github.com/adamkarvonen/SAEBench) · [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-270m-pt) · [Sparsify](https://github.com/EleutherAI/sparsify).
