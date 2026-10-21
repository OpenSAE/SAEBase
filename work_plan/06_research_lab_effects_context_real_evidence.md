# Work Plan 06 — Fill Feature Effects and Context Lab with Existing Real Evidence

Date: 2026-10-20
Status: Execution plan for the local agent. This document is not a claim that the private data have already been imported or that the deployed UI has been checked.
Owner: OpenSAE / SAEBase
Depends on: Work Plan 05, current local Research Lab UI, and the existing GPU experiment archive.

## 0. Objective and release rule

Make the second and third Research Lab workspaces genuinely useful:

1. Feature Effects: a visitor opens the module and can immediately explore measured feature interventions, with working site, feature, prompt-case and operation selectors.
2. Context Lab: a visitor can manipulate all 64 authored inputs and observe actual SAE responses, plus browse existing RSS results by model, dataset and transformation.
3. Every visible result corresponds to a preserved observation and an exact model/SAE/site/protocol identity. Never invent token probabilities, generation text, feature labels, activations, curves, or scientific conclusions.
4. Reuse already recorded measurements. The default GPU budget is ZERO. Only narrowly scoped short follow-up measurements are permitted after a documented inventory proves that an important UI capability cannot be produced from existing records.
5. The first module, Feature Atlas, is finished from the product owner's perspective and MUST NOT be rebuilt, restyled or replaced. The same applies to its existing curated examples. Reuse its site/feature selection only through safe, explicit links.
6. Do not ship a public page with generic "unavailable", "no data", "GPU disconnected" or empty charts in the second and third modules. Instead, only expose combinations that have measured backing data, choose a measured default, and fail the release acceptance gate if either core module remains empty. A blank value is never filled by inference.

This is **real evidence browser** work, NOT an online inference product. Do not open unauthenticated GPU execution or change the old POST /api/jobs safety policy.

## 1. First action: inspect the ACTUAL local/deployed UI rather than replace it with old GitHub frontend

The GitHub main snapshot of apps/research-lab/frontend may be older than the deployed interactive v07 UI. The local agent must first:

- In the local checkout, record pwd, git status, current branch and commit.
- Locate actual source for the user's currently deployed Feature Atlas, Feature Effects, Context Lab and remaining workspaces. Search local working tree, additional frontend packages, deployment assets, and the actual Hugging Face Space checkout.
- Inspect the exact v07 second/third module route names, selectors, components, current JSON schema, data loading paths and build pipeline. Do not assume the old GitHub main sections named Intervene/Analyze are equivalent to the new workspace routes.
- Record the current initial selection in the first module and screenshot before/after. Do not touch its internal content except for compatible navigation.
- Inspect the previous interactive-upgrade agent handoff if it exists locally. Reconcile with local changes; preserve edits already made since that upgrade.
- Create a local source inventory: UI module -> loader/API -> data asset -> observed emptiness reason. No modifications to the public site until the inventory is complete.
- Treat noncommitted work as valuable. Do not git reset, force checkout, overwrite files, or rebase away local changes.

The GitHub Work Plan 05 source and experiment record are reference material; the local agent has the authoritative latest deployment working tree.

## 2. The local real measurement sources already recorded

Use this order of precedence: raw private measurement -> verified derived data -> GPU executor decision/summary -> public historical acceptance. Do not turn the final summary table into per-prompt observations.

Known private run recorded by Work Plan 05:

    experiments/research-lab/7fb2be8f29a1-20261019T035727Z/

The original host previously used:

    /workspace/SAEBase/experiments/research-lab/7fb2be8f29a1-20261019T035727Z/

The path is a locator from the prior execution report, NOT a guarantee it is mounted on the current machine. If absent, search the current local archive and backups by run manifest, decision.json and SHA-256 identities before proposing any recompute.

### 2.1 Feature Effects, highest priority

- GPU-04/decision.json and GPU-04/interventions/pairs.json: SIX deep-dive model/SAE sites. The executor recorded one clean-versus-edited evaluation prompt per site, including ablate/scale/identity/reconstruction and supplementary clamp/add/coalition where defined.
- Pairs include site_id, prompt_group_id, real checkpoint hashes, feature IDs and saved output logit L2 values; inspect individual pairs and embedded feature_checks/extra_operators rather than infer a universal list of operations.
- The earlier GPU-01 pilot and any additional local paired-intervention arrays or JSON must also be searched. They may contain further actual per-token results. A logit L2 scalar is NOT a token-probability distribution.
- GPU-03/feature_cards.json and private per-site ..._codes.jsonl contain actual feature IDs and activation statistics. Join only on exact site_id and prompt/sample/group identity. They support feature activation context; they do not by themselves prove downstream causal effects.
- Existing historical Stage D reports can supplement browsing in a clearly separate protocol; do not mix them into the new 500-prompt cohort or substitute one model's site for another.

The six pinned deep sites from apps/research-lab/data/coverage.json are:

    qwen3_1_7b_base:qwen3_1_7b_base_resid_w32k:L14
    gemma1b:gemma1b_scope_resid:L13
    gemma3_4b_pt:gemma3_4b_pt_scope_resid:L22
    gemma3_4b_it:gemma3_4b_it_scope_resid:L22
    llama31_8b:llama31_8b_lxr_8x:L21
    mistral7b_instruct:mistral7b_instruct_tc_l16_131k:L16

### 2.2 Context Lab: real six-factor / 64-state experiment

- GPU-07/real_model_six_factor_design.json contains 64 project-authored prompts, state masks 0..63, factor names and two levels per factor.
- GPU-07/localize_status.json contains summarized results, but the interactive 64xfeature matrix must be built from actual saved codes rather than recreated from the summary.
- private/pythia410m_pythia410m_mlp_topk_L12_codes.jsonl contains rows for the 64-state controlled design. Select rows with dataset_id == controlled6. Match by the exact sample_id in the design, preserve the original SAE dictionary feature IDs, and expand the sparse nonzero_codes into observed-support columns.
- The prior run reports 55 feature columns observed as nonzero somewhere on this finite grid, 64 complete states, and a reported 90% variance budget of 132. Recompute simple UI statistics from the genuine 64x55 matrix on CPU. Recalculate/check the saved variance and localization summary without relabeling 55 observed columns as the full SAE dictionary.
- This is ONE controlled input population with 64 states, not 64 statistically independent prompts. Feature behavior on this specific grid does not establish universal feature meaning.

### 2.3 Context Lab: RSS measured variants

- GPU-06/rss/<site-id-with-colons-replaced-by-underscores>.jsonl stores measured rows by sample_id for SIX sites and 500 parent groups/site. Each row holds settings for local_format, instruction_wrapper and context_wrapper, with support feature IDs, mean_rss, held_out_rate_mean, status and semantic_review.
- The recorded full run covers 3,000 site x parent groups, 45,000 selection variants, 90,000 held-out variants and 3,000 shared clean observations (138,000 logical site-input occurrences). Do not present this count as independent prompt groups, independent model forwards or 138,000 feature cards.
- Older 99,200 nominal calls with unverified tokenizer positions are a separate legacy population. They must not be pooled into the correctly aligned RSS data.
- The summaries above may NOT contain full per-variant feature activations or user-readable prompt text. If full delta vectors are absent, build the UX around real MeanRSS, support membership/count and held-out directional rate. Do not draw fabricated per-token or feature-wise delta curves.
- semantic_review remains false. These are PROPOSED transformations and are not all certified to preserve meaning. Source text may remain private.

## 3. Work unit A: evidence census, no GPU

Deliver a machine-readable inventory to:

    outputs/research_lab_fill_06/inventory.json
    outputs/research_lab_fill_06/inventory.md

Scan under the real run root for exact source files, sizes, statuses, row counts, sample/group identifiers, sites, operation types, number of real paired logits (if any), possible public text permissions, loaded checkpoint hash identities and missing arrays. Include file sha256 and source relative path. Do not emit private prompt text in inventory logs.

Mandatory census questions:

1. How many valid real Feature Effects records by site, prompt, feature and intervention operation? Which carry only logit_l2, and which carry actual baseline/edited vectors or top tokens?
2. How many of the 64 controlled states have actual matching sparse SAE codes, and how many unique support feature IDs? Are masks/order/IDs exactly bijective?
3. How many measured RSS parent groups per site x setting x source dataset? Which rows have defined supports and held-out rates? Are selection and held-out correctly distinguished?
4. Which feature descriptions or excerpts have explicit rights and content review? Do not label unknown text as approved.
5. Which local v07 controls render an empty state because of filtering, incorrect join keys, an asset path mismatch, incomplete manifest, API contract mismatch or data genuinely missing?

First fix loader paths/joins. Missing webpage examples are NOT evidence that the expensive GPU run was absent.

## 4. Work unit B: canonical small publish bundle, no GPU

Implement an offline CLI, proposed location:

    scripts/research_lab/export_real_cases.py

Input: --run-root <private run dir> --site-registry <registry> --out <temporary output>.
Provide --inspect, --build and --verify modes. The script must read the existing JSON/JSONL, stream large files, not require torch/model weights and not download models. It must not recursively publish private source directories.

Use VERSIONED schemas and compact browser assets:

    apps/research-lab/data/interactive_v1/manifest.json
    apps/research-lab/data/interactive_v1/effects.json
    apps/research-lab/data/interactive_v1/context64.json
    apps/research-lab/data/interactive_v1/rss.json

If the current v07 frontend already has an accepted bundle location, adapt paths instead of duplicating stores. Generate static JSON for the HF static Space, or expose equivalently through a read-only API when the live architecture requires that. No new cloud backend or GPU endpoint is required.

Common per-case provenance:

    release_schema
    source_type = real_checkpoint
    protocol_id
    run_id
    source_relative_path
    source_sha256
    site_id
    model_sha256
    sae_sha256
    sample_id and/or prompt_group_id
    observation_position
    measured_fields
    public_text_reviewed
    rights_reviewed

The selected experiment must match site, checkpoint identity, source population and readout protocol; otherwise fail/skip with a private audit reason. Never silently replace a mismatch with a similar checkpoint.

Every published asset receives its own SHA-256 in manifest.json, with observation counts and a concise source population description. Publishing is an explicit build action after validation. Preserve the original experiment directory untouched.

### 4.1 effects.json contract

Each record has unique case_id, site_id, feature_ids, operation, operation_parameters, prompt_group_id, source_protocol, baseline/edited measurement identity, measured logit_l2 where available, optionally real top-token logits/probabilities where saved, and explanatory notes derived only from actual results. Separate:

- measured logit L2 (valid now)
- real paired next-token logits/top-k (display only if stored or truly remeasured)
- real complete text generation (display only if an actual rollout exists)
- validated feature activations (not a substitute for model output differences)

If one pair has several operators, create selectable operations under ONE unique prompt case, not several falsely independent experiments. Store values as measured, not interpolated across strengths or reused between unrelated features.

### 4.2 context64.json contract

One complete design with six named factors, 64 bitmask-indexed authored prompts, site_id, dictionary feature_ids and precisely 64 activation rows aligned to those IDs. No arbitrary synthetic examples. Verify all 64 masks exactly once, no missing record, no duplicate sample ID and all finite values.

Compute on CPU from the actual matrix, per feature:

- 64 observed activation values and observed nonzero count
- overall mean/max and ranking for choosing readable examples
- six marginal effects, for each factor the mean of the 32 paired-on-minus-off differences at fixed remaining factors
- optional pairwise factorial contrasts only if exactly computed from the complete grid
- 64-state rank/heatmap and user-selected two-state activation difference

The factorial effects are descriptive differences within this one controlled input design, NOT evidence of causal invariance or independently replicated significance. Keep feature IDs and source site visible in research details.

### 4.3 rss.json contract

For each measured site x dataset x transformation, publish aggregated source-backed rows with parent group count, observed MeanRSS distribution summary, mean held_out_rate_mean over defined cases, support count distribution, number of undefined supports and source protocol. Optionally expose a paged, rights-safe per-prompt-case view by opaque sample_id and measured support IDs/rates.

Selection and held-out roles must remain separate. Compute averages only on defined values, record exact denominators, and show uncertainty only when actually calculated using valid independent parent groups. Keep semantic_review false, source-license labels and the published wording "proposed input transformations" intact.

Never put private dataset prompt strings or sampled third-party variants into the static bundle without explicit approval. Project-authored controlled64 prompts are distinct and may be presented as authored demo inputs.

## 5. Work unit C: redesign Feature Effects around evidence that ALREADY exists

Objective: no empty landing screen, actual effects on first load, immediate human-readable explanation.

Required interface:

1. Default to a selected, checksum-valid measured intervention. Load it immediately; do NOT default to arbitrary site/feature combinations without measured pairs.
2. A compact gallery of selectable measured CASES: model/SAE, layer, operation, source task category (if known), and result magnitude. Feature ID is always paired with its site, not globally interchangeable.
3. Case detail: feature IDs, chosen experiment condition, clean vs edited measured logit L2, comparison with other OPERATORS on the same saved prompt when available, and what the score DOES and DOES NOT mean.
4. A within-site operation chart displays only actually measured operators for the same prompt. Do not compare raw L2 magnitudes across sites as though they were an SAE quality leaderboard.
5. If actual paired per-token logits exist, render top-token identities, logits, normalized probabilities and change for exactly the same readout position/vocabulary; any probability must be calculated from real complete logits or explicitly valid stored softmax values. If only an L2 scalar exists, show the measured intervention result and an interpretable within-case operator comparison, NOT a fake before/after probability graph.
6. "Try another example" cycles to a different real measured case. A case is shareable by a stable URL and has "Open source evidence" / protocol details.
7. Prefer a small, accurate one-sentence explanation to a long list of technical terms. No guessed labels or claims of full text steering.
8. Support click-through from a Feature Atlas card ONLY if that exact site x feature x intervention has backing evidence. Otherwise do not expose an invalid action or redirect to the nearest valid feature without saying so.
9. Keep an advanced foldout for all available protocol, hashes and detailed research metadata. Do not make that the main display.

The saved six-site clean/edited logit evidence is enough to deliver a working Effects browsing experience even WITHOUT a new GPU job. Capture all existing operations per site from the original records and make the first real case automatic.

## 6. Work unit D: Context Lab as two functioning experimental modes

### 6.1 Controlled Inputs — default

This MUST be populated from GPU-07 real_model_six_factor_design, not a sample synthetic truth table.

- Start with a real feature from the 55 observed-support columns and a selected state with a real activation.
- Six clickable factor switches with descriptive, human-readable labels and their actual two choices (setting, season, event, venue, voice, format).
- Update the authored prompt and highlight exactly the input factors that changed. Preserve mask 0..63 bit ordering and choose a consistent default.
- Display 8x8 state heatmap with all 64 true activations for the selected SAE feature. Clicking a square changes all six switches and source prompt.
- Show a visible before/after activation readout for the selected toggle. Avoid extrapolation to unobserved states; all 64 have actual recordings.
- Add "Most responsive features" selector ranked by genuine finite-grid variation or absolute factor effect. Feature IDs are real SAE dictionary indices, NOT indices 0..54 of the support matrix.
- Allow choosing another factor and inspecting its 32 paired differences in a small chart, plus a concise explanation.
- Offer "Surprise me" random selection only among true measured site/feature/state combinations; no synthetic values.
- Show localized measurement scope in the detail foldout: one 64-state controlled population; this is not the 500-natural-prompt corpus.

### 6.2 Input Transformations — RSS

Make this a second tab within Context Lab, using GPU-06 recorded JSONL rather than an external placeholder page.

- Default to a real site/dataset/setting combination with nonempty valid measured rows.
- Model/site and dataset dropdowns only include combinations with data. Transformation tabs are local_format, instruction_wrapper and context_wrapper, displayed with short user-friendly explanations.
- Show genuinely measured MeanRSS and held-out direction consistency; denominator and statuses belong in the details.
- Plot real setting-by-site or setting-by-dataset comparative charts aggregated from the saved rows.
- Offer selecting opaque recorded prompt cases and inspecting support feature IDs and held-out summaries. If original prompt text is unreviewed, DO NOT print it; the measured case can still be meaningfully browsed using source category and measurement plots.
- Do not treat "semantic_review=false" as meaning preservation; show "proposed transformation" in plain language.
- If original delta arrays exist in the private archive, add feature delta charts after validating alignment and permission. Otherwise omit this optional detailed chart; the default RSS experience must still be complete through genuine summary/distribution evidence.

The third module must open into the complete 64-state experiment, and the RSS tab must have a populated measured default. Users should not have to manually discover a site with nonempty data.

## 7. Optional fast gap-fill; only after genuine extraction and UI work

DEFAULT ACTION: No GPU. No 48-site or 500-prompt recapture. No reproduction of SAR/RSS/Localize. No new training or downloading all model weights. Existing numerical results should remain unchanged.

Allow a QUICK, BOunded follow-up only if all of these are true:

- The user-visible Effects experience is still materially unclear because the original archive contains ONLY logit L2 and no actual top-token results.
- The complete local archive was checked, including pilots, arrays and offline caches.
- The additional result cannot be derived honestly from the saved observations.
- The existing local checkpoint/cache is ready; no costly remote compute purchase, multi-model reprocessing or major setup.
- Explicit budget: at most 12 prompt-site pairs across at most TWO already usable model/SAE sites, using a handful of project-authored short prompts. Prefer a small verified model and reuse weights/session. Never rerun 500 prompts or all 48 sites.
- Work is interruptible, bounded, independently verified and logged. Abort the optional step if model loading, memory or throughput is unexpectedly expensive; keep the default case-based UI fully working without it.

For such a tiny remeasurement, save true baseline/edited top-token logits and token IDs at exactly the same readout position, prompt tokens, SAE hook and intervention definition; verify identity patch and loaded checkpoint hashes. This is a NEW micro-study with its own protocol ID, run ID and public authorship-reviewed prompts. Do not silently append it to the previous GPU-04 population.

If this optional step is skipped, the user still gets real six-site Effects comparisons and the full Context experiment. Do not display the phrase "unavailable" in place of measured existing results.

## 8. Product, code and release boundaries

- Preserve Feature Atlas styling, content, navigation and selection behavior. Avoid globally scoped CSS or JS state collisions.
- UI titles should be short: "Feature Effects", "Context Lab", "Controlled Inputs", "Input Transformations", "What changed?", "Compare edits", "Feature activation". Keep "SAR", "RSS" and exact scientific terms in research details when useful.
- Human-readable charts first; protocols and checksums in an expandable "Evidence" section.
- Support narrow/mobile layouts, keyboard operation, accessible labels and screen-reader announcements on case/state changes.
- Cache compact static JSON once; do not fetch individual private records per interaction. All factor toggles and case tabs should be instant CPU-side interactions.
- The UI MUST not treat case metadata, class labels or prompt descriptions as verified feature explanations. Real feature annotations from the first module are only reused when separately reviewed and identity-matched.
- Do not overwrite existing Space assets blindly. Integrate against current local v07 source. The static HF deployment may require a separate asset generation or upload step; include that in the agent's release script.
- If adding tests or scripts, put cleanly named files under scripts/research_lab/ or apps/research-lab/tests/ as appropriate. Do not add private samples or ephemeral cache files to Git.
- Update the Lab's README with concise public usage text, NOT internal Work Plan nomenclature, defensive explanations, or internal stage names. Preserve scholarly accuracy in detailed documents.

## 9. Mandatory executable acceptance tests

The agent must implement and run these, not just write a checklist:

A. Extraction and manifest
- Detect and parse real GPU-04 pairs, GPU-06 RSS JSONL, GPU-07 64-state codes and the authored design, using original SHA-256 references.
- Export is deterministic: same inputs produce byte-identical JSON and digests.
- Rejected rows remain in a private audit, never converted to a passing observation.
- No private original-source prompt/variant text leaks to public bundle.
- All displayed site_id/SAE/layer, feature IDs and recorded protocol join exactly.

B. Feature Effects
- At least one genuine measurable case for EVERY successfully archived GPU-04 deep site (target six); show actual available operators, not assumptions.
- The default screen shows a nonempty, source-backed readout without user configuration.
- Switching measured cases/operators changes the values and provenance as expected.
- L2-only cases do not render nonexistent token probabilities or generation.
- Feature Atlas -> Effects deep-link only resolves valid matching experiment pairs.

C. Context Lab 64-state
- Exactly 64 masks and 64 corresponding real SAE code observations.
- Exactly the original distinct observed-support feature IDs (expected 55 given prior record, verify against archive); all 64xN values finite.
- Switching any factor changes the correct bit and actual authored input; 8x8 selection is bijective.
- Toggle-pair marginal effects agree with independent CPU reference calculation.
- At least 12 representative real measured features are easy to discover, while all recorded 55 remain selectable. No artificial labels.

D. RSS
- A measurable RSS default loads without clicking a "find available data" button.
- Audit six measured sites, three setting types and recorded parent group populations; report exact actual valid counts, not a hard-coded assumption of every combination's completeness.
- Aggregated means and held-out rates reproduce CPU recalculation from selected source rows within numerical tolerance.
- Empty/undefined supports are handled by accurate denominators, not displayed as zero.
- Legacy position-unverified records never mix into position-corrected values.

E. Browsers and deployment
- Browser smoke test on desktop and mobile and with keyboard for BOTH lab modules.
- Each shown figure has a visible specific experiment behind it and a link to provenance.
- Network console: no 404 data assets, uncaught JS errors, failed JSON parsing or inaccessible default case.
- No "unavailable" or fake placeholder figures on the initial screen or on any VISIBLE selectable case.
- Existing Feature Atlas screenshots/behavior remain unchanged.
- Verify the ACTUAL deployed Hugging Face Space URL, not only a local dev server.
- If data inventory fails, do not deploy an empty shell; preserve currently working pages and report the specific missing files.

## 10. Deliverables and reporting format

Commit code, public data bundles, tests and documentation on the current project branch, then push to the normal remote using the existing authorized agent workflow. If an agent cannot push because of authorization, stop short of claiming a successful deployment; leave a complete local commit and handoff.

Expected change set:

1. scripts/research_lab/export_real_cases.py (or equivalently named importer)
2. apps/research-lab/data/interactive_v1/ compact reviewed public assets + manifest (or the same current-deployment data target)
3. Version-compatible UI module integration for Effects and Context, preserving Atlas
4. Tests for the provenance loader, exact joins, factorial math and UI selectors
5. Clear public-facing Lab README, plus private audit/asset inventory outside public deployment
6. Local screenshots for both workspaces, deploy validation results, final git commit hash and live URL

Write a final execution report:

    outputs/research_lab_fill_06/final_report.md

Report actual data-backed case counts by site/operation, Context feature/state coverage, RSS by site/dataset/setting, what was imported from preexisting files, whether OPTIONAL tiny GPU work happened (default no), what was withheld for rights review, tests passed, published URLs and new commit(s).

A workplan is not complete until the second and third live workspaces both open on a populated real measurement, with working controls and validated provenance. If the first pass finds a loader path problem, FIX the path and continue to this end state rather than ending with a diagnosis.

## 11. Agent start sequence

1. Compare local/deployed v07 code and archived Work Plan 05 state; snapshot existing Atlas.
2. Search and inventory private real measurements before adding any compute work.
3. Export small validated data bundles from GPU-04 / GPU-06 / GPU-07.
4. Populate and wire Feature Effects and Context Lab; preselect real cases and hide unsupported combinations.
5. Run unit, data integrity, browser and layout tests.
6. Only if required, consider the tiny bounded GPU follow-up in Section 7.
7. Commit and deploy; verify BOTH live modules and produce final report.

Core rule: existing measured cases first, truthful interactivity second, GPU work last.
