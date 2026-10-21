# Work Plan 07 — Research Lab Final Product Redesign and Visual Unification

Created: 2026-10-20
Status: **ACTION REQUIRED — execute through production acceptance, not another proposal.**
Previous context: Work Plans 05 and 06. This plan supersedes their public UI design instructions where inconsistent. Preserve their scientific evidence and source restrictions.
Scope: Feature Effects redesign, Feature Atlas visual compatibility, consistent product shell for ALL Lab modules, Context Lab and remaining-module regression tests, final production deployment.
Project: OpenSAE SAEBase, https://github.com/OpenSAE/SAEBase
Live entrypoint to validate: https://opensae-saebase.static.hf.space/index.html (use fresh deployed assets; do not assume v=07 is current).

## 0. Decision and nonnegotiable outcome

**Stop building a research ledger with buttons. Build an understandable interactive explanation using real measured experiments.**

The prior Effects screen used "Two plus two equals" and reported P(next token "four") = 44.9% before and 43.8% after a feature removal. The top prediction was unchanged, and there was no evidence that this particular feature represents arithmetic. Showing it as the default "discovery" is misleading as a product: it conveys neither a compelling effect nor a reliable semantic explanation. The labels "What we changeRemove this featureCut its activity in halfAdd 1 to its activity" also show a broken control layout. Earlier L2-only charts with sample IDs and hash strings were even less useful.

This is a product failure, not a missing footnote. Do not solve it by adding more explanatory paragraphs, metrics, badges or operator buttons.

**Target user test:** Someone without ML knowledge can tell, at a glance, (a) what input the model saw, (b) what we changed, (c) what specifically changed in the model's response or its predictions, and (d) what the evidence does not establish.

The entire Research Lab must look and behave like one product. The existing Feature Atlas is the approved visual reference. Keep its existing scientific content and main interaction. Adjust shared chrome and the other Labs to match Atlas; do not replace Atlas with a different design.

No simulated output, fabricated natural-language feature labels, inferred model answers, invented source text or unverifiable percentages. Existing source provenance stays available, but is hidden by default behind one small details disclosure.

**Execution ends only after a visual review of the actual deployed application on desktop and mobile, with working curated real cases.** Do not finish with "plan delivered", "code deployed" or unit tests alone.

## 1. First 20 minutes of work: locate and protect the ACTUAL deployment source

This instruction is about the agent's procedure, not a promised duration.

1. Record local working tree, branch, HEAD SHA, uncommitted changes, current Space source revision, asset pipeline and URL. Preserve all current work.
2. The GitHub `apps/research-lab/frontend` may be the old HTML application, whereas the live HF Space has newer Feature Atlas / Real interventions / Input Lab interfaces. Do NOT overwrite the live frontend with the old GitHub page or the previous ZIP.
3. Locate current `Feature Atlas`, `Feature Effects`, `Context Lab`, model comparison and research components and their CSS/JS; identify whether they live in the HF Space repository, elsewhere in the local checkout, or under generated static assets.
4. Save baseline screenshots at desktop 1440x900, tablet 768x1024 and mobile 390x844. Include screenshot of the Atlas and Effects current state. Record full DOM/layout overlap issues and text wrapping.
5. Make a map of components, routes/hash states, global stylesheet imports, shared containers, data loaders, feature navigation and build/deployment. Preserve routing behavior and deep links.
6. Inventory saved case artifacts and their real baseline/edited readout fields before changing the data model. Work Plan 06 contains source paths.

Output this brief inventory locally at `outputs/research_lab_final_07/inventory.md`. No private prompts or credentials in public logs.

## 2. Product concept: one strong experiment, one obvious interaction

New Effects first view MUST have just four visible areas:

### A. One compelling, public-safe example

Heading: `What changes when we remove a model feature?`
Subheading: `Pick a real example, then compare the model's prediction before and after one internal signal is removed.`

Start with the **exact human-readable prompt** from a rights-approved source or short project-authored prompt that has actually been measured. Do not show an opaque `arc_challenge:TIMSS...` ID or "prompt not in public record" as the first impression. If a source is not approved for public text, that case can appear only in advanced anonymized research browsing, not the introductory carousel.

Display small familiar case names (e.g. `A factual question`, `Complete the sentence`, `Reading comprehension`, `Code completion`) derived from actual task, not invented feature meaning. `Gemma 1B` is model identity; "arithmetic" may describe the question but must NOT be called the feature's semantic label. Show a compact `Try another example` control and 3–6 *actually measured* curated examples initially. Keep the full measurement browser one click deeper; no need to expose all raw recordings at the top.

### B. One primary user action

Use ONE obvious action: `Remove this feature`. The action changes the right comparison from the baseline to the **measured** intervention state. A two-state segmented control `Original / Feature removed` can be used to compare, but baseline is not a scientific edit. Preferred: keep a persistent baseline on the left and reveal the edited result on the right by clicking `Remove this feature`; allow `Restore original` / `Replay`.

Do NOT make novices choose among `ablate`, `scale`, `clamp`, `add`, `coalition` or arbitrary strength fields before seeing a result. Put actual other measured edit methods in an optional `Explore other changes` subsection below the primary takeaway. When opened, give human descriptions and exact measured parameter values; never interpolate a continuous slider across unmeasured strengths.

Changing the selected case resets the interaction consistently, updates BOTH sides, and preserves accessible keyboard operation.

### C. Clear observable result

Prioritize actual complete continuation text if independently recorded with identical comparison protocol. Otherwise use actual next-token predictions and probability percentages calculated from correctly aligned complete-vocabulary logits. A next-token prediction is not necessarily a whole English word; decode and display tokenizer fragments honestly, including leading whitespace. Do not claim a sequence-level answer changed from a single next-token prediction. If labels are complete multi-token answers, score full continuations with the original experimental protocol rather than first-token proxies.

Show readable, aligned before/after results with **the same candidates and consistent scale**:

```text
A REAL INPUT                         [Another example]
<exact, approved prompt>

[Remove this feature]                [Restore original]

          ORIGINAL                      FEATURE REMOVED
Top prediction: "..." (..%)         Top prediction: "..." (..%)
<shared candidate> [bar] ..%        <same candidate> [bar] ..%
<shared candidate> [bar] ..%        <same candidate> [bar] ..%

WHAT CHANGED?
<one short sentence based on the REAL measured result>
<one concise sentence about whether this actually changed the top prediction>
[Explore other changes ▾]   [Experiment details ▾]
```

Emphasize the largest **percentage-point** changes, not raw logit L2. Use the union of candidates from both distributions rather than two differently ordered top-5 lists that cannot be visually compared. Never normalize only the top-5 probabilities to 100%. If the top candidate changes, point it out clearly. If it does not, say `The model still favored the same continuation; its confidence changed by X percentage points.`

Do NOT use sample counts, model hashes, feature IDs, raw L2, tokenizer IDs or protocol strings as hero KPIs. Avoid "prediction means better answer". A gold label supports task correctness only if a matched fully evaluated task readout was actually measured.

### D. One meaningful takeaway, not a grand claim

Text is determined entirely by recorded facts. Example wording:
- `Removing this feature changed the most likely next piece of text from "A" to "B" for this input.` Only when truly supported by aligned logits.
- `The top prediction stayed the same, but its probability dropped by X percentage points.` For measured smaller effects.
- `For this particular prompt, the change was small. It does not establish what the feature represents.` For near-null examples.

Never label a feature as `arithmetic`, `truthfulness`, `language`, or another generalizable concept based on a single convenient prompt. A semantics claim requires separate multi-prompt contrastive evidence. The Atlas may have interpretation hypotheses; retain their uncertainty when linked.

**Remove this public sentence:** `A feature is a learned internal signal...` from the main title area if it crowds the example. If needed use a one-line tooltip `A feature is a pattern detected inside the model.`

## 3. Curate examples from saved REAL evidence, rather than choose the first row

Default example selection must be deliberate and justified.

### 3.1 Search local evidence, zero GPU initially

Inventory sources from Work Plan 06:

- `GPU-04/interventions/pairs.json` and `GPU-04/decision.json` — real six-site baseline/edited measurements, often only logit-L2 aggregate;
- `GPU-01` pilot and any additional `logits`, `top_tokens`, `readouts`, `effects`, `generation`, `.pt`, `.npz`, `.npy`, `.jsonl` archived paired output arrays;
- `GPU-03/feature_cards.json` and private `*_codes.jsonl` — observed activation evidence (NOT a replacement for output edits);
- vetted public Space/demo records, HF dataset `OpenSAE/SAEBase-Evaluation` and the existing ten Atlas prompts — use exact (model, site, feature, input) joins, not same-looking examples from another model;
- original small benchmark/cached examples and any earlier data exported for the interactive upgrade.

Use raw arrays when possible. An L2 scalar cannot reveal token rankings, probabilities, which word changed, or task quality. Do NOT invent a readable text from a hash, retokenize without verifying, infer semantics from task label or splice two unrelated experiments. Keep an audit mapping each public case to exact source and hash.

### 3.2 Curation rules

Rank eligible measured cases for **visitor clarity**, not for proving research claims:

1. Approved, visible prompt or authored public-safe input.
2. Identical model/SAE/layer/feature/input/readout positions in baseline and edit.
3. Concrete, genuinely noticeable observable output difference. Prefer cases where the top prediction changes or where a well-identified candidate shifts substantially; a five-percentage-point change may be a pragmatic *presentation* filter, NOT scientific significance.
4. Cases that teach distinct possibilities: changed top prediction; substantial redistribution but unchanged top prediction; small/no observable change. The last is an optional contrast case, NOT the landing case.
5. Multiple task types and at least two genuinely measured model/SAE sites if data permit. Repeated snapshots of the same prompt are NOT distinct examples.
6. Each case's title must correspond to the prompt's task, not to an unsupported semantic interpretation of the feature.

At least three engaging, measured, public-safe cases should be reachable by simple `Another example`; target six. If local records do not support three, follow the bounded quick case-fill below rather than dumping technical rows on the public page.

### 3.3 Last-resort very small new measurement

Default = **NO rerun**. Only after archive inventory proves token-level outputs missing, a microstudy can be justified. This is a new curated *demonstration set*, not a replication of prior large runs.

- No 48-site recapture, no 500-prompt rerun, no retraining, no arbitrary live user prompts.
- At most TWO already locally available real checkpoint/SAE sites, at most TWELVE prompt-site pairs, preferably 3–6 short project-authored prompts per site.
- Reuse validated feature choices and exact residual-preserving ablation operator. Make clean baseline and edited full-logit output for identical prompt tokens; verify identity patch and checkpoint identities.
- Prompts should lead to interpretable observable comparison (short arithmetic, semantic completions, fact recall, simple format or code), but no result is predetermined. Include null results when they occur.
- Choose final gallery only from genuine outputs, identify it as selected examples, and do not infer population-wide frequency or cherry-picked performance claims.
- Save logits, tokenizer IDs, explicit readout position, prompt and edit definition, model/SAE hashes, original feature ID, protocol, reproducible scripts, public permission and SHA-256. Compute probabilities with the full vocabulary. If full generation isn't measured, do not show fabricated completion sentences.
- If weights are missing, loading is expensive or the budget is exceeded, cancel microstudy. Do not block preservation of existing working modules or silently call a synthetic demo "real". Report exactly what was blocked, while only exposing honest measured cases.

The earlier 44.9% -> 43.8% `Two plus two equals` case can be retained as an optional `Small change` teaching example, but it must NOT be the default or described as an arithmetic feature.

## 4. Completely unify the Lab design using the Feature Atlas visual system

This work is mandatory; the user reports Feature Atlas and other Labs now look structurally incompatible.

### 4.1 The Atlas is the design reference

Extract the **actual deployed** Atlas's design decisions, not approximate guesses:

- Page max width and content gutters;
- Header/logo, research-lab branding, primary navigation and active tab styling;
- Typography family, H1/H2/H3 sizes, weights, line heights and paragraph widths;
- Colors, surface elevation, border color/radius, horizontal/vertical whitespace;
- Cards, headings, controls, hover/focus, data visualization annotations;
- Desktop/tablet/mobile breakpoints and footer alignment.

Build ONE shared design-tokens file and reusable shell/components in the actual HF Space frontend source (CSS variables or framework theme). **Match the current Atlas**, rather than restyling Atlas to match the recently appended Effects module. The Atlas's feature cards, ten-prompt experience, evidence and content remain intact.

### 4.2 Consistent architecture

One common application shell wraps Atlas, Effects, Context Lab, Compare Models and Research:

```
OpenSAE logo | Research Lab
Feature Atlas | Feature Effects | Context Lab | Compare Models | Research

<shared width, background, typography and section heading>
<module content with shared spacing/components>
<consistent lower navigation and footer>
```

Avoid separate embedded app/iframe visual contexts, duplicated header/footer bars, abrupt container width changes, "one module is dark/another is light" or mixed card style. A navigation switch should not cause the header, content grid or page width to jump.

Define reusable `PageHeader`, `Section`, `ExampleSelector`, `ExperimentCard`, `PrimaryAction`, `ComparisonChart`, `DetailsDisclosure` and `ModuleNavigation` or equivalents matching the actual framework.

Use scoped module CSS, shared typography/design tokens and predictable box sizing. Audit global selectors (`button`, `h1`, `main`, `section`, `article`, `#app`, `.card`, etc.) for collision with Atlas styles. Do NOT paper over layout breaks with indiscriminate `!important`, random margin overrides, inline hard-coded widths or separate external stylesheet that globally resets components.

### 4.3 Explicitly fix the broken operator control

Current rendered string: `What we changeRemove this featureCut its activity in halfAdd 1 to its activity`.

This is a failed layout. Primary flow only needs ONE large, labeled `Remove this feature` action. Optional advanced controls must be in a distinct secondary section, each button with visible spacing, hit area >= 44px, actual numeric edit value, selected state and responsive wrapping; no labels may concatenate. Add snapshot tests for long text, narrow devices and zoom.

### 4.4 Shared visual language with existing Atlas

- Keep exactly the Atlas visual rhythm; no sudden giant KPI cards in Effects.
- Use a common responsive grid, content max width, cards, section headers, chart typography and hover/focus states.
- Align link tone: `Explore example`, `See what changed`, `Compare another`, `View evidence`.
- Decide once whether "Lab" tabs use text and glyphs; apply across all modules.
- Harmonize default headings and subtitles without changing scientific meanings. Different modules ask different questions but should feel like successive steps in the SAME research journey.
- Connect Atlas -> Effects only for the SAME measured site/feature/prompt (otherwise offer a neutral "Browse measured effects", no silent substitution).
- Preserve working Context 64-grid and RSS modes and Compare Models/Research functionality. Harmonize their containers/styles only unless a genuine regression is found.

## 5. Information architecture: the five Labs should make sense together

1. **Feature Atlas — Find a feature.** Keep existing approved experience.
2. **Feature Effects — Change it and observe the model.** Redesigned single primary action with concrete Before/After.
3. **Context Lab — Change the input and observe the feature.** Preserve actual 64-state activation study and RSS as secondary mode.
4. **Compare Models — Observe representation tradeoffs across layers.** Preserve true historical measurements, no universal ranking.
5. **Research — Study interaction and redistribution.** Preserve SAR/RSS method identities.

All modules follow a shared pattern: short question, real example, one primary interaction, chart/result with legible legend, one grounded takeaway, collapsed evidence details. Keep each scientific protocol's scope visible through subtle ordinary-language labels; do not expose raw code/IDs in first viewport.

The feature Atlas remains unchanged unless a minimal shell-level integration is required. Do not introduce any new polished-but-empty module or arbitrary text playground pretending to run inference.

## 6. Scientific and release rules

- Both columns must compare the SAME measured input and readout. Annotate whether output is a next-token forecast, whole answer probability from a measured continuation readout, or actually generated text.
- Probabilities = valid softmax of complete matched vocabulary logits, or a separately verified saved probability distribution. Never softmax just a displayed top-k subset.
- Display union of top candidates, aligned token decoding (spaces/punctuation), constant axis and explicitly labeled percentage points. The most likely next token is NOT necessarily a finished answer.
- A real feature's numerical ID belongs in the evidence details, and must match the exact SAE/site. No unfounded labels from the prompt topic.
- Larger output change = more numerical movement for THIS case; NOT improved correctness, better explanation, feature semantics or cross-model superiority.
- Keep source private if license/content rights unresolved. The meaningful default must use approved source content or correctly measured authored examples. Do not assume previously sampled third-party strings can be reprinted.
- Source paths, model hashes, edit definitions, audit results and provenance remain intact but behind `Experiment details`, default CLOSED.
- Do not use synthetic charts, fake token probabilities, invented "successful steering", fictitious rollout, placeholder case titles, nonfunctional toggles or mysterious demo numbers.
- Keep the existing offline/read-only deployment security; no public arbitrary GPU API.

## 7. Implementation order — one complete delivery, no fragmented phases

A. **Snapshot/inventory** (Section 1) and select source of truth for current frontend.

B. **Evidence-backed curated cases** (Sections 2–3). Produce an auditable data manifest, 3–6 quality verified first-screen cases and several real optional comparison examples. No GPU unless proven necessary.

C. **Implement the simple Effects experience**. Replace clutter with one meaningful before/after, one action and a clearly described change. Hide advanced edit types and research metadata by default. Keep a consistent baseline and numeric validation.

D. **Unify ALL Lab modules visually** (Section 4). Extract the Atlas tokens, replace conflicting wrappers and styles, align every route and navigation state. Fix broken concatenated controls and all responsive layouts.

E. **Regression and accessibility** (Section 8). Do not break the 10 Atlas prompts or original analysis datasets.

F. **Build/publish** updated static assets and any public-safe curated data; deploy to the real Space and verify actual served version (use a new asset build fingerprint to defeat stale caches, with no cache-busting loops). Commit changes, push to the appropriate source repositories, update concise public Lab readme only where needed.

G. **Produce final report** with actual counts, visual screenshots, checksums, commit SHAs and deployed routes. Report a genuine blocker only if one exists, don't label a failed release as "complete".

## 8. Hard acceptance gates — every item must PASS

### 8.1 Plain-language comprehension

1. First viewport shows an intelligible real prompt (not source ID) and ONE obvious intervention.
2. Original and edited measurements are clearly contrasted, with correctly aligned, readable candidate strings and actual probabilities if these are recorded; no L2-only pseudo-probabilities.
3. A one-sentence takeaway explains the actual observed change. If top prediction unchanged, clearly say so.
4. No prominent `Feature IDs`, `Recorded prompts`, `Logit L2`, `gpu04`, 64-character hash, numeric operator code or jargon wall.
5. A newcomer can answer "what input, what changed, what happened, does it mean better?" without opening research details.
6. Controls do NOT render as concatenated words at 390px, 768px or 1440px, including keyboard focus and 200% browser text zoom.
7. "Another example" visits a distinct real prompt, updates all state consistently and never lands on an empty or unapproved record. All 3–6 curated starter cases pass.
8. Any `Explore other changes` options are measured, uniquely labeled, clearly secondary and change the displayed recorded result.

### 8.2 Evidence integrity

9. Each public example has a source record, stable identifier, proper text rights, exact source hash and matched site/feature/readout.
10. Stored/recomputed values agree with raw original arrays to stated numeric tolerance. Two-token-word formatting is honest and consistent.
11. No invented generated answers, no probability from top-k-only logs, no model/task performance claim from L2 distance.
12. Current Atlas interpretability hypotheses are not silently elevated into verified feature semantics.
13. Before/after examples selected from data are described as examples, NOT a representative statistical sample.

### 8.3 Cross-module visual and functional parity

14. Capture screenshots at **1440x900, 768x1024 and 390x844** for ALL five modules, before and after. Atlas keeps recognizable approved look/content and ten examples.
15. All five modules use the same header, nav, max width, grid, fonts, card/border conventions, spacing and footer. No jumps on tab changes.
16. No horizontal overflow, clipped title, illegible controls, uncontrolled text wrapping, mismatched density or doubled borders in any module.
17. The Atlas -> Effects link preserves exact identity where available; other navigation defaults to a valid real case.
18. Context Lab 64-state toggle/heatmap, RSS tab and existing model comparison and research module interactions pass regression tests.
19. Mobile touch targets and focus outlines meet basic accessibility needs; charts carry textual alternatives.
20. Browser console is clear of uncaught JS errors and public assets all respond successfully, including after hard reload and deep-link navigation.

### 8.4 Real deployment

21. Verify the deployed HF Space source revision and actual static asset fingerprint against the pushed commit; confirm not an old cached version.
22. Open live `#effects`, `#context` and Atlas on desktop/mobile; manually execute at least three different curated cases, use restore/replay, and test every five-module navigation transition.
23. Compare measured screenshot evidence to the pre-change captured Atlas baseline. Inspect both global design coherence and the specific "concatenated controls" issue.
24. Save final artifacts under `outputs/research_lab_final_07/`, with test report, three viewport screenshots per module, per-case source/provenance manifest, deployment revision and original/updated screenshot map.

If ANY gate fails, fix and rerun the test before declaring completion. Do not ask for another independent work plan or stop at "next steps."

## 9. Explicit final deliverables

- Updated current source-of-truth Lab frontend, working live HF Space, and source commit(s).
- Curated versioned real Effects case dataset and deterministic small readout/validation script, where needed.
- Existing Feature Atlas visually preserved, with a single shared product design system across all five modules.
- No private data leakage, fake behavior, or new broad GPU job.
- `outputs/research_lab_final_07/final_report.md` containing: screenshots for 5 modules x 3 viewports, exact case counts/which were newly measured, any blocked source rights, provenance checksum report, tests, Space build fingerprint, commit SHA(s), and 24 acceptance gate PASS/FAIL lines.

**The success metric is visitor understanding, not the number of plotted values or the number of interface buttons.** A code-accurate screen of "44.9% to 43.8%" with no meaningful user action and no unified layout is NOT finished.

## 10. Agent's first instruction

Read this entire Work Plan 07 and execute it from Section 1 to Section 9 without splitting into further planning documents. The user's existing Feature Atlas is the design reference. Reuse recorded data first. Rebuild Feature Effects as a concrete before/after with one main action. Unify the shared shell for all Labs. Run actual browser tests, deploy, verify the public page, and leave a factual final report. No invented evidence, no broad model reruns, and no "done" until all acceptance gates pass.
