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

## 2. Final interaction design — identify the feature BEFORE asking users to edit it (revised 2026-10-20)

This section **replaces the earlier Effects UX proposal**. The latest deployed view still has redundant microcopy such as `A REAL INPUT`, the unexplained buttons `Remove this feature` / `Restore original`, and an isolated `Original` heading. This is not understandable. **Do not add another introductory paragraph or tooltip to an otherwise unchanged interface. Change the information architecture itself.**

### 2.1 The actual research question

What a visitor must understand, in precisely this order:

1. **Input** — What did the user write, and which model read it?
2. **Selected feature** — Which exact learned SAE coordinate is the experiment changing, and why was THIS coordinate chosen? A feature is not a prompt word, class label, weight, neuron or permanently deleted component.
3. **Intervention** — What does the control do physically? It sets this selected coordinate's activation to zero at the recorded intervention site and token, while keeping the input and model parameters unchanged. In SAEBase's actual residual-preserving patch, the change is propagated through the remaining model layers. Avoid calling this "removing a neuron" or "deleting knowledge".
4. **Outcome** — What did that ONE edit change in the measured next-token distribution, not what we hope it means?
5. **Limit** — One example shows a local model response. It does not by itself identify the feature's general semantic function or prove answer quality improved.

The screen is not a research dashboard. The natural flow is **input → one named, observed internal feature → switch its activation off → compare predictions**.

### 2.2 Exact visual hierarchy and copy

Use a short, clean top heading:

`What happens when we turn off one feature?`

Do NOT display a lead paragraph that repeats the heading. Do NOT print a generic all-caps `EXAMPLE` / `A REAL INPUT` / `WHAT WE CHANGE` / `MEASURED OUTPUT` / `LIVE EXPERIMENT` / `REAL DATA` / `ACTIVE EVIDENCE`. No rows of small explanatory text, badges, unnecessary provenance warnings, decorative icons or repeated footers inside modules.

One source-backed curated card contains the following (illustrative layout only, NEVER use made-up values in production):

```text
What happens when we turn off one feature?

"Two plus two equals"                          [Another example]
Gemma 1B

Inside the model
Feature #74449       [small observed-activation visual]
This feature activated while the model processed this input.
[Only if evidence exists: "Also observed on ..." with actual reviewed examples]

We set this feature's activity to zero for this prediction.
The input and model weights do not change.

Feature #74449           [ ON   |   OFF ]   < one switch with clear active state >

WHEN ON                                WHEN OFF  < shown after user toggles off >
four               44.9%              four                  43.8%
five               24.4%              five                  27.7%
...                                   ...
<identical vocabulary candidates on identical scale>

What changed?
The top prediction stayed "four", but its probability decreased by
1.1 percentage points for this particular input.

[Another example]                     [Experiment details ▾]
```

**Critical:** The quoted text and the above numbers are ONLY a structural illustration based on a previously shown measurement; that weak case is NOT an acceptable default. The actual chosen case MUST be one of the strongest PUBLIC-SAFE, measured and comprehensible cases selected under Section 3. Numbers and words must be read from its raw evidence. The site must not ship the above static mock content.

### 2.3 Make the feature a real visible object, not a mysterious pronoun

- The selected feature is identified ON SCREEN as `Feature #<real ID>`, associated with a human-readable model name, and matched to exactly one SAE checkpoint/site. This identifier is essential to answer "**which** feature?" — it is NOT the old meaningless KPI card labeled `Feature IDs`.
- Next to that identity, show a minimal actual activity indicator for the current input. Use `Active on this input` only if a nonzero activation is actually measured for the SAME source prompt, same site and same token. Optionally show `measured activity → zero` if the precise saved activation value is matched and finite. Do not fabricate a percentage activation meter or a universal feature activation scale.
- If the approved Atlas contains reviewed descriptions or activation examples for this exact model/site/feature, use a concise description like `This feature also activates on these recorded examples` and show 1–2 real, rights-cleared snippets on an optional expanded card. A reviewed Atlas description is a hypothesis, NOT a proved single-concept identity.
- If there are no reviewed semantic examples, DO NOT label it `arithmetic feature`, `fact feature` or `truthfulness feature`. An exact ID + measured activation + mechanistic intervention explanation is honest. For the introductory gallery, strongly prefer measured features with approved contextual examples so a visitor learns why the feature was selected, not just its index.
- If the selected feature is not observed active for this same input, do not say it "activated" or choose it as the beginner default. Check the original codes and intervention semantics; do not pretend setting an already-zero coordinate off caused a visible effect.
- The feature is a *single SAE coordinate*; the model is not selecting the "most important" feature or proving it corresponds to a linguistic concept. Avoid the phrase `we remove this feature` unless it is immediately clear that **only its local activation is zeroed**.

### 2.4 One understandable switch, no competing buttons

**DELETE the unexplained pair `Remove this feature` / `Restore original`.** Replace with a single **stateful** ON/OFF control attached directly to the visible feature card: `Feature #74449 [On | Off]`, whose selected state is visually unmistakable. Use `Turn off` as the action hint if needed, not another separate button. `On` denotes the normal, unedited model computation; `Off` denotes the measured zero-activation intervention at the pinned site.

- Default state: ON, normal output shown. When the visitor switches OFF, reveal/animate the actual edited prediction side by side with the fixed normal prediction. When switched ON again, restore the original presentation. Never imply the permanent model weights changed.
- This is comparison of two previously measured states, NOT live GPU computation. Avoid `Run inference`, `AI is thinking`, spinners that suggest online model generation, and fake intermediate states.
- Switching to a different example changes the prompt, model, feature identity, matching activation, measured outputs and state together. Reset the switch to ON to prevent stale outputs.
- **Do not put other edits on this introductory screen.** If needed, a collapsed secondary `Try other measured changes` view may provide detailed factor/mode controls after the primary interaction is understood; omit it entirely when it creates clutter. Never show generic `Add 1`, `Cut its activity in half`, `clamp` or `coalition` controls disconnected from a named and measured feature.
- The phrase `Original` as a stand-alone section heading is prohibited. The prediction labels must answer which state: `Feature ON (normal)` versus `Feature OFF`, or `Normal prediction` versus `With feature #74449 off`.
- Real readouts must use candidate alignment and same probability axis as required by Section 6. A token may be a token fragment; do not masquerade next-token predictions as complete answers.

### 2.5 The correct explanation is built into the action, not packed into footnotes

The ONLY mandatory explanation, placed immediately adjacent to the feature switch, is concise ordinary-size text:

`We set this feature's activity to zero for this prediction. The input and model weights stay the same.`

This answers "**how**" before user interaction. If feature activation and token position are actually known, the Evidence drawer can describe the exact residual-preserving activation edit and selected token. Never leave `remove` unexplained.

Place one clear result sentence after the plot:

- Changed top candidate: `Turning off feature #N changed the model's top next-token prediction from “A” to “B” on this input.`
- Unchanged top candidate: `The top prediction stayed “A”, but its probability fell from X% to Y%.`
- Similar distributions: `Turning off this feature had little effect on the recorded next-token predictions for this input.`

Use real matched numerical evidence and avoid ungrounded adjectives. If data are only L2 aggregate, this probability flow cannot be shown; do not invent before/after words. Such cases belong in secondary measurement browsing, not the introductory experiment.

The only appropriate limitation in the main flow, IF necessary to prevent misunderstanding, is one normal-size sentence: `This shows a change for one input, not what the feature means in general.` Do not repeat it in small print, badges, captions, nav, and footer.

### 2.6 Visual design: remove microcopy, preserve meaning

1. Short main heading, one actual prompt, one named feature with real activation evidence, one switch, one prediction comparison, one result. That is the **entire default reading path**.
2. Normal body text for any sentence required to understand the action; no 10–12px disclaimers or dense explanatory notes. The existing Feature Atlas sets the typography, cards, spacing and palette.
3. Delete headings and labels that do not add meaning, especially `A REAL INPUT`, `EXAMPLE`, `What we change`, `Original` standing alone, `Recorded prompts`, `Selected logit L2` and meaningless accuracy/quality badges.
4. Model identity and feature identity remain visible because they ARE meaningful. Keep raw feature/checkpoint hashes, protocol, site hook name, exact layer, raw activation value (if visually cluttered), and source links inside CLOSED `Experiment details`.
5. The switch must sit INSIDE the selected feature card, not above the prompt or on a disconnected action toolbar. All controls get adequate spacing, keyboard focus, responsive wrap and mobile touch size.
6. No image-generation assets, decorative stock images, tiny legends, pictograms or made-up colorful data graphics. Use text, spacing, and measured compact bars to convey the actual observation.
7. Do not make every line a card. Prefer one integrated experiment panel with a light visual flow and one chart.

### 2.7 Immediate acceptance tests (add these to the existing 24 gates)

- T01 **Identity first:** before the user sees any editing control, they can point to the selected `Feature #N`, name the model, and state that this was an actually observed signal for the selected input.
- T02 **Mechanism before action:** without opening advanced settings they can read that the feature's activity is set to zero *for this prediction*; the prompt and model weights do not change.
- T03 **One switch:** `Remove this feature` and `Restore original` are gone. One feature-attached ON/OFF switch can be toggled both ways; no vague standalone `Original` heading remains.
- T04 **Real examples:** all introductory examples have an input the visitor can actually read, matched nonzero feature evidence and genuine paired predictions; case selection updates all three and returns the switch to ON.
- T05 **No hallucinated meaning:** arithmetic prompt ≠ `arithmetic feature`; no feature label based only on a single input. Atlas reviewed interpretations retain their uncertainty.
- T06 **Honest presentation:** raw logit-L2-only records never masquerade as probability bars. If same-input activation is absent, no fake activation visual or unsupported `Active` claim appears.
- T07 **Compact screen:** desktop and mobile first viewport have no all-caps `A REAL INPUT`, `EXAMPLE`, extra tiny legend, repeated interpretation disclaimers, raw hashes, separate restore button or redundant metadata boxes.
- T08 **What changed:** aligned normal/off candidate displays and plain-language result explain exactly which prediction changed or stayed the same, with real percentage-point arithmetic.
- T09 **Unified styling:** match the CURRENT Feature Atlas title hierarchy, card proportions, content width, nav and footer across all five modules; no Effects-specific CSS resets that modify Atlas.
- T10 **Release proof:** screen recordings or browser interaction evidence verify the full sequence: change case → identify signal → switch OFF → observe changed comparison → switch ON. Verify at 1440px, 768px and 390px and publish the tested HF Space revision.

Any violation is a FAILED Section 2 redesign even if unit tests pass. An appealing but incomprehensible "turn off this feature" button is not complete.

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

This is a failed layout. Primary flow needs ONE feature-attached, explicitly explained ON/OFF switch, with a clear identity and state as specified in the revised Section 2. Optional advanced controls belong in a collapsed secondary section, each with visible spacing, touch area >= 44px, actual numeric edit value, selected state and responsive wrapping; no labels may concatenate. Add snapshot tests for long text, narrow devices and zoom.

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
2. **Feature Effects — Identify a measured internal feature, turn its activity off, and observe the model.** One named feature, one explained ON/OFF switch, a real prediction comparison.
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

C. **Implement the simple Effects experience**. Replace clutter with one named and evidenced feature, an explanation of setting its local activation to zero, one attached ON/OFF switch, and one measured prediction comparison. Hide advanced edit types and research metadata by default. Keep a consistent baseline and numeric validation.

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
22. Open live `#effects`, `#context` and Atlas on desktop/mobile; manually execute at least three curated real cases, verify the feature identity and zero-activation explanation, toggle the same feature ON/OFF in both directions, and test every five-module navigation transition.
23. Compare measured screenshot evidence to the pre-change captured Atlas baseline. Inspect both global design coherence and the specific "concatenated controls" issue.
24. Save final artifacts under `outputs/research_lab_final_07/`, with test report, three viewport screenshots per module, per-case source/provenance manifest, deployment revision and original/updated screenshot map.

If ANY gate fails, fix and rerun the test before declaring completion. Do not ask for another independent work plan or stop at "next steps."

## 9. Explicit final deliverables

- Updated current source-of-truth Lab frontend, working live HF Space, and source commit(s).
- Curated versioned real Effects case dataset and deterministic small readout/validation script, where needed.
- Existing Feature Atlas visually preserved, with a single shared product design system across all five modules.
- No private data leakage, fake behavior, or new broad GPU job.
- `outputs/research_lab_final_07/final_report.md` containing: screenshots for 5 modules x 3 viewports, exact case counts/which were newly measured, any blocked source rights, provenance checksum report, tests, Space build fingerprint, commit SHA(s), and 24 acceptance gate PASS/FAIL lines.

**The success metric is visitor understanding, not the number of plotted values or the number of interface buttons.** A code-accurate screen of "44.9% to 43.8%" with no named, evidenced feature, clear local zero-activation intervention, meaningful user action and unified layout is NOT finished.

## 10. Agent's first instruction

Read this entire Work Plan 07, including the REVISED Section 2, and execute it from Section 1 to Section 9 without splitting into further planning documents. The existing Feature Atlas is the design reference. Reuse recorded data first. In Feature Effects identify the exact observed SAE feature before the user can intervene, explain that its activation is set to zero for one prediction, then show one ON/OFF control and matched real before/after predictions. Delete unnecessary microcopy and labels. Unify the shared shell for all Labs. Run real browser tests, deploy, verify the public page, and leave a factual final report. No invented evidence, no broad model reruns, and no "done" until all acceptance gates pass.
