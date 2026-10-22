# Work Plan 14 — Ship a Useful SAEBase Research Lab

> **Superseded for current execution by [WP15](15_complete_six_workflows_and_release.md).** This WP14 file records an earlier ship-or-cut strategy. G1–G6 are now mandatory engineering-completion targets; do not follow its HIDE/DEFER directives for the current release.


Updated 2026-10-22. Release-first decision by the project owner.

**This is the active release plan.** It supersedes WP12's all-50-capability goal, formal P8 signoff, and WP13's broad LIVE-00–10 completion sequence **as launch requirements**. Older plans, experiments and the 50-item ledger remain historical engineering records; do not delete or rewrite their evidence. Read the [short GPU Agent assignment](14_gpu_agent_release_handoff.md) and [release scope sheet](14_release_scope.csv) with this plan.

## 0. Mission and operating rule

Ship the smallest **useful, attractive, working** research platform rather than completing a feature-count checklist. Give researchers several real things they can do reliably. Polish supported workflows, repair straightforward blockers, and remove unavailable menus or claims. No feature receives a release commitment merely because a wrapper, route or unit test exists.

**Release success = someone can open the Lab, understand what works, inspect real feature data, reach a measured intervention, find a usable API/CLI entry point, and repeat the advertised experience.** This is a practical product check, not a comprehensive paper-quality science audit.

Work in two surfaces, and label them correctly:
- **Public Research Lab**: CPU-hosted, immediately usable **recorded real** evidence and reports. It does not promise live inference on arbitrary input or a hosted GPU.
- **Local SAEBase**: Python/CLI against a supported real model and SAE, with **fresh live** capture/intervention for an operator who has the weights and GPU. An authenticated private worker is useful if the already-developed route works; it is not an obligation to host a public inference service.

Do not switch the existing Hugging Face Space to Gradio or ZeroGPU merely to add platform badges. Prefer the existing FastAPI/Docker + browser app if it can serve a polished and reliable release. Gradio, MCP and agents are future integration options, not launch blockers.

## 1. Release scope, not the 50-capability ledger

### A — Commit to ship, fix until working

1. **Real feature explorer.** Choose a known working model/SAE/site, open a curated prompt, inspect original tokens and actual nonzero feature codes, click a token, open a feature detail page, navigate back, and search within recorded examples. Use existing Gemma-3-270M L5/L9 LIVE-02 traces if usable. Publish one excellent end-to-end example first; expand to roughly 6–10 curated prompts only when effortless. Do not block on bulk 500-source dataset curation.
2. **Measured intervention.** From a feature page open a matching real baseline/no-op/ablation/scale comparison. Use the existing LIVE-03 Gemma L5 F506 and L9 F95 small summary, and distinguish edited-position from final-position logits. No-op must remain unchanged. Show a clear before/after effect, not an unexplained number dump. Recorded teacher-forced logits are useful without pretending they are generated text.
3. **Research results and existing Python workflow.** Preserve working installation, supported model/SAE loading, activation caching, intervention API/CLI, existing valid benchmark/reliability results, and SAR/RSS/Input Localization integration. Make the fastest supported Python example and one clean install or documented reproducible environment actually run. Expose existing results with readable summaries and links rather than empty modules.
4. **Public Lab deployment.** A staging build and then the existing Space serve the same real demo and accessible navigation. Public preview must be usable after restart. The user must not encounter a wall of nonfunctional feature menus. Include accurate scope, links to code/docs/data, and an easy launch path.

These are the release-critical workflows. If the first real experience is broken, fix it before adding more features.

### B — Candidate features: one bounded attempt, then ship or cut

- **Fresh-input local/private worker (LIVE-08):** high-priority candidate. Exercise two new owner prompts through authentic load → capture/edit → result; keep private. If durable queue plumbing is troublesome, ship direct local Python/CLI live inference and defer the queue. No public GPU service required.
- **Actual generation-time steering:** try one tiny continuation with an existing working local backend and actual active feature. If output and control are correct, offer it as a separate live local operation. Otherwise ship measured logits only.
- **Original native SAEBench pilot:** one compatible original Core or Sparse Probing run producing actual importable results. If it works and the UI is useful, ship; if the upstream package/dataset is unavailable, hide the execution control but keep existing recorded supported benchmarks.
- **One native circuit:** one compatible real attribution graph and graph browser. If generating a graph is unavailable, a previously licensed authentic graph may be browsable but **do not** call it a newly computed circuit. Hide live circuit computation controls if missing.
- **Tiny native SAE training and checkpoint reload:** only if existing cached sources and original native trainer make this cheap. Test **one** framework rather than completing both SAELens and Sparsify for launch. If it cannot produce and reload a usable checkpoint quickly, defer training UI; existing separately documented code remains available as experimental.
- **Automatic feature explanation:** only if positive/negative examples and a compatible local model already exist, and it produces a meaningful reviewed example. No label from a single token or a fabricated geography concept. Otherwise omit explanation rather than blocking feature exploration.
- **TL4, NNsight, pyvene, Scope 2 CLT adapters:** keep supported working code, but do not require official real-checkpoint parity for every backend. Only advertise a specific backend workflow if a short real example runs in the release environment.

**Candidate rule:** one bounded real run; a small targeted repair if clearly tractable; then choose SHIP or HIDE/DEFER. No feature is permitted to absorb the schedule without an evident user benefit.

### C — Remove from this release's scope now

- Remote NDIF service and external credentials (former F20).
- True two-or-more-GPU distributed training/resume (former F50).
- Mandatory full 50/50 demonstration, 50 independent P8 certifications, TL4/native parity at every registered site, and repeating GPU-00/48-site archives.
- 500-source prompt corpus curation and comprehensive benchmark sweep.
- Public arbitrary-prompt GPU inference, general remote execution, a payment service or hosted model serving promise.
- Gradio/ZeroGPU/MCP/agent migration for its own sake, automatic publishing of model-written interpretations, or replacing original scientific estimators.

Keep old documentation readable, but hide incomplete public controls; do not market these as launched. They can return later **only** if there is user demand and a working path.

## 2. What “done” looks like for an individual released feature

Practical checks, not a giant certification process:

1. **Actually performs the advertised action.** A real model, authentic existing record, or original upstream evaluator executes as stated; an empty UI, mocked example or JSON-only ingestion cannot masquerade as fresh compute.
2. **Usable by someone besides the agent.** A readable entry point, one correct working example, obvious next step, sensible error and return navigation. Do one genuine browser walkthrough, not just a unit test.
3. **Output is correctly labeled.** Recorded versus live, token logits versus generated text, correlation versus intervention, and model/SAE identity only where needed. An observed effect need not prove semantics.
4. **Reasonable speed and operation.** A small prompt finishes under the available hardware without OOM; no needless reloading in a hot path; data/assets survive staging restart.
5. **No accidental data or security exposure.** Do not publish private prompt text, secrets, weights or raw trace databases without permission. This is ordinary product hygiene, not a reason to demand a historical raw-archive audit.

A significant crash, incorrect displayed measurement, inaccessible demo, or exposed private endpoint blocks the affected feature. A missing optional backend does **not** block the rest of the launch: remove that item.

## 3. Immediate GPU batch and CPU integration order

**Do not rerun LIVE-00/01/02/03 or GPU-00 by default.** Previous agent reports provide product-development leads; no new “independent P8 parity” project is necessary for this release. Inspect actual GPU assets and scripts before deciding a missing file needs regeneration.

| Priority | Owner | Job | Desired product outcome | Release decision |
| --- | --- | --- | --- | --- |
| P0 | CPU Integration | Integrate existing LIVE-02 and LIVE-03 compact data into an isolated CPU staging Lab with one curated, authorized prompt | Working Prompt → Token → Feature → Measured Intervention, screenshots and click path | Mandatory. Fix first. |
| P0 | GPU Agent | G1 actual fresh owner prompt via LIVE-08; fallback direct local Python if queue is unavailable | Working new-input SAE codes and/or measured edit, tiny result + resource note | Ship local live even if hosted queue is cut |
| P0 | CPU Integration | Working install + actual README example, navigation cleanup, production-like staging and restart | Real usable public preview, no dead buttons | Mandatory |
| P1 | GPU Agent | G2 one short actual generation-time feature edit | Optional continuation demo beyond logits | Ship if working, otherwise hide |
| P1 | GPU Agent | G3 one genuine native SAEBench evaluation | One new real benchmark example and openable result | Ship if working, otherwise recorded metrics only |
| P1 | GPU Agent | G4 one compatible native graph and node flow | One browseable circuit with source-backed graph and honest result | Ship if working, otherwise hide |
| P2 | GPU Agent | G5 one small native train → reload, optional resume if already supported | A usable checkpoint and two-command local tutorial | Candidate, easy to defer |
| P2 | GPU Agent | G6 one local explanation with held-out examples | An example users can interpret responsibly | Candidate, easy to omit |
| Final | CPU Integration + reviewer | One end-to-end deployed browser walkthrough and CLI check | Concrete release candidate with working screenshots | Go / go with cuts / no-go |

Run G1 first and candidates in parallel only when models, assets and VRAM permit. The GPU Agent should stop when the release decisions are settled; unused experiments from WP12/WP13 are not debt for this preview. See the short handoff for current known scripts.

## 4. CPU Integration Agent deliverables

- Work on the WP13 integration branch and reuse PRs #50–54; do **not** pretend they are in main already. Reconcile the stacked PRs before the final release merge.
- Stage **one** rights-appropriate owner-authored prompt with the existing LIVE-02 complete token codes using the local in-place review/export path. Start with one; improve copy, visual hierarchy, token highlighting, feature snippets and next-step links; then add selected prompts.
- Stage the two measured LIVE-03 intervention cards through the existing validation adapter only for reviewed source-bound records. Do not invent “Paris interpretation”, continuation or causality claims.
- Landing page must contain a prominent real demo and a concise “What you can do” explanation. Group or hide dead navigation such as Training/Circuits/Jobs when unavailable. If only historical comparison results exist, mark them “Recorded results” rather than “Run”.
- Provide an obvious primary Python/CLI quick start for actual local use. Keep dependencies focused; do not require optional backends for a default install.
- Smoke API against **populated** read-only staging data, not just HTTP 200 or test fixtures; open the same deployed page from a clean browser and verify reload, back/forward and one export. Preserve production Space revision and rollback path.
- Deploy the existing static Space **only after** isolated staging visibly works. Do not publish raw SQLite/traces/tokens from unreviewed third-party prompts or operator tokens.

## 5. GPU Agent output: small and actionable

For G1–G6, output a single short release receipt instead of a big archive:

- Source commit and actual hardware/VRAM; exact model, SAE, layer/hook, library versions and prompt ID/hash.
- What a user did (command or API request), what actually worked and one real value/output. For no-op/ablation record meaningful deltas and the generation distinction.
- Runtime, peak VRAM, filesystem location of the private artifact, one short sanitized result JSON or screenshot reference.
- Decision recommendation **SHIP**, **FIX_ONCE**, **HIDE**, or **DEFER**, with one concise reason. Failed dependency is a valid final outcome; no repeated retries to manufacture a PASS.
- Do not transfer huge ZIPs, full logits, private prompt text, checkpoints or complete SQLite files into chat. Keep original artifacts on the GPU host for reproducing a specific problem.

The reviewer checks usability, reproducibility of the advertised path, clear product claims and that public staging contains the actual data. The reviewer may ask for **one targeted fix or demonstration**, not a new all-sites audit.

## 6. Release assembly and go/no-go

1. **Cut the feature menu.** Complete the A workflows; take only B items marked SHIP with actual linked output, hide others from public pages and marketing.
2. **Build one release candidate.** Document a clean installation or tested environment, exact demo configuration, known unsupported modes, one documented CLI command, one accessible recorded browser link, and rollback revision.
3. **Quick acceptance by reviewer (not P8):** open deployed staging, complete demo without special knowledge, inspect one real feature and intervention, run one local quick-start or inspect GPU Agent's executable receipt, confirm no private route is public and that restart does not erase records.
4. **Release decision:** GO if all A workflows work; GO_WITH_CUT if A works and optional features are hidden; NO_GO only for broken core paths, false displayed values, dangerous exposure or unreachable public demo.
5. Merge the reviewed WP12/WP13 integration chain, publish version/release notes and updated README/Space, link a clear tutorial, and keep the earlier release recoverable. Announce **what is working**, not a 50-capability achievement.

There is **no requirement** for P8 RELEASED 50/50, historical full-native audit, comprehensive paper-level statistical validation, 3 external-user signoffs or broad model coverage before this product preview. Honest limits remain visible without turning the release into a certification exercise.

## 7. First actions, in order

1. GPU Agent receives [14_gpu_agent_release_handoff.md](14_gpu_agent_release_handoff.md) and works from WP13 source rather than main.
2. CPU Integration Agent immediately stages the already available LIVE-02/03 outputs and repairs the one end-to-end Research Lab page.
3. When the compact GPU receipts arrive, the reviewer checks each live capability and makes the **ship or hide** call from actual user behavior.
4. Stage and release only the resulting compact, functional product. Everything else waits for user demand.

Related current records: [WP13 live summary](13_live03_review_and_gallery.md), [WP13 execution matrix](13_live_execution_matrix.csv), [WP12 source assessment](12_core_function_assessment.md). These are evidence/context, **not extra release gates**.
