## Work Plan 19 · 200 个候选考试与验证器可信性

[**WP19 — Harness 大规模考试与 Verification 可信性计划**](19_harness_large_scale_examination_and_verifier_assurance.md)：提出 H27–H226 共 200 道下一阶段候选考试，并明确冻结 Case Spec、可信 Reference、Mutation / Metamorphic / Differential / Stateful / Compositional / Multi-Agent 测试及长任务恢复。**仅为设计规划，尚未实现 200 道评分器，也不替代正在运行的 WP18。**

## New Work Plans · 预先声明 Contract

新建 Work Plan 时，先从 [通用 Contract Registry](../contracts/README.md) 选择现有 `CT-` ID，在 Work Plan 的 `harness-contracts-json` 声明中逐 CASE 绑定 tags 和合同。Always-on 自动继承，不能取消；未知新任务先从固定 tags 选择，并把选择结果写入测试 Run 收据。**不允许为了通过一次考试临时重定义 Contract 或让本地 Agent 修改治理目录。** 历史 Work Plan 未迁移声明的只能算 legacy，不能声称已经完成 Contract 验证。

## Harness-only local Agent examination — WP18

[**WP18 — Harness 测验与独立验收**](18_harness_local_agent_examination_and_independent_acceptance.md)：针对 Cursor Rules、Skills、Hooks、Agent Flow、Work Order、Code/Context/Environment 与 Atlas 五棵树的测试计划。包含 12 组单元/负面验证、6 个需要在隔离分支中真实修改代码的小任务、候选 Tree 交付要求及独立评审评分门槛。**本次治理 PR 合并后，Harness 测验可通过 Cursor Agent 模式自然语言直接启动，产物统一放在 `harness/tests/`；正式治理代码仍需独立审批。当前尚未执行 H01–H06；不涉及 SAE 科研功能，也不取代 WP15–WP17。** [开考前缺陷登记及门禁操作手册](18_harness_exam_preflight_defects_and_gate_runbook.md)。

## Current product execution — WP17

[**WP17 — Seven playground compute handoff**](17_seven_playgrounds_agent_handoff.md): the active, single GPU Agent task list. All seven example-first browser modules are implemented; reuse WP13–WP15 assets and replace only marked Demo results with real compute.

## Current website execution — WP16

[**WP16: OpenSAE Vision + real SAEBase API Workspace**](16_web_vision_api_single_execution.md) is the **single active web integration and GPU Agent handoff**. The V5 static website and API-aware Workspace are on the live website repository; existing WP13–WP15 model results and APIs must be reused. No new plan or repeated GPU sweep.

# Engineering Plan Archive

### Current execution directive: WP15

**WP15 is the authoritative release-completion plan.** G1–G6 are mandatory working product workflows; do not defer G2–G6 merely because packages, upstream checkouts or local checkpoints are missing. GPU Agent must prepare compatible inputs and execute the real original tools; Integration Agent must show real recorded results in a usable Research Lab. WP14 SHIP/HIDE/DEFER was a historical receipt review, not the current completion mandate.

- [**WP15 — Complete all six workflows and release**](15_complete_six_workflows_and_release.md). Concrete engineering decisions, native integrations, user journeys and practical acceptance.
- [**WP15 — GPU Agent command handoff**](15_gpu_agent_completion_runbook.md). Version setup, repair routes, commands, genuine artifacts and frontend integration.
- [**WP15 — Completion matrix**](15_release_completion_matrix.csv). One row per mandatory feature and the release candidate; missing dependency is not a terminal status.
- [**WP15 — Pre-release remaining handoff**](15_pre_release_remaining_handoff.md). Actual CI/CPU integration fixes completed; operator/GPU Agent follow-up is limited to approved real prompt assets, one private edit queue smoke, G2 decoded demos, G6 meaningful interpretation and remote staging proof.
- [**WP15 — GPU asset review**](15_gpu_asset_review.md). One approved trace, two reviewed generation examples, and one private edit whose logits actually changed. Not a production release.


## Work Plan 14 — Active release-first plan (takes precedence for launch)

**Current priority: ship useful, working SAEBase features.** This supersedes WP12's full 50-feature certification and WP13's eleven-job completion list as **launch requirements**. Existing work plans and data remain historical references, not prerequisites. Optional missing capabilities should be hidden/deferred rather than block a usable release.

- [**14 — Ship-first release plan**](14_ship_first_release_plan.md). Required real Feature Explorer → Prompt/Token → Feature → measured Intervention, working Python quick start, polished public CPU Lab, staged deploy/restart, GO/GO_WITH_CUT criteria.
- [**14 — GPU Agent release handoff**](14_gpu_agent_release_handoff.md). Bounded G1–G6 genuine smoke runs, exact feature-specific SHIP/FIX_ONCE/HIDE/DEFER reports, existing GPU asset reuse and no raw archive transfers.
- [**14 — Release scope**](14_release_scope.csv). Launch-required, optional and explicitly cut features with a user-visible minimum demo and owner.
- [**14 — GPU receipt review**](14_gpu_receipt_review_20261022.md). Agent-reported G1 accepted for local live; optional G2–G6 hidden/deferred; public staging remains the next release gate.


This folder holds plans and execution records used while building SAEBase. Start with the [project README](../../README.md) and [documentation](../github_development_doc/index.md) for current installation and research tasks.

- Early plans: model/SAE acceptance, GPU coverage and Research Lab rollout.
- Performance plans: model-forward optimization, checkpoint integrity, suffix replay and KV-cached steering.
- Final measured performance, including Qwen3-1.7B intervention and generated-token results, is reported in [Work Plan 11 results](../github_development_doc/performance/workplan11_final_results.md).

The two Work Plan 11 files reflect an original broad plan and the later GPU execution handoff; they are historical records, not competing user-facing installation instructions. Existing paths remain intact so external references and experimental records do not break.

## Work Plan 12 — Unified OpenSAE research infrastructure

The new long-term master blueprint integrates the **50 functional capabilities of 10 established interpretability projects** through a single SAEBase API and Research Lab, retaining all historical plans and experimental evidence.

- [**12 — Master blueprint**](12_unified_interpretability_master_blueprint.md). Nine phases P0–P8, upstream feature-to-UI mapping, API architecture, 50 exact acceptance tests, local/GPU split, C0–C3 compute tiers, scientific boundaries and release policy.
- [**12 — GPU agent handoff**](12_gpu_agent_handoff_protocol.md). Exact phased GPU/CPU responsibilities, preflight, run artifacts, rate measurement, stop rules and independent signoff.
- [**12 — 50-capability ledger**](12_capability_acceptance_matrix.csv). Core implementation and final release are tracked separately: 48 source-level PASS, 0 core PARTIAL, 2 external BLOCKED, with all independent P8 verification pending. The historical WP12-00 audit remains a dated baseline.
- [**12 — P0 source audit**](12_p0_source_inventory_20261021.md). Current code, Research Lab, CI/release and compatibility inventory; P0 gate remains open.
- [**12 — GPU-00 runbook**](12_gpu00_p0_parity_execution.md). First bounded C1 real-model numerical parity test with environment preflight, exact model/SAE/site choices, operation definitions, tolerances, resource stop rules and immutable handoff outputs.
- [**12 — GPU-00 frozen input pairs**](12_gpu00_prompt_pairs.json). Eight small original paired prompts, site list and tokenization acceptance policy.
- [**12 — GPU-00 agent report and independent review**](12_gpu00_agent_report_20261022.md). Reports four passing legacy fp32 model/SAE sites, preserved failures and resource counts, while explicitly leaving raw-artifact verification, TL4 compatibility and overall P0 acceptance open.
- [**12 — P1 implementation and short smoke**](12_p1_model_backend_implementation.md). Version-aware TransformerLens loading, missing-hook protection, residual component inspection, local NNsight tracing, Gemma Scope 2 metadata/variants, lightweight tests and limited real-weight smoke.
- [**12 — P2 feature explorer**](12_p2_feature_explorer.md). Real-only feature dashboards, explanation/recorded-prompt/token searches, local sparse prompt capture, correlations, hypothesis provenance and portable offline HTML; no fake online inference.
- [**12 — P3–P8 consolidated implementation**](12_p3_p8_consolidated_implementation.md). Typed interventions, bounded original SAEBench, native training/restart, Circuit Tracer graph conversion/UI, private worker and independent release checks. Scientific and public release acceptance still remains open.
- [**12 — Core functionality verdicts**](12_core_function_assessment.md). A strict 50-item implementation assessment: **48 core PASS, 0 PARTIAL, 2 BLOCKED**, with per-item executable criteria, source/test references and specific remaining gaps. Independent P8 scientific validation is separate.

Work Plan 12 now has developed core source routes but is **not** a claim that all functions have been executed with real official checkpoints or deployed. Follow older WP04–WP11 experimental protocols and accepted findings.

## Work Plan 13 — Real-environment integration and deployment

Turn WP12's source-level implementations into actual runnable research workflows and a useful deployed Research Lab, prioritizing genuine model/prompt experience over repeated formal numerical certification.

- [**13 — Real-environment integration and deployment**](13_real_environment_integration_and_deployment.md). Architecture, 10-prompt real-feature browser, genuine interventions, native training/benchmarks/circuits, private local GPU queue, source-reviewed dataset publication, HF staging/rollout, usability and recovery.
- [**13 — Local GPU Agent execution handoff**](13_gpu_agent_real_environment_handoff.md). Exact LIVE-00 → LIVE-10 task sequence, commands, isolated library environments, bounds, return artifacts and Integration/GPU agent repair ownership.
- [**13 — Machine-readable LIVE task queue**](13_live_execution_matrix.csv). Eleven execution tasks with dependent prerequisites, resource caps, measured success definitions, minimal handoffs and honest external blockers.
- [**13 — Ten candidate demo prompts**](13_demo_prompt_candidates.json). Original engineering text candidates with fixed IDs and 6/4 discovery/evaluation split, all explicitly unreviewed; real model activations and release approval must be added separately.
- [**13 — LIVE-00/01/02 agent report**](13_live_00_02_report_20261022.json). Trusted reported runtime results and exact GPU/database identities, kept separate from source verification and public approval.
- [**13 — Small LIVE-02 handoff and staging publication**](13_live02_small_handoff_and_publication.md). Metadata-only GPU report, source-hash-bound local review, compact Research Lab staging bundle; no huge ZIP transfers.
- [**13 — LIVE-03 measured intervention integration**](13_live03_review_and_gallery.md). PR #55/#56 merged; L5 F506 and L9 F95 no-op/ablation/scale with separate edited-token and final-token logits, optional approved CPU gallery and remaining private job work.

WP13 changes **operational readiness and deployment**, not the untouched WP12 P8 scientific release verdicts. Start with one complete real Prompt → Feature → Intervention workflow, publish a useful reviewed CPU Lab, then extend to native training and circuits.
