# WP14 GPU Receipt Review — Product Decision

Date: 2026-10-22  
Original execution source: `5aa054dd491d7c1c63204aa2ee7e5292cdd5ac38`  
GPU: RTX 5070  
Receipt: `/workspace/wp14/release_gpu_receipt.json` on the GPU Agent host, **not** a published repository artifact.

## Verdict

**Approve the current GPU work for release integration. Not yet approve a public release.** This is a product usability decision based on the Agent's summarized measurements, not direct access to its private receipt/raw logits. There is no need to rerun GPU-00, LIVE-00–03 or finish all 50 WP12 feature rows.

## Feature triage

| Task | Observation reported by GPU Agent | Product decision |
| --- | --- | --- |
| G1 | Fresh Gemma-3-270M L5 / `gemma270m_scope_resid` token capture, 13 tokens and 972 nonzero codes; F1627 active at token 5; baseline and no-op logits identical; ablation changes teacher-forced argmax 1607 → 7549; measured 10.835 s and 2,907,527,680 B peak VRAM | **ACCEPT for local-live capture and teacher-forced SAE edit**. Integrate a reproducible Python/CLI example. No general semantic claim. |
| G1 private queue | Two separate owner-prompt **inspect** jobs succeeded; bad bearer token returned HTTP 401; loopback FastAPI/Uvicorn available | **ACCEPT inspect-only private demonstration**; **do not** call queue-based edits, deployment, cross-user security or public inference proven. |
| G2 | Real 4-token greedy run; baseline/no-op/ablation decoded continuations all ` The compass is a` | **HIDE any claimed text-changing steering demo.** The generation protocol ran and output equality is a measured result, not evidence the feature editing API is broken. Optional single targeted search for a visible example is permissible but must not block release. |
| G3 | Original SAEBench upstream checkout and installation absent, no native scores computed | **DEFER live native benchmark**, not “failed implementation”. Keep reliable existing recorded metrics. No broad dependency-install work is required to ship the core preview. |
| G4 | Original circuit-tracer package missing; no native graph produced | **HIDE new circuit computation**. Any older authentic graphs may be listed as recorded only after actual browser and rights checks. |
| G5 | Sparsify importable; no compatible frozen local model/dataset snapshots, no training run | **DEFER native training**. Do not call it impossible, do not list it as demonstrated. |
| G6 | Local instruct directory contained only ~20 KB stub code and no model weights; no real interpretation | **HIDE automatic interpretations** and avoid fabricated feature names. |

These statuses apply to the **initial public preview**. Preserve working source code and documented experimental APIs, but trim dead navigation and claims. The absence of a prerequisite is not proof of a broken algorithm.

## Highest-value remaining work is CPU/public UX, not more GPU auditing

1. **One genuinely populated public feature journey:** source-approved authored prompt, original existing LIVE-02 L5/L9 real trace, token hover/click, active feature inspection, back navigation and concise explanation of actual recorded numbers. One excellent example is preferable to ten empty ones. Verify that the specific source text is authorized before publishing.
2. **One intervention journey:** stage the existing LIVE-03 L5 F506 and L9 F95 small source-linked metrics through the actual adapter; show token-position L2/argmax versus final-position results, no-op control and explanatory labels. Keep these separate from G1 F1627 fresh-input example unless the latter has its own reviewed display bundle. Do not mix runs or imply a free-generation result.
3. **One local-live quick start:** reproduce the actual supported model/SAE/site and one fresh prompt via CLI/Python; show measured user output and the real configuration. The private queue can remain operator-only and need not ship as a public service.
4. **Product pruning and polish:** remove or visibly mark as unavailable Live Generation, New Circuit, Auto Interpretation, Native Training and New Benchmark Run actions. Keep valid historical results and source links. Prefer fewer functional pages to many blank placeholders.
5. **Staging and release check:** build the real read-only CPU Docker image with populated records; open a fresh browser and click Prompt → Token → Feature → Intervention; verify it still works after restart; test install/CLI and local API; verify public API has no private GPU route or secrets; store production HF Space revision for rollback. Merge PR chain #50–54 only after integration validation.

### Very small optional follow-up (not a release blocker)

If a visible generated-text demo would significantly improve the launch, ask GPU Agent for **one** further bounded steering attempt: choose 1–2 genuinely active features with reported edit effect, try one or two prompts and 16–32 generated tokens with the existing working incremental protocol; record baseline/no-op/edited token IDs and logits. If no visible difference, stop and omit the claim. Do not treat unchanged greedy tokens as a computational failure or start a tuning campaign.

Native SAEBench is the next possible post-preview upgrade if the exact pinned upstream package can be obtained in a separate environment. Circuit tracer, training and interpretation follow only when usable compatible assets and a practical benefit are evident.

## Release recommendation

**Current status: GPU_RECEIPT_ACCEPTED; PUBLIC_PREVIEW_NOT_YET_READY.**  
**GO_WITH_CUT** once real Prompt → Token → Feature → measured Intervention is browsable in deployed staging, supported local live quick start works, data rights are cleared for the included examples, dead menu entries are removed, and the deployment survives restart with a known rollback revision.  
**NO_GO** for broken core journeys, empty public demo, misleading recorded-vs-live claims, or exposed private data. Optional G2–G6 results never prevent the release.
