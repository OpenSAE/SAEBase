# WP12 GPU-00 — Agent Report and Independent Review Queue

Report received: 2026-10-21 (America/Chicago); reported run ID carries 2026-10-22 UTC.
Status: **FOUR SITE-LEVEL AGENT-REPORTED PASSES; NOT INDEPENDENTLY VERIFIED. P0 NOT CLOSED.**
Report provenance: user's pasted GPU-agent response, **not** a fetched/validated artifact bundle.
Run source: `feat/wp12-p0-foundation` at `490b18c0594e709e7dfb2841f7ec0b8e7bc9ec37`.
Agent machine run directory: `/workspace/wp12_gpu00_runs/gpu00-20261022T031830Z-490b18c/`.
Agent-reported summary: `summary.md` inside that run directory.

This document records an **unreviewed claim**, not accepted experimental evidence. The external GPU machine's `/workspace` directory is not mounted in the independent reviewer environment; the reported `summary.md`, `parity.json`, raw arrays, harness, failure records and checksums were **not available for inspection at the time of this update**. Do not confuse a filesystem path supplied in conversation with a publicly accessible artifact URL.

## 1. Reported numerical outcome

| Site | Real checkpoint route | GPU agent's result | Independent reviewer |
| --- | --- | --- | --- |
| A1 | `pythia70m` / `pythia70m_resid_sm` / L3 / `blocks.3.hook_resid_post` | PASS, legacy fp32 | PENDING RAW ARTIFACTS |
| A2 | `pythia70m` / `pythia70m_resid_sm` / L5 / `blocks.5.hook_resid_post` | PASS, legacy fp32 | PENDING RAW ARTIFACTS |
| B1 | `gemma270m` / `gemma270m_scope_resid` / L5 / `blocks.5.hook_resid_post` | PASS, legacy fp32 | PENDING RAW ARTIFACTS |
| B2 | `gemma270m` / `gemma270m_scope_resid` / L9 / `blocks.9.hook_resid_post` | PASS, legacy fp32 | PENDING RAW ARTIFACTS |

Reported run conditions and counts:

- Legacy TransformerLens **3.9** and SAELens **6.54** using `fp32`. TransformerLens **4 not installed or tested**; no TL4 compatibility claim.
- CPython **3.12.3** because this host does not have 3.11; Python 3.12 is allowed by SAEBase's declared `>=3.11,<3.13` support.
- Agent reports **85 CPU preflight tests passed**. Raw test log was not made available to reviewer.
- Eight frozen clean/corrupt text pairs (**16 variants per site**) from `12_gpu00_prompt_pairs.json`. Agent reports equal clean/corrupt token lengths, matched CPU/GPU token IDs and full model/SAE SHA256 matching on CPU and GPU. Exact token/hashes still need independent inspection.
- Native versus SAEBase activation, SAE feature codes, reconstruction, clean and corrupt logits, clean-to-corrupt patched logits, and residual-preserving ablation of the highest nonzero feature have **reported maximum absolute difference 0** in the run's stated fp32 protocol. No tolerance relaxation reported. Arrays not independently read.
- Matrix used **283 forward-equivalent calls**, below the predeclared 320 cap. Reported peak VRAM: **Pythia ~0.56 GiB**, **Gemma ~2.71 GiB**. Elapsed time, disk/RAM peak and raw resource trace have **not** been provided to reviewer.
- Separate legacy M0 SimpleCopying smoke was reported as `passed`, identity max error 0. Its eight M0 examples are distinct from the eight GPU-00 clean/corrupt pairs and **must not be counted in this matrix**.

## 2. Failures and negative controls remain visible

1. The first A1 harness attempt failed due to a hook argument-name error. Agent reports preserving the original traceback in `failures.json`. A corrected rerun passed. Reviewer must inspect both attempts and the harness diff to confirm only the harness call changed and that the failing attempt did not silently disappear.
2. A probe with a nonexistent TransformerLens `names_filter` reportedly returned without raising an error. The agent correctly **did not count that test as passing**. This alone does **not** establish SAEBase's API silently accepts an invalid hook: in the audited source, `saebase/api/model.py:ModelAdapter.capture` and `saebase/store/collect.py:capture_selected` explicitly check `site.hook_name in cache` and raise if absent. Test the **actual SAEBase entrypoints**, not the raw upstream `run_with_cache(names_filter=...) != raise` assumption.
3. Agent reports that a wrong layer hook, stale model state SHA and wrong SAE SHA were rejected. Original negative-control records, expected error names and per-case pass flags still require inspection. Test nonexistent in-range hook against **SAEBase** capture and patch routes, including whether the replacement hook actually fired, rather than trusting the native filter call.
4. The record contains **no** evidence for TransformerLens 4, independent multi-host raw evidence verification, new hosted GPU service, circuits, training or distributed execution.

## 3. Independent review is the next required task, not GPU-01

**Evidence transfer:** before acceptance, the agent must produce a rights-reviewed, immutable bundle of the **existing** GPU-00 run, not a fresh 48-site experiment. Ask for:
- `summary.md`, `manifest.json`, `preflight.json`, `environment.txt`, `requirements-resolved.txt`, `cpu_preflight.log`, `tokenization.json`, `harness.py`, `failures.json`, `parity.json`, `cache_identity.json`, `resource_usage.json`, `checksums.sha256`, plus separate **unrounded native and SAEBase arrays** and any first-attempt harness snapshot/diff.
- Recompute an **outer archive SHA256** and record the source commit and exact native-package revisions. Preserve internal checksums; do not recompute them as though they were original.
- Review for secrets, HF tokens, private prompts, inaccessible licensed data, private cache paths containing credentials, copyrighted model weights and third-party source redistribution restrictions. The eight authored engineering text pairs may be shared; third-party checkpoint binaries should **not** be included.
- Deliver a private authorized evidence archive to the reviewer or an access-controlled durable evidence reference. A local `/workspace` path alone is **not** evidence transfer; do not push raw weights/prompts to the public Git repository.

**Independent CPU verification:**

1. Recalculate all file checksums and archive checksum from supplied bytes; inspect original and corrected harness versions, source commit and all CPU/GPU package, model, SAE, tokenizer and prompt hashes.
2. Confirm 4 exact sites × 8 groups × 2 variants without omissions or replacements; verify last-valid-token positions, original token IDs, native vs OpenSAE full tensor shapes and dtype.
3. **Independently compute** per-sample elementwise differences from preserved native versus SAEBase numeric arrays, including full-vocabulary logits rather than recorded rounded probes. Do not accept only `parity.json` flags or agent-generated max-abs summaries.
4. Check that both code paths actually executed the intended hooks and operations. A zero difference between paths is not sufficient if both paths mistakenly bypassed a patch. Verify native and SAEBase controlled patch effects relative to their clean/corrupt baselines; verify selected nonzero features, residual-preserving intervention algebra and per-site actual readout/activation changes.
5. Confirm cache positive reuse causes no unnecessary forward and mismatched model/SAE/token identity fails closed. Specifically validate nonexistent in-range hook against SAEBase capture and patch operations. A raw upstream `names_filter` returning an empty cache without an exception is **not** equivalent to SAEBase acceptance of a bad site.
6. Check the logged 283 matrix forwards against the actual call ledger and 320 cap; check elapsed wall clock ≤7200s, new scratch ≤20 GiB and free-disk reserve ≥20% at submission. Preserve all negative/failed attempts.
7. Only after steps 1–6, record an independent reviewer verdict per tested **combination and backend**. The version scope is **legacy TL3.9 + SAELens6.54 fp32**. Never mark TL4/new-driver parity or full-feature verification from this smoke alone.

## 4. Ledger and roadmap implications

- Relevant capability IDs: F06, F07, F08, F09, F11, F14; F41 and F42 are **narrow Gemma checkpoint/site smoke only**, not full Gemma Scope 2 asset catalogue or multiple hook families.
- Add this report as `GPU_AGENT_REPORTED_PASS_PENDING_REVIEW` in the ledger's `gpu_evidence` field, while keeping **all existing PARTIAL statuses**, leaving independent `native_parity_evidence`, `artifact_link` and `reviewer` empty.
- P0 CPU environment coverage remains incomplete (this host's 85 preflight tests are helpful but do not prove all P0 clean-install/browser/data-rights gates); independently deployed HF Lab remains unverified.
- Next CPU step: inspect evidence, fix any real API-level invalid-hook rejection issue with regression tests if found, lock supported legacy environment and design a **separate** TransformerLens 4 driver/bridge test. Do **not** automatically launch GPU-01, C2/C3 or set `P0=PASS`.

**Decision as of this report:** strong provisional evidence for existing legacy-path correctness; independent GPU-00 numerical acceptance **OPEN**, comprehensive P0 **OPEN**.
