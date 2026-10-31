# SAEBase Harness · WP21 stage closeout (2026-10-30)

The WP21.1 exam infrastructure and WP21.2 Harness maintenance phase are finalized. Future work should start from the consolidated Harness on \`main\`. Existing WP18, WP19 and WP20 test trees, reference verifiers and Contract catalog are preserved unchanged.

## Exam record

- **Round01:** 17 PASS, R08 ENV_BLOCKED, R17 FAIL and R20 INCOMPLETE.
- **Round02:** a new independent R08 review passed all five positive and four negative checks (100/100). The R17 and R20 evidence audits now accept their real directory artifacts. R17's same-seed training weights still exceed the frozen tolerance; no new R20 final PASS has been issued.
- This yields 18 of 20 distinct cases with a recorded independent PASS across the two rounds, not a single all-pass examination.

## Scope and compatibility

WP21.2 fixes the nonempty-directory evidence check, resource inventory, runtime import provenance, negative-command recording, reviewer provenance archives, multi-cause diagnostics and tensor comparison. The 68 frozen suite inputs and the 22 predeclared Contracts are unchanged. Git tree identity checks preserved the exact WP18, WP19 and WP20 subtrees, and the prior Core, Code, Knowledge, Environment and Visualization trees.

The local finalization run covered 233 passing selected assertions; 28 long-running or workspace-copy cases were not all rerun in this pass. The earlier WP21.2 engineering validation reported 108/108 in separate batches. These are different validation statements. Remote CI and genuine model grading remain separate.

Open product-level issues include R17 training reproducibility and Benchmark pad-token/device forwarding. Five historical reviewer worktrees showed script-hash mismatches, with supplementary "b" trees archived locally. R08 probe dtype must be distinguished from the candidate Factory dtype. Do not retroactively rewrite any frozen verdict, relax tolerances, merge candidate product patches or present the archived evidence as a scientific release.

See the [Chinese closeout](21_harness_phase_closeout_20261030.md), [WP21 tests](../tests/wp21/README.md), [WP21.2 maintenance](../tests/wp21/WP21_2_HARDENING.md), [round01 report](21_exam_round01_report_20261030.md), and [round02 follow-up](21_wp21_2_followup_report_20261030.md).
