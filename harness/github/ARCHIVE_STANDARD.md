# Harness GitHub · Versioned archive contract

This is the permanent ownership and naming contract for
`harness/github/development_doc`, `work_plan` and `work_test`.
It governs new files and future archive operations only. Previously frozen
reports, exams, scores and evaluation evidence are **not rewritten or
revalidated against newer WP/WT writing rules**.

## Stable roots and historical folders

- Permanent developer guidance, release process and API entrypoints remain
  in `development_doc/` and retain their existing paths.
- `work_plan/WRITING_STANDARD.md`, `WORKPLAN_TEMPLATE.md`,
  `DELIVERABLE_STANDARD.md`, `lifecycle.py` and current Work Plan
  coordination remain stable, together with current `from_human/`.
- Current CI, Contract and release tests stay executable in `work_test/`.
  Old WT files may be retired **only when the latest CI and release workflow
  do not depend on their original path or test behavior**.
- Versioned histories live under each owner's `archive/<version>/`.
  No new top-level archive subsystem or duplicate scientific authority exists.

| Archive type | File or folder pattern | Example |
| --- | --- | --- |
| Dev | `<semver>_dev<NNN>_<word>_<word>.md/json` | `0.6.5_dev001_gpu_accept.md` |
| WP | `<semver>_wp<NNN>_<word>_<word>.md` | `0.6.5_wp001_qa_smoke.md` |
| WT | `<semver>_wt<NNN>_<word>_<word>/` | `0.6.5_wt001_qa_smoke/test_run.py` |

The two final words are short, lowercase ASCII words; numbers are 001–999.
A matching `.en.md` variant is permitted for translated source.
The `dev/wp/wt` sequence is independent **within each version** and
never renumbered after publication. Versions are sorted numerically as
semantic versions rather than lexicographically.

**Original WP IDs are not archive sequence IDs.** A historical WP20 may
be indexed as `0.6.5_wp001_exam_closeout.md` and retains its internal
`WP20` identifier, Contracts and frozen result claims.

## Evidence and safe migration

`archive_index.json` is the only file-location registry; it is not a new
Work Plan, Context or scientific status authority. Each archive entry carries
the previous path, new path, exact Git blob SHA, version and retention reason.

- `move`: original path removed after actual references are audited; the
  archival bytes must match the original Git blob exactly.
- `copy_compat`: preserve identical archival bytes while retaining the
  original compatibility path because active sources, release records or
  published links still depend on it. This **is not a completed root cleanup**:
  it is deliberately classified and may be promoted to a move only in a
  separately reviewed future change.
- Unknown version or missing original evidence is `UNRESOLVED`, not guessed.
  Historical documents with unresolved versions remain in their current home.

An archive check validates filenames, Git blob identity, stable index ownership
and the absence of accidentally collected retired WT tests. It does not
re-execute or reinterpret frozen historical experiments.

## Future release preparation

1. During development, prefix *new* documents and task folders with the
   intended version and category before starting the next release.
2. Before Release Validation, run the manual `Archive Preparation` workflow
   on current `main`. It produces a deterministic candidate manifest and
   read-only report as an Actions artifact; it **does not push to main or
   create a PR**.
3. Use Atlas Human Check to review the individual logical changes and all
   their linked Core, Code and Context parts. The human can independently
   Approve/Deny/Unchecked each change after browsing its related nodes.
4. Apply only approved, independent scopes, update the archive index, and
   merge the archive PR into `main`. Record `prepared_for_versions` for the
   next release and recheck the exact SHA.
5. Run Full Release Validation against that final source SHA. Release
   validation refuses an unprepared newer version. No archive automation
   writes files after the release's full validation.

The archived reports are reference material, not input to subsequent
strict WP/WT authoring standards. New/current documents and executable
tests continue to follow those standards.

## Approval record for the initial transition

The current initial migration was approved by the human in this conversation
for all six logical groups of `HC-Archive-20261101`. It is attributed
as **human chat authorization**, *not* as clicks in the localhost-authenticated
Atlas ledger. See `archive_approval.json`. No release, GPU, scientific
verification or future unspecific file edits are authorized by that record.
