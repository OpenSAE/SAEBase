# Human Check · Pre-PR Atlas review

Human Check is the default pre-implementation review for conversational GitHub
maintenance. It does **not** create a PR, edit source code, execute a model, or
approve scientific findings. A typed proposal and the written report use the
same immutable facts. Review only changes a local ignored ledger.

## Mandatory Atlas UI step

Every Human Check MUST deliver a directly browsable Atlas review UI, not only
a JSON proposal, report, ZIP or shell script. Use the existing Atlas review,
not a separate question form.

**Default for conversational GitHub reviews:** publish the source-backed
read-only proposal snapshot under
`harness/visualization/review-proposals/<HC-ID>.json`, including its real
Atlas node IDs, groups, Before/After, risk, checks and explicit context.
The source Pages fallback reads this directory directly; the Actions Pages
bundle must also ship these snapshots beside the five tree pages. Confirm
that the actual SAEBase Pages `/atlas/core.html?review=<HC-ID>` link loads the
proposal before giving the user that URL. Do not substitute a custom HTML
preview, local ZIP or script for that link. The existing Atlas visual design,
four review arrows and grouped reading order remain authoritative.

**Authenticated decisions:** public GitHub Pages permits browser-only draft
decisions; it is not a server-side approval receipt. When formal saved
decisions are required in an accessible local checkout, the one-command
authenticated entry is:

~~~bash
python -m harness.github.human_check quickstart --manifest /path/to/proposal.json
~~~

This command freezes or resumes the proposal, verifies the actual local Atlas
HTML/JS/CSS/graph/review endpoints, prints the authenticated localhost URL and
tries to open it in a desktop browser. It does not record approval. A failed
browser launch reports the verified URL instead. A remote/headless environment
may still require port forwarding.

If no executable checkout or accessible UI is available, do not claim that
Human Check is ready. Provide a clearly labeled browsable preview if possible,
explain that it cannot save formal authenticated decisions, and report the
blocker. The legacy prepare/report/serve/check commands remain available.

## 1. Author one proposal before editing

Start with the current GitHub source and the smallest relevant Core, Knowledge,
Code, Environment and Context identities. A proposed new Python module is a
proposal, not an observed AST. Use existing Context term IDs rather than
inventing definitions. Record file owners and separate changes that can be
approved independently.

Save the following example as a JSON file **outside tracked source**, adapting
the node IDs and file paths to real existing source before use:

~~~json
{
  "id": "HC-Example-001",
  "goal": "Validate the proposed registry and task changes independently",
  "items": [
    {
      "view": "core",
      "node_id": "g.registry",
      "operation": "change",
      "files": ["saebase/registry.py"],
      "after": {
        "title": "Keep a shared Registry",
        "description": "Retain a single authoritative component lookup."
      },
      "reason": "Preserve the existing Core module boundary.",
      "risk": "Backward compatibility for callers",
      "checks": ["Registry tests", "CI"],
      "context_term_ids": ["term.registry"]
    }
  ]
}
~~~

An actual plan should enumerate each separately reviewable change. Supported
views: core / knowledge / code / environment / context. Supported operations:
add / change / remove. An existing node must resolve in the selected Atlas
authority; a new Code node remains explicitly proposed until implementation.
No change of Atlas graph or glossary occurs during this step.

From a clean SAEBase checkout:

~~~bash
python -m harness.github.human_check prepare --manifest /tmp/proposal.json
python -m harness.github.human_check report HC-Example-001
python -m harness.github.human_check serve HC-Example-001 --port 8766
~~~

The first command writes the ignored
\`experiment/_human_check/HC-Example-001/review.json\` and \`report.md\`
and freezes the source identity and per-file original hashes. Review the
written report **before** starting the local server. The third command prints
an authenticated localhost URL into the current terminal. Open that URL
yourself; a conversational assistant cannot silently open a local browser
or grant an approval on your behalf.

## 2. Review *each* Atlas item

Use the existing five-tree Atlas navigation and Approve / Deny / Unchecked
controls. Decisions are saved synchronously by the authenticated loopback
service and retained individually. There is **no Approve Plan action**.
Approval never adds a Knowledge node or changes the authority Code tree.
An item with a changed meaning or scope requires a new proposal.

If only one item is denied, unrelated approved items remain eligible. If
an approved and a denied/unchecked item both claim the same file, *that file*
is blocked until the human narrows or revises the proposal. This prevents
path-granularity permission from silently executing rejected changes.

Human Check starts **ON**. The Atlas review bar has an explicit **Human Check
ON · Disable** control; clicking it is a human opt-out that persists actor
and time in the ledger. Re-enabling restores the default approval requirement.
Opt-out bypasses only Human Check: it never grants rights to edit protected
governance, run paid/GPU/native tasks, publish, or waive existing Contracts.

Public GitHub Pages review is *not* an authenticated approval endpoint.
Only the local loopback service can save formal decisions.

## 3. Verify the exact scope before editing

Before the first GitHub source edit or PR, run:

~~~bash
BASE_SHA=$(git rev-parse HEAD)
python -m harness.github.human_check check HC-Example-001 \
  --base-sha "$BASE_SHA" --path saebase/registry.py
~~~

This fails on unapproved, denied or unchecked paths; mixed-decision file
ownership; source/hash drift; and a source SHA different from the proposal
baseline. Every independent implementation unit checks **only its paths**,
so unrelated denied changes cannot block other approved tasks.

After implementation, inspect the actual diff and use
\`--after-edit\` with the same approved baseline SHA and each observed
changed file. This is a scope check; AST behavior, tests and scientific
results still require their original independent verification. If the scope
changed materially, stop, create a revised proposal and review affected
items again.

Keep the eventual PR description linked to the reviewed proposal, baseline
commit, approved node IDs and separate test evidence. Local session keys,
machine credentials and confidential absolute paths never enter Git.
A text/Atlas proposal is not itself a code-write security sandbox; actual
GitHub protections, agent permissions and CI remain necessary.

## 4. Bootstrap and limitations

This feature itself was authorized in the conversation before its first
implementation branch. Existing Work Orders, Contracts, technical CI and
post-implementation Atlas review remain separate. The historical
\`harness.core.agent_flow\` blank-Knowledge approval path is unchanged:
Human Check uses a distinct proposal-only record and does not adopt blank
Knowledge nodes.

To use the feature in ChatGPT, the assistant should provide the proposed
report in conversation for human inspection before any GitHub write and
offer the local Atlas review URL **only when a real service is available**.
The implementation cannot programmatically open the ChatGPT application's
native UI or guarantee that uncooperative direct GitHub API clients invoke
the local gate.


## Multi-part Atlas review (linked local changes)

A logical GitHub change may affect multiple separate Atlas nodes. To review
it **as one independently authorizable change**, give each member the same
`group_id` and `group_title` in the manifest:

~~~json
{
  "id": "HC-Multi-001",
  "goal": "Review one repository change through linked Core and Code perspectives",
  "items": [
    {
      "view": "core", "node_id": "g.context", "operation": "change",
      "files": ["harness/github/ARCHIVE_STANDARD.md"],
      "after": {"title": "Document archive naming", "description": "Specify stable versioned names."},
      "reason": "Preserve existing architecture", "context_term_ids": ["term.core_tree"],
      "group_id": "HC01", "group_title": "Archive naming"
    },
    {
      "view": "code", "node_id": "code:module:harness.github.archive", "operation": "add",
      "files": ["harness/github/archive.py"],
      "after": {"title": "Archive path selector",
                "description": "Prepare a bounded deterministic archive plan.",
                "path": "harness/github/archive.py"},
      "reason": "Implement reviewed naming rules",
      "context_term_ids": ["term.python"],
      "group_id": "HC01", "group_title": "Archive naming"
    }
  ]
}
~~~

The **existing Atlas Tree**, not a separate question form, displays each
proposal at its actual or explicitly proposed node. The group selector above
the existing review bar moves between independent logical changes; **Next
linked part** moves among the related tree nodes. The local service records
each visited part with the plan revision. Approve and Deny remain disabled
until all of the group's parts have been visited. The selected decision then
applies atomically to that group's node items only; no global Approve Plan.
Unchecked resets the group's decision. Unrelated approved groups may proceed.

The original single-node manifest remains supported for legacy plans.
The source version, declared paths, source and authority checksums, and human
authentication still apply. A grouped approval is not release, scientific
validation or code implementation. Do not create speculative Code AST nodes
as if they already existed: mark them as Proposed.
