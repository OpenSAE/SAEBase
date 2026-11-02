# Context-grounded Dialogue

Use for a substantive SAEBase project question, not ordinary conversation.

1. Identify whether the request needs Knowledge/Context definitions, Code behavior, an observed Environment, or some combination.
2. Respect the active Work Order's required/optional reading permissions and existing Core shielding. Confirm exact-file read access; do not turn a user question into broad repo access.
3. Read the smallest relevant sources, using `harness.knowledge.dialogue_context.read_project_context(root, requests, allowed_paths)` when useful. Check each returned `status`, `source`, `source_sha256` and actual revision. A path appearing in an index is not evidence that its contents were read.
4. Explain using the existing Context terminology, distinguishing document fact, actual source implementation, current runtime observation, inference and unavailable information. A `SYNTHETIC` or stale Environment snapshot is not a current-machine measurement.
5. Carry observable IDs/paths and unresolved access in the existing Work Order contract receipt when applicable. Never claim that source access verifies arbitrary natural-language semantics.

Access is read-only; edits, downloads, native testing and publication retain separate authorization. Do not expand permission or bypass AGENTS.md.
