# Core: Project Architecture and Management

Core Tree maintains stable project-management IDs in [project_graph.json](project_graph.json). It describes architecture, infrastructure, experiment engineering, and project decisions separately from scientific Knowledge.

```bash
python harness/core/validate.py
python harness/core/select.py q.research s.explainability g.code_tree
```

Project decomposition and official status are governed on GitHub. Agents normally receive a bounded Work Order and may read only its authorized core IDs. Proposals require source-backed evidence; a proposal does not overwrite the authoritative Core state.

- [TREE_CONTRACTS.md](TREE_CONTRACTS.md) defines the distinct evidence contracts for Core, Knowledge, Code, and Environment.
- [BOUNDARIES.md](BOUNDARIES.md) describes module ownership and write restrictions.
- [GITHUB_ONLY.md](GITHUB_ONLY.md) concerns GitHub-level planning and is not part of normal agent context.
- [idea/](idea/) stores unimplemented proposals.

[Open Core Tree](https://atheavenma.github.io/opensae/openresearch/atlas/core.html).

## Agent execution governance

The review, scoped authorization, source inspection, experiment observation, and human decision process is described in [AGENT_FLOW.en.md](AGENT_FLOW.en.md). Core remains protected from ordinary task-agent writes.
