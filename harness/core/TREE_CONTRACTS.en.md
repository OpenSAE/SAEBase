# Contracts for the Five Atlas Trees

Core, Knowledge, Code, Environment, and Context have distinct authoritative sources and editing rules.

| Tree | Main question | Authoritative source |
| --- | --- | --- |
| Core | Why is the research system organized this way? | `harness/core/project_graph.json` |
| Knowledge | Which scientific questions, methods, and validation goals are recorded? | `doc/knowledge/context_graph.json` |
| Code | Which Python modules, classes, functions, and methods actually exist? | Python AST and `doc/code/code_tree.json` |
| Environment | Which resources are available for executing the work? | Hardware probes and environment snapshots |
| Context | What do technical terms mean, and what supports their definitions? | `doc/knowledge/glossary.json` and `doc/knowledge/atlas_terms.json` |

## Core and Knowledge are distinct

Core uses stable project-management IDs; Knowledge uses stable scientific IDs. The two graphs cannot substitute for one another. Their authoritative hierarchical relations are source-backed; display-only group nodes do not create new scientific hypotheses.

A node's status depends on its own implementation and evidence. A partially implemented child does not automatically mark its parent partial. Display order follows authoritative edges.

## Shared terminology

Knowledge nodes reference terminology through `term_ids`. Each glossary definition has a unique ID, aliases, evidence, and references. Links in both directions are validated. Technical and scientific terms should not be silently rewritten by generated prose.

## Code and source evidence

Code Tree nodes are extracted from real source. Annotations may reference both management goals and scientific questions. Planned changes and observed AST changes must be reported separately. A source file path, callable signature, or module name is never invented to satisfy display structure.

## Environment and reproducibility

Machine facts must be measured. Unknown measurements remain unknown. The public example environment is synthetic. Model execution and independent experimental acceptance require separate evidence.

## Clear status descriptions

A node describes its question, method, goal, evidence, observed state, and next action as applicable. The displayed hierarchy may group nodes for readability, but this does not change source-level dependencies.
