# Scientific Context expression norm

## Purpose

Describe how scientific issues, methods, objectives, terminology and their sources can be clearly expressed.

## Scope

It is used to maintain the text, terminology and true realization of the scientific maps.

## Unified semantics

Core describes the project structure. Knowledge describes specific scientific objectives. Glossary defines terminology. Code Tree provides actual Python entities. Each knowledge uses stable ID and traceable sources.

## Period and Information Density

- ** Positive**: Write down the actual functioning, input, output and evidence of the component.
- **Question**：`question：how_to_achieve <specific_research_goal>？`
- **Solve**：`approach：use <existing_method> complete <specific_steps>。`
- **Goal**：`goal：obtain <verifiable_result>。`
- **Terminology**:`Definitions:<Precise definition>.`
- **Update**:`Status:<The facts.>.Evidence:<Source Code or Run Record>.Change:<Specific results>.`
- The card uses a default explanation. Key details are presented by the inner extension of the node.

## Corresponding entity code

The concept and method details are used to view what is actually needed. The source uses the repository's relative path and code line numbers.

## Formulas and Icons

The formula indicates the scope of application. The diagram indicates the specific calculation relationship. The code extracts directly from the real source code. The source path points to the actual document. The scientific indicator is from the replicable experimental record.

## Status & Level

Open scientific tree collection is already achieved. The state of each node is determined independently by the evidence of that node.; partial Do not transmit along the parent chain. Nodal order follows reading order from top to bottom. All terms are used `term_ids` References [glossary.json](../code/atlas_site/source/knowledge/glossary.json).Stable scientific goals and responses ID Create a connection.

## Checking data by mission

The user requires knowledge or Atlas check to locate ResearchHarness through `.researchharness/central.json`. `src/harness/knowledge/knowledge.md` describes the knowledge structure and terminology maintenance and applies to the relevant input changes; `src/harness/core/visualization/visualization.md` diagram generation and presentation, applicable to the Atlas process.

When a problem is detected, a specific node, basis and handling action is located, an amendment is completed within this mandate to preserve the actual meaning of the scientific assertions.
