# SAEBase and External Engineering Management

## Purpose

Describe the actual division of duties between the scientific components, co-operating support and ResearchHarness and connect to the current development portal.

## Scope

This is used to understand the division of labour between SAEBase and ResearchHarness. Component maintenance is carried out according to the current mandate.

## Current maintenance entrance

[Current structure](current-architecture.md)Centralizes the maintenance of component duties, public operations and results interfaces. Task Realization of reservations, re-use of public capacity by actual callers.

SAEBase operates independently of the project by its own dependencies, configurations, and commands. ResearchHarness provides options for sharing skills, binding of in-house resources and registration of structural results. Code review, formal planning, and collaborative authorization to use the project GitHub process. Change Runtme first to locate the common support interface and direct caller and check the scope for selection on the basis of actual impact.

## Explain generator

`python -m saebase.runtime.train task=explain` cannot put a Python callable in YAML. With no generator it refuses to invent text. Two supported entries:

- `execute(cfg, generator=callable)` from `tasks.explain.operation`.
- `task.explain.generator=saebase_generate` with `task.explain.approved=true` and `task.explain.generator_model` set to a cached model id or directory. That name calls `tasks.generate.serving.generate` (vLLM). Gemma-2 needs `generator_dtype=bfloat16`. Any other string is rejected and is not evaluated. The main environment on this machine does not have a complete vLLM install; use an isolated environment that already imports vLLM.

`heldout_score` measures whether a supplied predictor agrees with the SAE feature being active. It is not a semantic accuracy for the generated sentence.
