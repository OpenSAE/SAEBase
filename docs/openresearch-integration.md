# OpenResearch and SAEBase

SAEBase integrates the **OpenResearch** research-organization approach and the public agent Harness. Core, Knowledge, Code and Environment structures provide traceable context for agent-assisted research development. No private repository is referenced or required.

The SAE-only component layout is `research_datasets/<name>/`, `models/<model>/`, and `tasks/<task>/`. The `saebase/` API, original CLI and public model registry retain their existing interfaces. A separate SAE-specific research registry at `saebase/research_registry.py` supports the integrated task implementations.

The structural gate is `python .cursor/hooks/validate_repo.py --full`. Model execution needs actual runtime and checkpoint verification.
