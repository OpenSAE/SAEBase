# Public statement of realization

## Purpose

Description of the duties of SAEBase public interface, registration and operation support.

## Scope

Read and develop as applicable to `saebase/`. Maintenance is carried out in accordance with the mandate of project root AGENTS.md.

- `registry.py` Registration of the management component, `runner.py` for uniform distribution; access points are shared.
- `extensions` for external access, `api/` to provide common models and SAE operations, `runtime/` to maintain co-configuration, running paths and records.
- `datasets/` to maintain compatible import of top-level data sets.
- Common capabilities are built on existing call needs, and method-specific algorithms and indicators are kept in specific scientific missions.
- When developing public interfaces and generic checks, return information from the work that the actual caller has to do next. Retain the results interpretation, source and recommendations from the specific module, using the same meaning as the unified entry; repeat information is obtained from existing facts. Return elements that need to be saved are carried by existing run records, and interface checks are used to read and use the results through the actual caller.
