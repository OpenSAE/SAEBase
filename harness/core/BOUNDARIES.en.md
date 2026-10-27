# Core: Implementation Boundaries

**Core** defines project architecture and management goals in `harness/core/project_graph.json`. It is separate from scientific theory or experimental evidence.

**Knowledge** records research questions, methods, and source evidence in `doc/knowledge/`; implemented code cannot itself establish a scientific result.

**Code** describes real Python entities extracted from the AST. Generated code nodes reference Core design and scientific Context by stable identifiers and must not overwrite source code.

**Environment** relies on measurements made on the execution machine. Synthetic examples remain clearly labeled.

**Experiments** are governed by `saebase/runtime/experiment_paths.py` and Hydra configuration in `conf/`. Harness interprets and validates output but must not become a training-loop dependency.

**Transfer** requires explicit source, destination, and authorization; machine credentials must not be committed.

**Visualization** provides read-only public views of the Atlas trees, with separately authorized local review functions. Publishing does not reveal private machine state.

For each work order, select only the relevant source nodes and paths. Compare expected Code Tree changes with the AST after implementation. Verify the resulting model, checkpoint, configuration, and experiment independently before marking a scientific goal complete.

Static views are not evidence that real training succeeded. Reused validation/test data does not constitute an independent test. Browser-level or Cursor-level restrictions are not an operating-system security boundary.
