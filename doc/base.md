# Run access and base level

## Purpose

Describe how the project reads configuration, selects components and starts tasks.

## Scope

It is used to maintain orders, distribution, public training and reading and writing of results.

From[Run Entry](../base/application/cli.py)Read parameters, select components according to configuration. Public running support is in [`base/runtime/`](../base/runtime/README.md),Python Import Continue `saebase.runtime`.Press first when used[Parameter Description](conf.md)(Checks configuration files, default values and overwhelming rules, how the model's operating device is set) to confirm the current combination, and the result is from[Experimental report](experiment.md)(Available for comparison results and limits based on real operations BBH What resources are used for pre-capture) into the corresponding running directory.
