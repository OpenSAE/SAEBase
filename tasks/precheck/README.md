# Configurable Precheck

Precheck is the eighth registered research Task. The original standalone
activation-health command remains a legacy compatibility path. No particular
study is imported by this Task.

## Contract

Each `conf/model/<Model>.yaml` owns its `model.sae_profiles` and
`model.default_saes`. These are parameter lists, not an additional primary
component Registry. They reuse existing SAE identities during the migration.

Precheck supports `task.mode=sae`, `task.mode=task`, or
`task.mode=all`. It selects `task.sae_list=default`, `all`, or an explicit
list, and `task.check_tasks` may select any of the seven existing Tasks.
An empty Task list in task/all mode selects all seven.

The Setup phase stages dependencies/checkpoints and records hardware facts;
it does not run model forwards, SAE encoding, smoke tests, or research Task
operations. Real testing happens only inside Precheck workers. A cached or
downloaded checkpoint is not a PASS.

### Dry config check (no weights, network or GPU)

```bash
python -m saebase.runtime.train task=precheck --cfg job
python -m saebase.runtime.train task=precheck
```

### Explicit small real check (requires prior prepared/cached assets)

```bash
python -m saebase.runtime.train task=precheck task.execute=true \
  task.mode=sae task.sae_list=default task.device=cuda \
  task.allow_compute=true task.allow_gpu=true \
  task.auto_setup=true task.allow_network=false \
  task.max_samples=2 task.max_seconds=120
```

Network installation and checkpoint downloading are *separate*, opt-in
permissions: `task.allow_network`, `task.allow_install`,
`task.allow_download` and `task.max_download_bytes`. No nonzero network
budget is granted by default. Native Train/Circuits/Explain checks require
`task.allow_native=true` and their own bounded `task.task_configs`.
A Task that cannot meet its budget reports BLOCKED, not an invented PASS.

Results are written to the existing Runtime run leaf as
`metadata/precheck_result.json` and `metadata/precheck_audit.json`, with
isolated worker receipts in `analysis/precheck_jobs/`. BLOCKED reports name
the stage, reason, attempted actions, required fix and retry status.
Offline tests exercise config, negative cases and schema only, not scientific
acceptance. Workers enforce elapsed time; these checks do not measure model
quality or provide independent Harness approval.
