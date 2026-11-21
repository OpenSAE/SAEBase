"""SAEBase task registration.  Phase A bundles the SAE activation-health task."""

from saebase.import_aliases import register_aliases as _register_aliases

_register_aliases({
    "tasks.benchmark.cli": "tasks.benchmark.commands",
    "tasks.benchmark.methods.rss": "tasks.benchmark.methods.stable_support",
    "tasks.benchmark.methods.saebench": "tasks.benchmark.methods.sae_benchmark",
    "tasks.benchmark.methods.sar": "tasks.benchmark.methods.response_surface",
    "tasks.benchmark.reliability.cli": "tasks.benchmark.reliability.commands",
    "tasks.capture.sar": "tasks.capture.response_surface",
    "tasks.intervene.cli": "tasks.intervene.commands",
    "tasks.sae_runtime": "saebase.runtime.sae_operations",
})
