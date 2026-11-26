"""Expose SAEBase tasks and compatibility import names."""

from pathlib import Path

from saebase.import_aliases import register_aliases as _register_aliases


# Look up task modules in the live source tree during a strict editable install.
# Other packages keep the link tree because their build combines code and resources.
_package = Path(__file__).absolute().parent
_image = _package.parent
if _image.name.startswith("__editable__.") and _image.parent.name == ".build":
    _source = _image.parent.parent / "tasks"
    if (_source / "__init__.py").is_file():
        __path__[:] = [str(_source)]

_register_aliases(
    {
        "tasks.benchmark.cli": "tasks.benchmark.commands",
        "tasks.benchmark.methods.rss": "tasks.benchmark.methods.stable_support",
        "tasks.benchmark.methods.saebench": "tasks.benchmark.methods.sae_benchmark",
        "tasks.benchmark.methods.sar": "tasks.benchmark.methods.response_surface",
        "tasks.benchmark.reliability.cli": "tasks.benchmark.reliability.commands",
        "tasks.capture.sar": "tasks.capture.response_surface",
        "tasks.intervene.cli": "tasks.intervene.commands",
        "tasks.sae_runtime": "saebase.runtime.sae_operations",
    }
)
