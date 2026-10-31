"""Build SAEBase without shadowing third-party Hugging Face datasets.

Physical dataset implementations remain in the repository-level datasets/
tree; installed distributions expose them as saebase.datasets.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Copy source-owned dataset modules into the installed SAEBase namespace."""

    def run(self):
        super().run()
        source = Path(__file__).resolve().parent / "datasets"
        destination = Path(self.build_lib) / "saebase" / "datasets"
        for module in sorted(source.rglob("*.py")):
            output = destination / module.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(module), str(output))


setup(cmdclass={"build_py": build_py})
