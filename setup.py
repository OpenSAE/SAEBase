"""Map physical /datasets sources into the saebase.datasets Python namespace.
This avoids shadowing the third-party Hugging Face datasets package.
"""
from pathlib import Path
from setuptools import find_packages, setup
datasets = Path(__file__).resolve().parent / "datasets"
packages = set(find_packages(exclude=("tests*",)))
packages.update({"saebase.datasets"})
for init in datasets.rglob("__init__.py"):
    parent = init.parent.relative_to(datasets)
    if parent.parts:
        packages.add("saebase.datasets." + ".".join(parent.parts))
setup(packages=sorted(packages), package_dir={"saebase.datasets": "datasets"})
