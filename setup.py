"""Build SAEBase with stable Python package names and physical component directories."""

from pathlib import Path
import shutil

from setuptools import find_namespace_packages, setup
from setuptools.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.editable_wheel import _LinkTree, _empty_dir, editable_wheel as _editable_wheel
from setuptools.command.egg_info import egg_info as _egg_info, manifest_maker


SOURCE_MANIFEST = "doc/code/distribution/source/manifest.in"
PACKAGE_DIRS = {
    "saebase": "base/application",
    "saebase.api": "base/interfaces",
    "saebase.datasets": "base/dataset_access",
    "saebase.extensions": "base/extensions",
    "saebase.research_lab": "base/research_lab",
    "saebase.research_lab.backend": "base/research_lab/service",
    "saebase.runtime": "base/runtime",
    "saebase.setup": "base/installation",
    "saebase.store": "base/store",
    "saebase._vendor": "third_party/accelerator_kernels",
    "conf": "base/configuration",
}


class build(_build):
    """Keep generated build trees with local packaging metadata."""

    def initialize_options(self):
        super().initialize_options()
        self.build_base = ".build/package"


class egg_info(_egg_info):
    """Read the source distribution manifest from its documented build directory."""

    def finalize_options(self):
        if self.egg_base is None:
            Path(".build").mkdir(exist_ok=True)
            self.egg_base = ".build"
        super().finalize_options()

    def find_sources(self):
        maker = manifest_maker(self.distribution)
        maker.ignore_egg_info_dir = self.ignore_egg_info_in_manifest
        maker.manifest = str(Path(self.egg_info) / "SOURCES.txt")
        maker.template = SOURCE_MANIFEST
        maker.run()
        self.filelist = maker.filelist


class editable_wheel(_editable_wheel):
    """Use a regular package tree for the mapped modules in editable installs."""

    def finalize_options(self):
        super().finalize_options()
        if self.mode is None:
            self.mode = "strict"

    def _select_strategy(self, name, tag, build_lib):
        if self.mode.lower() != "strict":
            return super()._select_strategy(name, tag, build_lib)
        link_tree = _empty_dir(Path(self.project_dir, ".build", f"__editable__.{name}-{tag}"))
        return _LinkTree(self.distribution, name, link_tree, build_lib)


class build_py(_build_py):
    """Bundle configuration, dataset adapters and runtime resources from their owners."""

    def run(self):
        root = Path(__file__).resolve().parent
        output = Path(self.build_lib).resolve()
        if output == root:
            raise ValueError("Package build output must differ from source")
        for package in ("saebase", "models", "tasks", "conf"):
            if (output / package).exists():
                shutil.rmtree(output / package)
        super().run()
        for target, source in self.resource_files():
            target.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(source), str(target))

    def resource_files(self):
        root = Path(__file__).resolve().parent
        output = Path(self.build_lib).resolve()
        for source_dir, target_dir, patterns in (
            ("conf", "conf", ("*.yaml", "*.yml", "*.json")),
            ("datasets", "saebase/datasets", ("*.py",)),
            ("assets/research_lab/application/files/data", "saebase/research_lab/data", ("*.json",)),
            ("doc/code/research_lab/source", "saebase/research_lab/frontend", ("*.html", "*.js", "*.css")),
            ("assets/dataset_cohorts/stage_d/files", "assets/dataset_cohorts/stage_d/files", ("*.jsonl",)),
        ):
            source = root / source_dir
            for pattern in patterns:
                for file in sorted(source.rglob(pattern)):
                    yield output / target_dir / file.relative_to(source), file

    def get_output_mapping(self):
        return {
            **super().get_output_mapping(),
            **{str(target): str(source) for target, source in self.resource_files()},
        }


packages = find_namespace_packages(include=("models*", "tasks*"))
for package, location in PACKAGE_DIRS.items():
    packages.append(package)
    exclude = ("service", "service.*") if package == "saebase.research_lab" else ()
    packages.extend(f"{package}.{name}" for name in find_namespace_packages(location, exclude=exclude))

setup(
    cmdclass={"build": build, "build_py": build_py, "egg_info": egg_info, "editable_wheel": editable_wheel},
    packages=packages,
    package_dir=PACKAGE_DIRS,
)
