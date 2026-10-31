"""Check common allowed and rejected repository layouts."""

from harness.core.check_architecture import validate_path

ROOTS = {
    ".github",
    ".cursor",
    "conf",
    "doc",
    "harness",
    "models",
    "saebase",
    "tasks",
    "datasets",
    "scripts",
    "tests",
    "atlas",
}


def test_owned_layouts():
    for p in [
        "scripts/workplan/wp22.sh",
        "tests/workplan/wp22/test_run.py",
        "scripts/ioi/gemma2/capture/basic/run.sh",
        "tests/ioi/gemma2/capture/basic/test_run.py",
        "scripts/experiments/qa500.sh",
        "datasets/BoolQ/qa_adapter.py",
    ]:
        assert not validate_path(p, ROOTS), p


def test_unowned_layouts():
    for p in [
        "datasets/qa500_runner.py",
        "tasks/capture/run.sh",
        "scripts/random.sh",
        "tests/test_random.py",
        "unknown_new_module/runner.py",
        "experiments/qa500/runner.py",
    ]:
        assert validate_path(p, ROOTS), p

def test_python_package_markers_are_not_unowned_programs():
    assert not validate_path("tests/__init__.py", ROOTS)
    assert not validate_path("scripts/__init__.py", ROOTS)
    assert validate_path("tests/test_unowned.py", ROOTS)
