"""Check that public task configuration keys follow SAEBase's file structure.

This lightweight check reads the shipped task recipes and matches them to their
implementation directories. Run it when adding, renaming, or moving task keys.
"""

from ast import AnnAssign, ClassDef, parse
from pathlib import Path
import re

import pytest
import yaml

from scripts.documentation import (
    checked_configuration_files,
    configuration_entries,
    configuration_navigation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIGS = tuple(sorted((PROJECT_ROOT / "conf" / "tasks").glob("*/task.yaml")))
# Consumed by the dispatcher, common resource handling, or shared runtime.
SHARED_TASK_KEYS = {
    "name",
    "extensions",
    "execute",
    "sae_key",
    "layer",
    "device",
    "prompt",
    "performance",
    "resume",
    "timeout_seconds",
}


def test_task_recipes_match_implementation_directories():
    recipe_names = {path.parent.name for path in TASK_CONFIGS}
    implementation_names = {path.parent.name for path in (PROJECT_ROOT / "tasks").glob("*/task.py")}
    assert recipe_names, "No task recipes found in conf/tasks/"
    assert recipe_names == implementation_names, (
        f"Task recipes without tasks/<name>/task.py: {sorted(recipe_names - implementation_names)}\n"
        f"Task implementations without conf/tasks/<name>/task.yaml: "
        f"{sorted(implementation_names - recipe_names)}"
    )


@pytest.mark.parametrize("recipe", TASK_CONFIGS, ids=lambda path: path.parent.name)
def test_task_configuration_prefix_matches_recipe(recipe: Path):
    owner = recipe.parent.name
    location = recipe.relative_to(PROJECT_ROOT)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    assert isinstance(config, dict) and isinstance(config.get("task"), dict), (
        f"{location} must declare a task mapping"
    )
    task = config["task"]
    assert task.get("name") == owner, (
        f"{location}: task.name is {task.get('name')!r}; expected {owner!r} "
        f"from tasks/{owner}/task.py"
    )
    assert isinstance(task.get(owner, {}), dict), f"{location}: task.{owner} must be a mapping"

    misplaced = sorted(task.keys() - SHARED_TASK_KEYS - {owner})
    if misplaced:
        pytest.fail(
            f"{location}: task-specific keys belong under task.{owner}.* "
            f"(tasks/{owner}/task.py). Found: " + ", ".join(f"task.{key}" for key in misplaced),
            pytrace=False,
        )


# Capture groups with a named implementation in the existing project.
CAPTURE_SUBTASK_SOURCES = {
    "sar": "sar_mode.py",
    "i": "scoring/measure_behavior.py",
    "rss": "rss_mode.py",
    "model_answer": "scoring/model_answer.py",
}


def test_capture_subtask_prefixes_match_implementation_files():
    recipe = PROJECT_ROOT / "conf" / "tasks" / "capture" / "task.yaml"
    task = yaml.safe_load(recipe.read_text(encoding="utf-8"))["task"]
    groups = task.get("capture", {})
    assert isinstance(groups, dict), f"{recipe}: task.capture must be a mapping"
    for name, filename in CAPTURE_SUBTASK_SOURCES.items():
        relative_source = Path("tasks") / "capture" / filename
        assert (PROJECT_ROOT / relative_source).is_file(), (
            f"task.capture.{name} refers to missing implementation {relative_source}"
        )
        assert isinstance(groups.get(name), dict), (
            f"{recipe.relative_to(PROJECT_ROOT)}: {relative_source} owns task.capture.{name}.*"
        )


def test_documented_task_keys_match_their_configuration():
    recipes = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in TASK_CONFIGS
    }
    path = PROJECT_ROOT / "doc" / "conf.md"
    backend = PROJECT_ROOT / "tasks" / "train" / "execution" / "training.py"
    source = parse(backend.read_text(encoding="utf-8"))
    plan = next(
        node for node in source.body if isinstance(node, ClassDef) and node.name == "TrainPlan"
    )
    plan_fields = {node.target.id for node in plan.body if isinstance(node, AnnAssign)}

    owner = None
    owner_depth = 0
    historical = False
    documented = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        heading = re.match(r"^(#{2,6}) (.+)", line)
        if heading:
            depth = len(heading[1])
            if depth <= owner_depth:
                owner = None
                historical = False
            match = re.search(r"`conf/tasks/(\w+)/task\.yaml`", heading[2])
            if match:
                owner = match[1]
                owner_depth = depth
                historical = False
            if "Migration of old capture commands" in heading[2]:
                historical = True
        if historical or not line.startswith("| "):
            continue
        fields = line.split("|")[1:3]
        if len(fields) < 2:
            continue
        for key in re.findall(r"`(task\.[\w.]+)`", fields[0]):
            where = f"doc/conf.md:{line_number}"
            parts = key.split(".")
            if parts[:3] == ["task", "train", "train_plan"] and len(parts) == 4:
                assert owner == "train" and parts[-1] in plan_fields, f"{where}: unknown {key}"
                continue
            if parts[1] in recipes:
                assert owner == parts[1], (
                    f"{where}: {key} belongs under conf/tasks/{parts[1]}/task.yaml"
                )
            candidates = [recipes[owner]] if owner else recipes.values()
            values = []
            for recipe in candidates:
                node = recipe
                for part in parts:
                    if not isinstance(node, dict) or part not in node:
                        break
                    node = node[part]
                else:
                    values.append(node)
            assert values, f"{where}: {key} is absent from the owning configuration"
            default = re.fullmatch(r"`([^`]+)`", fields[1].strip())
            if default:
                expected = yaml.safe_load(default[1])
                assert all(value == expected for value in values), (
                    f"{where}: wrong default for {key}"
                )
            documented.add(key)

    def leaves(node, prefix):
        for name, value in node.items():
            key = f"{prefix}.{name}"
            if isinstance(value, dict) and value:
                yield from leaves(value, key)
            else:
                yield key

    for task_owner, recipe in recipes.items():
        owned_keys = set(leaves(recipe["task"][task_owner], f"task.{task_owner}"))
        assert owned_keys <= documented, (
            f"{task_owner}: undocumented keys {sorted(owned_keys - documented)}"
        )


def test_configuration_navigation_matches_file_structure():
    doc = PROJECT_ROOT / "doc" / "conf.md"
    text = doc.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^(#{2,6}) (.+)$", text, re.MULTILINE))
    path_headings = {}
    for index, heading in enumerate(headings):
        title = heading[2]
        end = next(
            (h.start() for h in headings[index + 1 :] if len(h[1]) <= len(heading[1])),
            len(text),
        )
        path = re.search(r"`(conf/[^`]+)`", title)
        if path:
            relative = path[1]
            assert relative not in path_headings, f"doc/conf.md repeats a section for {relative}"
            assert (PROJECT_ROOT / relative).exists(), (
                f"doc/conf.md has a stale section for {relative}"
            )
            path_headings[relative] = heading, end

    assert text == configuration_navigation(text, PROJECT_ROOT), (
        "Refresh the path list with python -m scripts.documentation conf --write"
    )
    for path in sorted(configuration_entries(PROJECT_ROOT, PROJECT_ROOT / "conf")):
        assert path in path_headings, f"doc/conf.md: {path} needs a configuration section"
        heading, end = path_headings[path]
        assert len(heading[1]) == 2, f"doc/conf.md: {path} needs a top-level configuration section"
        if not path.endswith("/"):
            assert f"](../{path})" in text[heading.end() : end], (
                f"doc/conf.md: {heading[2]} must link to {path}"
            )
            continue

        for member in sorted(configuration_entries(PROJECT_ROOT, PROJECT_ROOT / path)):
            assert member in path_headings, f"doc/conf.md: {path} needs a subsection for {member}"
            member_heading, member_end = path_headings[member]
            assert heading.end() <= member_heading.start() < end and len(member_heading[1]) == 3, (
                f"doc/conf.md: {member} must be a direct subsection of {path}"
            )
            assert f"](../{member})" in text[member_heading.end() : member_end], (
                f"doc/conf.md: the subsection for {member} must link to its source"
            )

    for relative, (heading, _) in path_headings.items():
        parent = Path(relative.rstrip("/")).parent.as_posix() + "/"
        if parent == "conf/":
            continue
        assert parent in path_headings, f"doc/conf.md: {relative} needs a section for {parent}"
        parent_heading, parent_end = path_headings[parent]
        assert (
            parent_heading.end() <= heading.start() < parent_end
            and len(heading[1]) == len(parent_heading[1]) + 1
        ), f"doc/conf.md: {relative} belongs under {parent}"

    for target in re.findall(r"\]\(\.\./(conf/[^)#\s]+)(?:#[^)]+)?\)", text):
        assert (PROJECT_ROOT / target).exists(), f"doc/conf.md links to missing {target}"


# Entry points verified during the configuration review. The source location
# determines each expected directory, including for shared presets.
CONFIGURATION_READERS = {
    "datasets/plugins.py": ("catalog.yaml",),
    "models/plugins.py": ("catalog.yaml",),
    "base/interfaces/sae_adapters/sae_profiles.py": ("sae_profiles.yaml",),
    "base/runtime/environment.py": ("runtime.yaml",),
    "tasks/benchmark/comparison/schema.py": ("task.yaml",),
    "tasks/benchmark/results/evidence.py": ("task.yaml",),
}


def test_configuration_files_mirror_source_paths():
    actual = {path.relative_to(PROJECT_ROOT) for path in checked_configuration_files(PROJECT_ROOT)}
    readers = {
        **CONFIGURATION_READERS,
        **{
            source.relative_to(PROJECT_ROOT).as_posix(): ("task.yaml",)
            for source in (PROJECT_ROOT / "tasks").glob("*/task.py")
        },
    }
    expected = {Path("conf/config.yaml")}
    for reader, names in readers.items():
        source = Path(reader)
        assert (PROJECT_ROOT / source).is_file(), f"Configuration reader missing: {source}"
        owner = Path(*source.parent.parts[:2])
        expected.update(Path("conf") / owner / name for name in names)
    assert actual == expected, (
        f"Unverified configuration paths: {sorted(map(str, actual - expected))}\n"
        f"Configuration missing from its reader's directory: {sorted(map(str, expected - actual))}"
    )


def test_configuration_gate_rejects_a_real_file_in_a_wrong_directory(tmp_path):
    from shutil import copy2

    source = PROJECT_ROOT / "conf" / "base" / "runtime" / "runtime.yaml"
    misplaced = tmp_path / "conf" / "runtime" / source.name
    misplaced.parent.mkdir(parents=True)
    copy2(source, misplaced)
    with pytest.raises(ValueError, match=r"conf/runtime/runtime.yaml: expected a source"):
        checked_configuration_files(tmp_path)


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("conf/tasks/capture/sar/rules.yaml", "at most two nested directories"),
        ("conf/tasks/capture/other.yaml", "one file per configuration branch"),
    ],
)
def test_configuration_gate_rejects_split_task_configuration(tmp_path, relative, message):
    from shutil import copy2

    (tmp_path / "tasks").symlink_to(PROJECT_ROOT / "tasks", target_is_directory=True)
    source = PROJECT_ROOT / "conf/tasks/capture/task.yaml"
    extra = tmp_path / relative
    extra.parent.mkdir(parents=True)
    copy2(source, extra)
    copy2(source, tmp_path / source.relative_to(PROJECT_ROOT))
    with pytest.raises(ValueError, match=message):
        checked_configuration_files(tmp_path)


def test_configuration_change_refreshes_documented_default_and_snapshot(tmp_path):
    from shutil import copytree

    copytree(PROJECT_ROOT / "conf", tmp_path / "conf")
    for source in ("base", "datasets", "models", "tasks"):
        (tmp_path / source).symlink_to(PROJECT_ROOT / source, target_is_directory=True)
    path = tmp_path / "conf" / "config.yaml"
    original = path.read_text(encoding="utf-8")
    assert "seed: 42" in original
    path.write_text(original.replace("seed: 42", "seed: 43", 1), encoding="utf-8")
    doc = (PROJECT_ROOT / "doc" / "conf.md").read_text(encoding="utf-8")
    changed = configuration_navigation(doc, tmp_path)
    assert "| `seed` | `43` |" in changed
    assert "seed: 43" in changed
    assert configuration_navigation(changed, tmp_path) == changed
