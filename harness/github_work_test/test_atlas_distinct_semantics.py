"""Regression checks for distinct Code Tree responsibilities and explanations."""

from pathlib import Path
import json

from harness.code.presentation import readable_code_tree, TASK_INSTANCE_SEMANTICS

ROOT = Path(__file__).resolve().parents[2]


def test_task_goal_wrapper_and_operation_describe_separate_work() -> None:
    source = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    tree = readable_code_tree(source)
    nodes = {node["id"]: node for node in tree["nodes"]}
    assert set(TASK_INSTANCE_SEMANTICS) == {
        "capture",
        "intervene",
        "generate",
        "circuits",
        "explain",
        "benchmark",
        "train",
    }
    for name in TASK_INSTANCE_SEMANTICS:
        family = nodes["code:instance:tasks:" + name]
        wrapper = nodes["code:module:tasks." + name + ".task"]
        operation = nodes["code:module:tasks." + name + ".operation"]
        assert len({family["title"], wrapper["title"], operation["title"]}) == 3
        assert len({family["summary"], wrapper["summary"], operation["summary"]}) == 3
        assert all(
            "源码依据和调用参数" not in node["summary"] for node in (family, wrapper, operation)
        )
        for node in (family, wrapper, operation):
            assert node.get("evidence")


def test_intervention_backends_have_specific_verified_purposes() -> None:
    source = json.loads((ROOT / "doc/code/code_tree.json").read_text(encoding="utf-8"))
    tree = readable_code_tree(source)
    nodes = {node["id"]: node for node in tree["nodes"]}
    names = (
        "clean_prefix",
        "engine",
        "hf_cached_steering",
        "kv_cache_exact",
        "operators",
        "results",
        "specs",
        "steering",
    )
    target = [nodes["code:module:tasks.intervene.backend." + name] for name in names]
    assert len({n["title"] for n in target}) == len(target)
    assert len({n["summary"] for n in target}) == len(target)
    assert all("负责实现教师强制特征干预" not in n["summary"] for n in target)
