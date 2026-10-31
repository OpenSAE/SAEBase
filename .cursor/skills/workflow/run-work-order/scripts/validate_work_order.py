"""Work Order v3: independently activate Core, Code, Knowledge and glossary."""

from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required (install project requirements).") from exc

ROOT = Path(__file__).resolve().parents[5]
ORDERS = ROOT / ".cursor/work_orders"
MODULES = {"core", "code", "knowledge", "experiments", "transfer", "environment"}


class InvalidOrder(ValueError):
    pass


def demand(condition, message):
    if not condition:
        raise InvalidOrder(message)


def keys(value, expected, label):
    demand(isinstance(value, dict), label + " must be a mapping")
    demand(set(value) == expected, label + " keys must be " + str(sorted(expected)))
    return value


def reference(value, kind, check_exists=True):
    demand(isinstance(value, str) and value and value == value.strip(), "Invalid reference")
    demand(
        not value.startswith("/") and "\\" not in value, "References must be relative POSIX paths"
    )
    demand(not any(c in value for c in "*?[]"), "Globs are prohibited")
    parts = value.split("/")
    demand(all(part not in ("", ".", "..") for part in parts), "Unsafe path traversal")
    demand(tuple(parts) == PurePosixPath(value).parts, "Noncanonical path")
    prefixes = (
        ("doc/knowledge/", "doc/code/")
        if kind == "doc"
        else ("harness/github/work_plan/",)
    )
    demand(any(value.startswith(p) for p in prefixes), f"Reference is outside {kind}: {value}")
    dest = (ROOT / value).resolve(strict=False)
    roots = ("doc/knowledge", "doc/code") if kind == "doc" else ("harness/github/work_plan",)
    demand(
        any(dest.is_relative_to((ROOT / p).resolve(strict=False)) for p in roots),
        "Reference symlink escapes root",
    )
    if check_exists:
        demand(dest.is_file(), "Reference file missing: " + value)
    return value


def valid_ids(ids, label):
    demand(isinstance(ids, list), label + " must be a list")
    demand(all(isinstance(x, str) and x for x in ids), label + " entries must be nonempty strings")
    demand(len(set(ids)) == len(ids), label + " contains duplicate IDs")
    return ids


def known_ids(relative_path, field):
    content = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    return {x["id"] for x in content[field]}


def validate(path, template=False):
    demand(
        path.suffix == ".yaml" and path.parent == ORDERS.resolve(),
        "Work Order must be a direct .cursor/work_orders/*.yaml",
    )
    demand(path.is_file(), "Work Order not found: " + str(path))
    order = yaml.safe_load(path.read_text(encoding="utf-8"))
    regular = {"version", "id", "status", "goal", "activation", "context"}
    demand(isinstance(order, dict), "Work Order must be a mapping")
    if "contract_binding" in order:
        keys(order, regular | {"contract_binding"}, "Work Order")
        binding = keys(order["contract_binding"], {"plan", "case", "contract_ids"}, "contract_binding")
        sys.path.insert(0, str(ROOT))
        from harness.contracts.registry import ContractError, resolve

        try:
            selection = resolve(binding["plan"], binding["case"])
        except (ContractError, TypeError, OSError) as exc:
            raise InvalidOrder("Invalid Contract Registry binding: " + str(exc)) from exc
        demand(binding["contract_ids"] == selection["contract_ids"],
               "Work Order contract IDs must match the predeclared Work Plan")
    else:
        # Historical v3 Work Orders remain parseable but do not claim to
        # have selected or passed universal task contract validation.
        keys(order, regular, "Work Order")
    demand(type(order["version"]) is int and order["version"] == 3, "Work Order v3 required")
    demand(
        order["status"] == ("template" if template else "ready"),
        "Expected ready status (templates are inert)",
    )
    demand(
        isinstance(order["id"], str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,63}", order["id"]) is not None,
        "Invalid ID",
    )
    if not template:
        demand(order["id"] == path.stem, "Work Order filename differs from ID")
    demand(isinstance(order["goal"], str) and bool(order["goal"].strip()), "Goal is required")
    activation = keys(order["activation"], MODULES, "activation")
    for name, enabled in activation.items():
        demand(type(enabled) is bool, f"activation.{name} must be true/false")
    ctx = keys(
        order["context"],
        {"doc", "core_ids", "knowledge_ids", "term_ids", "harness", "unlisted"},
        "context",
    )
    demand(ctx["unlisted"] == "deny", "Context policy must be unlisted: deny")
    doc = keys(ctx["doc"], {"required", "optional"}, "doc")
    ref = keys(ctx["harness"], {"read_only"}, "harness")
    all_paths = []
    result = {}
    for key, values, kind in (
        ("required", doc["required"], "doc"),
        ("optional", doc["optional"], "doc"),
        ("read_only", ref["read_only"], "harness"),
    ):
        valid_ids(values, key)
        for value in values:
            all_paths.append(reference(value, kind, check_exists=not template))
        result[key] = len(values)
    demand(
        len(all_paths) == len(set(all_paths)), "Duplicate file across required/optional/read_only"
    )
    sets = {
        "core_ids": valid_ids(ctx["core_ids"], "core_ids"),
        "knowledge_ids": valid_ids(ctx["knowledge_ids"], "knowledge_ids"),
        "term_ids": valid_ids(ctx["term_ids"], "term_ids"),
    }
    demand(not sets["core_ids"] or activation["core"], "core_ids require activation.core=true")
    demand(
        not (sets["knowledge_ids"] or sets["term_ids"]) or activation["knowledge"],
        "Scientific IDs and term IDs require activation.knowledge=true",
    )
    if not template:
        catalogs = {
            "core_ids": known_ids("harness/core/project_graph.json", "nodes"),
            "knowledge_ids": known_ids("doc/knowledge/context_graph.json", "nodes"),
            "term_ids": known_ids("doc/knowledge/glossary.json", "terms"),
        }
        for field, values in sets.items():
            missing = set(values) - catalogs[field]
            demand(not missing, f"Unknown {field}: {sorted(missing)}")
    for field, values in sets.items():
        result[field] = len(values)
    result["modules"] = sum(activation.values())
    result["contracts"] = (
        len(order["contract_binding"]["contract_ids"])
        if "contract_binding" in order else "legacy_unbound"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_order", help="Work Order ID or relative path")
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()
    key = args.work_order
    path = (
        ORDERS / (key + ".yaml") if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,63}", key) else Path(key)
    )
    if not path.is_absolute():
        path = ROOT / path
    try:
        result = validate(path.resolve(strict=False), template=args.template)
    except (InvalidOrder, OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print("INVALID Work Order:", exc, file=sys.stderr)
        return 1
    print("VALID Work Order:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
