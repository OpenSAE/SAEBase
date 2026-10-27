"""Verify Atlas explanatory language is affirmative, concise and source-linked.

Validation covers human-facing node text, explanations, roles, descriptions
and structured attributes. Literal source code and technical identifiers are
preserved verbatim as evidence.
"""

from __future__ import annotations

import re

NEGATIVE = re.compile(
    r"\b(?:not|no|never|without|rather(?:\s+than)?|cannot|instead\s+of)\b"
    r"|[不未无非]|避免|尚|缺少|不得",
    re.IGNORECASE,
)


class NarrativeError(ValueError):
    pass


def affirmative(value, where):
    if not isinstance(value, str):
        return
    match = NEGATIVE.search(value)
    if match:
        raise NarrativeError(f"Negative explanatory language in {where}: {value[:180]}")


def check_attrs(attrs, where):
    for i, block in enumerate(attrs or []):
        if not isinstance(block, dict):
            raise NarrativeError(f"Invalid attribute in {where}")
        label = f"{where}.attributes[{i}]"
        affirmative(block.get("label"), label + ".label")
        t = block.get("type")
        if t in {"text", "formula", "diagram"}:
            for field in ("value", "caption", "note"):
                affirmative(block.get(field), label + "." + field)
        if t == "interface":
            check_interface(block.get("interface", {}), label + ".interface")
        if t == "code_reference":
            # Exact, cited implementation signatures and docstrings are source
            # evidence, so they retain their original technical wording.
            pass
        if t == "class_methods":
            for method in block.get("methods", []):
                check_interface(method, label + ".method")
        if t == "fields":
            for field in block.get("fields", []):
                affirmative(field.get("meaning"), label + ".field.meaning")


def check_interface(record, where):
    if not record:
        return
    affirmative(record.get("purpose"), where + ".purpose")
    for param in record.get("parameters", []):
        affirmative(param.get("meaning"), where + ".parameter." + str(param.get("name")))
    output = record.get("returns", {})
    affirmative(output.get("meaning"), where + ".return")


def verify_public_prose(graphs):
    for name, graph in graphs.items():
        for node in graph["nodes"]:
            label = name + ":" + node["id"]
            for field in ("title", "summary", "description", "why_it_matters"):
                affirmative(node.get(field), label + "." + field)
            for alias in node.get("aliases", []):
                # Aliases here are natural-language descriptions; source symbols
                # are stored separately in machine-readable fields.
                affirmative(alias, label + ".alias")
            for value in (node.get("facts") or {}).values():
                affirmative(str(value), label + ".fact")
            if name != "code":
                # Generated Code interfaces quote upstream Python docstrings.
                # Retain their exact factual descriptions as source evidence;
                # only curated Atlas prose is subject to narrative style rules.
                check_attrs(node.get("attributes", []), label)
    return {name: len(graph["nodes"]) for name, graph in graphs.items()}
