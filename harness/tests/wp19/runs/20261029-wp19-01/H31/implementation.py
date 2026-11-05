"""Report domain contracts required by explicit IO signals but not selected."""

from __future__ import annotations

KNOWN = {
    "CT-AST",
    "CT-BASE",
    "CT-CHECK",
    "CT-CONFIG",
    "CT-CONTEXT",
    "CT-DATA",
    "CT-DETERMINISM",
    "CT-ENV",
    "CT-ERROR",
    "CT-EVIDENCE",
    "CT-INTERFACE",
    "CT-LOCALE",
    "CT-NAMESPACE",
    "CT-PRIVACY",
    "CT-RECOVERY",
    "CT-RELEASE",
    "CT-RESOURCE",
    "CT-REVIEW",
    "CT-SCOPE",
    "CT-STRUCT",
    "CT-TRACE",
    "CT-TRUTH",
}
SIGNALS = {
    "bilingual_fields": "CT-LOCALE",
    "config_metadata": "CT-CONFIG",
    "credentials": "CT-PRIVACY",
    "dataset_manifest": "CT-DATA",
    "environment_observation": "CT-ENV",
    "file_evidence": "CT-EVIDENCE",
    "glossary_terms": "CT-CONTEXT",
    "namespace_paths": "CT-NAMESPACE",
    "parent_edges": "CT-STRUCT",
    "release_gate": "CT-RELEASE",
    "resource_budget": "CT-RESOURCE",
    "resume_receipt": "CT-RECOVERY",
    "review_state": "CT-REVIEW",
    "sensitive_output": "CT-PRIVACY",
    "source_ast": "CT-AST",
    "source_reference": "CT-EVIDENCE",
    "tree_nodes": "CT-STRUCT",
}


class InputError(ValueError):
    """Raised for unknown signals, unknown contracts, or duplicated names."""


def solve(payload: dict) -> dict:
    """Return missing contract ids implied by the declared signals."""
    if not isinstance(payload, dict) or set(payload) != {"selected", "signals"}:
        raise InputError("schema")
    selected = set(_words(payload["selected"], KNOWN))
    signals = _words(payload["signals"], set(SIGNALS))
    return {"missing": sorted({SIGNALS[name] for name in signals} - selected)}


def _words(values: object, known: set[str]) -> list[str]:
    """Accept a unique list of nonempty strings from a closed vocabulary."""
    if not isinstance(values, list):
        raise InputError("list")
    seen: set[str] = set()
    clean: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value not in known:
            raise InputError("unknown")
        if value in seen:
            raise InputError("duplicate")
        seen.add(value)
        clean.append(value)
    return clean
