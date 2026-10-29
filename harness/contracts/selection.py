"""Small deterministic audit of task declarations; no AI classifier or inferred permissions."""
from __future__ import annotations

from harness.contracts.registry import ContractError

SIGNALS = {
    "source_reference": "CT-EVIDENCE",
    "file_evidence": "CT-EVIDENCE",
    "sensitive_output": "CT-PRIVACY",
    "credentials": "CT-PRIVACY",
    "tree_nodes": "CT-STRUCT",
    "parent_edges": "CT-STRUCT",
    "review_state": "CT-REVIEW",
    "bilingual_fields": "CT-LOCALE",
    "glossary_terms": "CT-CONTEXT",
    "source_ast": "CT-AST",
    "environment_observation": "CT-ENV",
}
def audit_selection(selected: list[str], declared_signals: list[str]) -> list[str]:
    """Return missing mandatory contracts from explicit structured IO/operation signals.

    A signal is an observable task declaration, NOT a keyword guess from prose.
    Unknown signals fail closed to avoid pretending completeness.
    """
    if not isinstance(selected, list) or not isinstance(declared_signals, list):
        raise ContractError("Contract selection and task signals must be lists")
    if any(not isinstance(x, str) or x not in SIGNALS for x in declared_signals):
        raise ContractError("Unrecognized task IO signal: CONTRACT_GAP")
    return sorted({SIGNALS[x] for x in declared_signals} - set(selected))
