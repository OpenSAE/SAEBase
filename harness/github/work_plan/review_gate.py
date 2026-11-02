"""Check Work Plan human-summary shape and declared six-domain node-impact decisions.

This is a documentation gate, NOT semantic correctness, source authorization,
Human Check approval, real test execution, or scientific acceptance.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PLAN_ID = re.compile(r'"plan_id"\s*:\s*"(WP[0-9]+)"')
MAIN_SECTION = re.compile(r"^## ([1-9][0-9]*)[.、．]\s+(.+)$", re.M)
SUMMARY_SECTION = re.compile(r"^## ([1-9][0-9]*)[.、．]\s+(.+)$", re.M)
OVERVIEW = re.compile(r"^## 概览\s*$", re.M)
CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
DOMAINS = frozenset({"code", "knowledge", "environment", "core", "harness", "context"})
DECISIONS = frozenset({"NONE", "ADD", "CHANGE", "REMOVE", "DEFER"})
FEATURE_EFFECTS = frozenset({"UNCHANGED", "CHANGED", "NEW", "REMOVED", "UNKNOWN"})


class ReviewGateError(ValueError):
    """A formal pre-review artifact is incomplete or internally inconsistent."""


def _assert(ok: bool, message: str) -> None:
    if not ok:
        raise ReviewGateError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_plan(text: str) -> tuple[str, list[int]]:
    ids = PLAN_ID.findall(text)
    _assert(len(set(ids)) == 1 and bool(ids), "Work Plan ID missing or inconsistent")
    headings = [int(x) for x, _ in MAIN_SECTION.findall(text)]
    _assert(5 <= len(headings) <= 12, "Work Plan needs 5–12 explicit main sections")
    _assert(headings == list(range(1, len(headings) + 1)), "Work Plan sections out of order")
    return ids[0], headings


def _body_between_sections(text: str, start: int, end: int) -> str:
    return text[start:end].strip()


def _count(text: str) -> int:
    return len(CJK.findall(text))


def check_summary(summary: str, plan_id: str, plan_sections: list[int]) -> dict:
    _assert(summary.startswith(f"# {plan_id} · 人类审阅摘要\n"), "Invalid human-summary heading")
    m = OVERVIEW.search(summary)
    _assert(m is not None, "Missing summary overview")
    sections = list(SUMMARY_SECTION.finditer(summary))
    nums = [int(x.group(1)) for x in sections]
    _assert(nums == plan_sections, "Summary must match every Work Plan section in order")
    _assert(m.start() < sections[0].start(), "Overview must precede section summaries")
    overview = _body_between_sections(summary, m.end(), sections[0].start())
    overview_n = _count(overview)
    _assert(40 <= overview_n <= 100, "Overview requires 40–100 Chinese characters")
    counts: dict[str, int] = {}
    for index, match in enumerate(sections):
        stop = sections[index + 1].start() if index + 1 < len(sections) else len(summary)
        body = _body_between_sections(summary, match.end(), stop)
        n = _count(body)
        _assert(25 <= n <= 115, f"Section {nums[index]} needs 25–115 Chinese characters")
        counts[str(nums[index])] = n
    total = overview_n + sum(counts.values())
    _assert(400 <= total <= 950, "Summary requires 400–950 Chinese characters total")
    return {"overview_characters": overview_n, "section_characters": counts, "total_characters": total}


def check_impact(data: dict, plan_id: str) -> dict:
    _assert(isinstance(data, dict), "Impact audit must be an object")
    _assert(data.get("schema") == "saebase.workplan.node_impact.v1", "Invalid impact schema")
    _assert(data.get("plan_id") == plan_id, "Impact audit plan mismatch")
    _assert(re.fullmatch(r"[a-f0-9]{40}", str(data.get("source_commit", ""))) is not None,
            "Impact audit must bind an exact source commit")
    entries = data.get("domains")
    _assert(isinstance(entries, list) and len(entries) == len(DOMAINS),
            "Exactly six explicit domain decisions required")
    decisions: dict[str, str] = {}
    for item in entries:
        _assert(isinstance(item, dict), "Invalid domain decision")
        domain = item.get("domain")
        _assert(domain in DOMAINS and domain not in decisions, "Missing/repeated/unknown domain")
        decision = item.get("decision")
        _assert(decision in DECISIONS, f"Invalid decision for {domain}")
        _assert(item.get("feature_effect") in FEATURE_EFFECTS, "Feature effect missing")
        _assert(isinstance(item.get("source_paths"), list) and bool(item["source_paths"]),
                "Each domain must cite source files")
        _assert(all(isinstance(p, str) and p and not p.startswith("/")
                    and ".." not in Path(p).parts for p in item["source_paths"]),
                "Unsafe/invalid source path")
        _assert(isinstance(item.get("reason"), str) and len(item["reason"].strip()) >= 20,
                f"Source-grounded reason missing for {domain}")
        _assert(isinstance(item.get("existing_nodes"), list), "Existing node IDs must be listed")
        proposed = item.get("proposed_fields", {})
        _assert(isinstance(proposed, dict), "Proposed node fields must be a mapping")
        if decision == "NONE":
            _assert(not proposed, f"{domain}: NONE cannot request changes to node fields")
        if decision in {"ADD", "CHANGE", "REMOVE"}:
            _assert(bool(proposed), f"{domain}: node change requires explicit fields")
            _assert(item.get("human_check_required") is True,
                    "A node mutation needs independent Human Check")
            if decision == "CHANGE":
                _assert(bool(item["existing_nodes"]), "CHANGE needs an existing node")
            for value in proposed.values():
                if isinstance(value, str):
                    _assert(not re.search(r"\bWP[0-9]+\b|\bQA500\b", value, flags=re.I),
                            "Durable node text may not absorb a specific Work Plan or QA500")
        if decision == "DEFER":
            _assert(item.get("feature_effect") == "UNKNOWN", "DEFER must identify unknown impact")
        decisions[domain] = decision
    _assert(set(decisions) == DOMAINS, "Six domains must be covered exactly once")
    semantic = any(x in {"ADD", "CHANGE", "REMOVE"} for x in decisions.values())
    expected = "NODE_HUMAN_CHECK" if semantic else (
        "DEFER_NODE_DECISION" if "DEFER" in decisions.values() else "NO_NODE_CHANGE"
    )
    _assert(data.get("node_review_decision") == expected, "Incorrect node-review decision")
    source = data.get("source_write_review")
    _assert(isinstance(source, dict)
            and isinstance(source.get("required"), bool)
            and source.get("status") in {"PENDING", "APPROVED", "NOT_REQUIRED"},
            "Source-write review must be decided independently from node-review")
    _assert(not source["required"] or source["status"] != "NOT_REQUIRED",
            "Required source review cannot be NOT_REQUIRED")
    return {"node_review_decision": expected, "domains": decisions,
            "source_write_review": source["status"]}


def validate(plan: Path, summary: Path, impact: Path) -> dict:
    plan_id, sections = _parse_plan(_read(plan))
    s = check_summary(_read(summary), plan_id, sections)
    a = check_impact(json.loads(_read(impact)), plan_id)
    return {"status": "STRUCTURE_VALID_ONLY", "plan_id": plan_id, "summary": s, "impact": a,
            "semantic_truth": "NOT_VERIFIED", "execution": "NOT_RUN"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--impact", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.plan, args.summary, args.impact)
    except (ReviewGateError, ValueError, OSError) as exc:
        parser.exit(2, "WORKPLAN REVIEW BLOCKED: " + str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
