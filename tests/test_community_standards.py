"""Check SAEBase contributor templates and actual CODEOWNERS formatting."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"


def test_issue_templates_are_valid_and_have_unique_field_ids():
    for name in ("bug_report.yml", "feature_request.yml", "model_sae_support.yml"):
        form = yaml.safe_load((TEMPLATES / name).read_text(encoding="utf-8"))
        assert isinstance(form, dict), name
        assert isinstance(form.get("name"), str) and form["name"], name
        assert isinstance(form.get("description"), str) and form["description"], name
        assert isinstance(form.get("body"), list) and form["body"], name
        ids = set()
        for field in form["body"]:
            assert field["type"] in {"textarea", "input", "checkboxes", "markdown", "dropdown"}
            if field["type"] == "markdown":
                continue
            assert isinstance(field.get("id"), str) and field["id"] not in ids
            ids.add(field["id"])
            assert isinstance(field.get("attributes"), dict)
            assert field["attributes"].get("label")
    config = yaml.safe_load((TEMPLATES / "config.yml").read_text(encoding="utf-8"))
    assert isinstance(config.get("blank_issues_enabled"), bool)


def test_codeowners_assigns_real_named_maintainer_to_critical_paths():
    text = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    rules = [
        line.split() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert all(len(parts) == 2 and parts[1] == "@AtheavenMa" for parts in rules)
    covered = {parts[0] for parts in rules}
    assert {"*", "/.github/", "/saebase/api/", "/saebase/sae/",
            "/saebase/store/", "/saebase/benchmarks/", "/models/"}.issubset(covered)
