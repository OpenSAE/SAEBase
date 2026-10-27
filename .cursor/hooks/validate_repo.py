"""Validate integrated SAEBase source structure, Cursor hooks and Python syntax.

The existing public SAEBase package is intentionally preserved alongside
OpenResearch's DLBase, dataset and task execution layers.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def check(full: bool=False) -> list[str]:
    errors=[]
    policy=json.loads((ROOT/".cursor/policy.json").read_text(encoding="utf-8"))
    hooks=json.loads((ROOT/".cursor/hooks.json").read_text(encoding="utf-8"))
    for entries in hooks.get("hooks",{}).values():
        for entry in entries:
            script=entry["command"].split()[-1]
            if not (ROOT/script).is_file():
                errors.append("Missing Cursor hook "+script)
    for rule in (ROOT/".cursor/rules").rglob("*.mdc"):
        body=rule.read_text(encoding="utf-8")
        if not body.startswith("---\n") or "\n---\n" not in body[4:]:
            errors.append("Invalid Cursor rule "+str(rule))
    for skill in (ROOT/".cursor/skills").rglob("SKILL.md"):
        body=skill.read_text(encoding="utf-8")
        if not body.startswith("---\n") or "name:" not in body:
            errors.append("Invalid Cursor skill "+str(skill))
    required=[
        "saebase/api/__init__.py","saebase/sae/__init__.py",
        "saebase/data/__init__.py","saebase/cli.py",
        "models/plugins.py","models/dlbase_plugins.py",
        "tasks/plugins.py","tasks/dlbase_plugins.py",
        "research_datasets/plugins.py","research_datasets/NLP/artifacts.py",
        "dlbase/train.py","dlbase/registry.py",
        "conf/config.yaml","conf/openresearch.yaml",
        "harness/core/agent_flow.py","harness/knowledge/validate.py",
        "harness/code/build.py",".cursor/hooks/guard_paths.py"
    ]
    for path in required:
        if not (ROOT/path).is_file():
            errors.append("Missing integrated contract "+path)
    if "saebase" not in policy.get("approved_roots",[]):
        errors.append("SAEBase root must be registered in the policy")
    if full:
        for directory in ("saebase","dlbase","datasets","models","tasks","harness"):
            for path in (ROOT/directory).rglob("*.py"):
                try: ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
                except (SyntaxError,UnicodeError) as exc:
                    errors.append("Python syntax "+str(path)+": "+str(exc))
        for path in (ROOT/"doc").rglob("*.json"):
            try: json.loads(path.read_text(encoding="utf-8"))
            except (ValueError,UnicodeError) as exc:
                errors.append("Document JSON "+str(path)+": "+str(exc))
    return errors

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--full",action="store_true")
    args=p.parse_args()
    errors=check(full=args.full)
    for error in errors: print("FAIL",error)
    if not errors: print("PASS: SAEBase / OpenResearch structural checks")
    return int(bool(errors))

if __name__=="__main__":
    raise SystemExit(main())
