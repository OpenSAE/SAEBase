"""Run fast repository structure and source contract checks without GPU/data."""
import argparse, ast, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
ERRORS = []
def verify():
    policy = json.loads((ROOT / ".cursor/policy.json").read_text())
    hooks = json.loads((ROOT / ".cursor/hooks.json").read_text())
    assert hooks["version"] == 1
    for kind, items in hooks["hooks"].items():
        for item in items:
            script = item["command"].split()[-1]
            if not (ROOT / script).is_file():
                ERRORS.append(f"Missing {kind} hook: {script}")
    for item in ROOT.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir() and item.name not in policy["approved_roots"]:
            ERRORS.append(f"Unapproved root directory: {item.name}")
        if item.is_file() and item.name not in policy["approved_files"]:
            ERRORS.append(f"Unapproved root file: {item.name}")
    for rule in (ROOT / ".cursor/rules").rglob("*.mdc"):
        c = rule.read_text(encoding="utf-8")
        if not c.startswith("---\n") or "\n---\n" not in c[4:]:
            ERRORS.append(f"Invalid frontmatter: {rule}")
    for skill in (ROOT / ".cursor/skills").rglob("SKILL.md"):
        c = skill.read_text(encoding="utf-8")
        if not c.startswith("---\n") or "name:" not in c or "description:" not in c:
            ERRORS.append(f"Invalid skill: {skill}")
    # SAEBase is infrastructure: no concrete datasets or separate task registry.
    library = ROOT / "saebase"
    domains = {p.name for p in library.iterdir() if p.is_dir() and p.name != "__pycache__"}
    if domains != {"api", "sae", "store"}:
        ERRORS.append(f"SAEBase must contain only api/ sae/ store/: {sorted(domains)}")
    if (library / "data").exists() or (library / "cfg").exists():
        ERRORS.append("SAEBase may not own dataset or duplicate Hydra configuration")

def full():
    for root in ("dlbase", "saebase", "datasets", "models", "tasks"):
        for file in (ROOT / root).rglob("*.py"):
            try:
                tree = ast.parse(file.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeError) as e:
                ERRORS.append(f"Python syntax: {file}: {e}")
                continue
            for node in ast.walk(tree):
                imports = []
                if isinstance(node, ast.Import):
                    imports = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                for value in imports:
                    if value.split(".")[0] in ("scripts", "tests"):
                        ERRORS.append(f"Production imports auxiliary module: {file}: {value}")
                    retired = (
                        "saebase.data", "saebase.cfg", "saebase.registry",
                        "saebase.model_plugins", "saebase.plugin_bootstrap",
                        "saebase.results", "saebase.api.component_analysis",
                        "saebase.api.fine_components",
                    )
                    if any(value == root or value.startswith(root + ".") for root in retired):
                        ERRORS.append(f"Retired SAEBase namespace imported: {file}: {value}")
    # Family directories contain concrete components. Their root dispatch
    # plugins own shared policy; a concrete model retains its own compatibility.
    families = {
        "datasets": {"vision", "NLP"},
        "models": {"vision", "LLM"},
        "tasks": {"vision", "LLM"},
    }
    for domain, names in families.items():
        if not (ROOT / domain / "plugins.py").is_file():
            ERRORS.append(f"Missing {domain}/plugins.py")
        actual = {p.name for p in (ROOT / domain).iterdir() if p.is_dir()
                  and p.name != "__pycache__"}
        if actual != names:
            ERRORS.append(f"Unexpected {domain} family directories: {sorted(actual ^ names)}")
    expected = {
        "datasets/vision/CUB_200/dataset_task_policy.py",
        "models/vision/resnet18_mlp/model_task_dataset_compatibility.py",
        "models/LLM/gemma2_2b_base/model_task_dataset_compatibility.py",
        "tasks/vision/classification/pipeline_checks.py",
        "tasks/LLM/sae_runtime.py",
    }
    for entry in expected:
        if not (ROOT / entry).is_file():
            ERRORS.append(f"Missing component contract: {entry}")
    for family in ("vision", "LLM"):
        for module in (ROOT / "tasks" / family).iterdir():
            if module.is_dir() and module.name != "__pycache__":
                if not any((module / fn).is_file() for fn in ("task.py", "classification_task.py")):
                    ERRORS.append(f"Missing task implementation: {module}")
    for item in (ROOT / "datasets/NLP").iterdir():
        if item.is_dir() and item.name != "__pycache__":
            if not any(p.name == item.name + ".py" for p in item.iterdir()):
                ERRORS.append(f"Missing NLP source implementation: {item}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    verify()
    if args.full:
        full()
    for error in ERRORS:
        print("FAIL", error)
    if not ERRORS:
        print("PASS: harness and repository checks" + (" (full)" if args.full else ""))
    return bool(ERRORS)
if __name__ == "__main__":
    sys.exit(main())
