"""Entry point for Work Plan 05.

Run from the SAEBase repository root:
  python tasks/05_research_lab_cpu_closure_and_gpu_handoff/__main__.py run all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import apply_env, ensure_gpu_idle, gpu_snapshot, log, release_root, write_json  # noqa: E402
from cpu_lock import run_cpu_lock  # noqa: E402
from fill import collect, write_work_plan  # noqa: E402
from later import run_gpu03, run_gpu08, run_gpu09  # noqa: E402
from run_gpu import run_gpu00, run_gpu01, run_gpu02, run_gpu07  # noqa: E402


def _tokenizer_report(root: Path) -> None:
    ledger = root / "GPU-01" / "pilot_20" / "ledger.json"
    if not ledger.is_file():
        return
    import json

    rows = json.loads(ledger.read_text(encoding="utf-8"))
    write_json(
        root / "GPU-00" / "tokenizer_and_position_report.json",
        [{"site_id": row.get("site_id"), "tokenizer_report": row.get("tokenizer_report")} for row in rows],
    )


def _finish(root, pilot) -> None:
    from analyses import run_gpu04, run_gpu05, run_gpu06, run_gpu07_support

    sources = json_sources(root)
    run_gpu04(root)
    run_gpu05(root, sources)
    run_gpu06(root, sources)
    run_gpu07(root, pilot)
    run_gpu07_support(root)
    run_gpu03(root)
    run_gpu08(root)
    run_gpu09(root)
    write_work_plan(collect(root))
    log(root, "section 6 filled")
    from section52 import run_section_52

    run_section_52(root)


def json_sources(root):
    import json

    path = root / "GPU-00" / "estimator_sources.json"
    if not path.is_file():
        from accel import prepare_estimators

        return prepare_estimators(root)
    return json.loads(path.read_text(encoding="utf-8"))


def run_all() -> int:
    from accel import prepare_estimators, probe_vllm

    root = release_root()
    apply_env(root)
    first = gpu_snapshot()
    write_json(root / "GPU-00" / "gpu_sample_start.json", first)
    log(root, f"release {root.name}")
    try:
        cpu = run_cpu_lock(root)
    except Exception as exc:
        log(root, "CPU lock failed: " + str(exc).splitlines()[0][:400])
        write_json(root / "GPU-00" / "cpu_lock_error.json", {"error": str(exc).splitlines()[0][:500]})
        write_work_plan(collect(root))
        return 1
    run_gpu00(root, cpu)
    prepare_estimators(root)
    probe_vllm(root)
    ensure_gpu_idle(root, first)
    pilot = run_gpu01(root)
    _tokenizer_report(root)
    run_gpu02(root, pilot)
    _finish(root, pilot)
    return 0 if pilot.get("go") and cpu.get("passed") else 2


def run_resume() -> int:
    from accel import prepare_estimators, probe_vllm

    root = release_root()
    apply_env(root)
    summary_path = root / "GPU-00" / "cpu_lock_summary.json"
    if not summary_path.is_file():
        raise SystemExit("CPU lock summary is missing; run all first")
    import json

    cpu = json.loads(summary_path.read_text(encoding="utf-8"))
    if not cpu.get("passed"):
        raise SystemExit("CPU lock did not pass; refusing to resume GPU work")
    first = gpu_snapshot()
    write_json(root / "GPU-00" / "gpu_sample_start.json", first)
    prepare_estimators(root)
    probe_vllm(root)
    ensure_gpu_idle(root, first)
    pilot = run_gpu01(root)
    _tokenizer_report(root)
    run_gpu02(root, pilot)
    _finish(root, pilot)
    return 0 if pilot.get("go") else 2


def run_report() -> int:
    root = release_root()
    apply_env(root)
    write_work_plan(collect(root))
    log(root, "section 6 refreshed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("target", choices=["all", "cpu", "resume", "report"])
    args = parser.parse_args()
    if args.target == "all":
        return run_all()
    if args.target == "resume":
        return run_resume()
    if args.target == "report":
        return run_report()
    root = release_root()
    apply_env(root)
    if args.target == "cpu":
        run_cpu_lock(root)
        return 0
    raise SystemExit("unknown target")


if __name__ == "__main__":
    raise SystemExit(main())
