"""Refresh the user-confirmed ablation selection while exhaustive inference runs.

The inference process retains its already-imported report code. This independent
reader applies the updated single definition without reloading model weights.
"""

import json
import os
import time

from tasks.capture.scoring.response_surface_report import main as build_report
from tasks.capture.scoring.reproduction_paths import RESULTS as OUTPUT, question_directory
from scripts.response_surface.verify_results import main as verify_results


def main():
    plan = json.loads((OUTPUT / "plan.json").read_text())
    directories = [question_directory(OUTPUT, item["key"]) for item in plan["questions"]]
    previous = -1
    while True:
        completed = sum((directory / "complete.json").is_file() for directory in directories)
        if completed != previous:
            build_report(report_name="report_ablation")
            previous = completed
        if completed == len(directories):
            break
        time.sleep(10)
    process_path = OUTPUT / "process.json"
    process = json.loads(process_path.read_text()) if process_path.is_file() else None
    if process is not None:
        while True:
            try:
                os.kill(process["pid"], 0)
            except ProcessLookupError:
                break
            time.sleep(5)
    build_report()
    verify_results()
    if process is not None:
        process["status"] = "complete"
        process_path.write_text(json.dumps(process, ensure_ascii=False, indent=2))
    print("All 50 questions reported and audited", flush=True)


if __name__ == "__main__":
    main()
