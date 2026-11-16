"""Refresh the user-confirmed ablation selection while exhaustive inference runs.

The inference process retains its already-imported report code. This independent
reader applies the updated single definition without reloading model weights.
"""

import json
import os
import time

from scripts.SAR.build_exhaustive_report import main as build_report
from scripts.SAR.exhaustive import OUTPUT
from scripts.SAR.verify_results import main as verify_results


def main():
    previous = -1
    while True:
        completed = len(list(OUTPUT.glob("*/complete.json")))
        if completed != previous:
            build_report(report_name="report_ablation")
            previous = completed
        if completed == 50:
            break
        time.sleep(10)
    process = json.loads((OUTPUT / "process.json").read_text())
    while True:
        try:
            os.kill(process["pid"], 0)
        except ProcessLookupError:
            break
        time.sleep(5)
    build_report()
    verify_results()
    process["status"] = "complete"
    (OUTPUT / "process.json").write_text(json.dumps(process, ensure_ascii=False, indent=2))
    print("All 50 questions reported and audited", flush=True)


if __name__ == "__main__":
    main()
