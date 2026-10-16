#!/usr/bin/env python3
"""Versioned SAE benchmark CLI. Dry-run never downloads checkpoints."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from saebase.benchmarks import load_config,build_plan
from saebase.benchmarks.runner import execute, validate_external_sources
from saebase.benchmarks.results import read_metrics
from saebase.benchmarks.report import write_report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument(
        "--extension", action="append", default=[],
        help="Opt in to an installed opensae.saebase.extensions.v1 entry-point by name",
    )
    p.add_argument("--output",default="runs",type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--real",action="store_true")
    mode.add_argument("--report-only",action="store_true")
    p.add_argument("--resume",action="store_true",default=True)
    p.add_argument("--no-resume",dest="resume",action="store_false")
    args=p.parse_args(argv)
    if args.extension:
        from saebase.extensions import discover_extensions
        discover_extensions(args.extension)
    cfg=load_config(args.config)
    plan=build_plan(cfg)
    if args.dry_run:
        print(json.dumps(plan,indent=2))
        return 0
    if args.report_only:
        root=args.output/cfg.run_id
        rows=read_metrics(root,cfg.run_id)
        validate_external_sources(cfg,rows,root=root)
        write_report(root,cfg,plan,rows)
        print(json.dumps({"run_id":cfg.run_id,"metrics":len(rows),"report":str(root/"report.html")}))
        return 0
    rows=execute(cfg,args.output,real=args.real,resume=args.resume)
    print(json.dumps({"run_id":cfg.run_id,"metrics":len(rows),
                      "report":str(args.output/cfg.run_id/"report.html")},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
