#!/usr/bin/env python3
"""Audit an already completed M3 benchmark using existing raw M1/M2 caches."""
from __future__ import annotations
import argparse,json
from pathlib import Path

from saebase.reliability.schema import AuditConfig
from saebase.reliability.observations import _read_m3
from saebase.reliability.runner import evaluate_m3
from saebase.reliability.report import write_audit,read_audit

def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m3-root",type=Path,required=True,
                   help="runs/<m3_run_id> with verified M3 metrics and source caches")
    p.add_argument("--config",type=Path,required=True)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run",action="store_true")
    action.add_argument("--run",action="store_true")
    action.add_argument("--report-only",action="store_true")
    p.add_argument("--ground-truth",action="store_true",
                   help="Run independent known-truth cases with original SAR/RSS estimators")
    p.add_argument("--sar-source",type=Path,
                   help="Trusted original SAR repository or sar.py")
    p.add_argument("--rss-source",type=Path,
                   help="Trusted original RSS repository or rss.py")
    p.add_argument("--sae-only",action="store_true",
                   help="Opt in to SAE checkpoint loading for decoded Core metrics; never loads model")
    args=p.parse_args(argv)
    if args.ground_truth and (not args.sar_source or not args.rss_source or not args.run):
        p.error("--ground-truth requires --run, --sar-source and --rss-source")
    cfg=AuditConfig.load(args.config)
    m3,rows,provenance=_read_m3(args.m3_root)
    if args.dry_run:
        print(json.dumps({"schema":"saebase.m4.dry_run.v1","audit_id":cfg.config_id,
                          "m3_run_id":m3.run_id,"selected_metrics":list(cfg.metric_ids),
                          "n_m3_metric_rows":len(rows),
                          "evaluation_seeds":list(cfg.evaluation_seeds),
                          "bootstrap_replicates_per_seed":cfg.bootstrap_replicates,
                          "model_weights_loaded":False},indent=2))
        return 0
    if args.report_only:
        report=read_audit(args.m3_root)
        if AuditConfig.from_dict(report["audit_config"]).config_id!=cfg.config_id:
            raise ValueError("--report-only configuration differs from stored M4 audit; use --run")
    else:
        resolver=None
        if args.sae_only:
            from saebase.sae.loading import load_sae_for_profile
            resolver=lambda model_key,sae_key,layer: load_sae_for_profile(
                sae_key,layer,device="cpu")[0]
        report=evaluate_m3(args.m3_root,cfg,sae_resolver=resolver)
        if args.ground_truth:
            from saebase.reliability.ground_truth_report import (
                evaluate_ground_truth, append_ground_truth)
            truth=evaluate_ground_truth(sar_source=args.sar_source,
                                        rss_source=args.rss_source,config=cfg)
            report=append_ground_truth(report,truth)
        write_audit(args.m3_root,report)
    # Rebuild native M3 report with the M4 section. This also verifies that
    # the audit belongs to the unchanged M3 metric ledger.
    from saebase.benchmarks.report import write_report
    from saebase.benchmarks.planner import build_plan
    write_report(args.m3_root,m3,build_plan(m3),rows,provenance=provenance)
    print(json.dumps({"schema":report["schema"],"m3_run_id":m3.run_id,
                      "n_metric_audits":len(report["metric_audits"]),
                      "n_paired_comparisons":len(report["paired_comparisons"]),
                      "model_forwards_during_audit":0,
                      "report":str(args.m3_root/"reliability.html")},indent=2))
    return 0 if report["status"]=="completed" else 1

if __name__=="__main__":
    raise SystemExit(main())
