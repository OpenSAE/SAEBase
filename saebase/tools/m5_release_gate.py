#!/usr/bin/env python3
"""Fail closed before publishing 0.2.0 distribution and checksums."""
from __future__ import annotations
import argparse
from pathlib import Path
from saebase.release.gates import generate_release_evidence
from saebase.benchmarks.results import atomic_json

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-root",type=Path,required=True)
    p.add_argument("--dist",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path("conf/release/m5_matrix.json"))
    p.add_argument("--inventory",type=Path,default=Path("conf/release/m5_asset_provenance.json"))
    p.add_argument("--license",type=Path,default=Path("LICENSE"))
    p.add_argument("--commit",required=True)
    p.add_argument("--output",type=Path,default=Path("release-evidence.json"))
    args=p.parse_args(argv)
    evidence=generate_release_evidence(args.matrix_root,args.dist,args.config,
        args.inventory,args.license,args.commit)
    if evidence["status"]!="passed":
        raise RuntimeError("Release integrity checks are incomplete")
    atomic_json(args.output,evidence)
    digests=evidence["gates"]["distributions"]["artifacts"]
    (args.output.parent/"SHA256SUMS").write_text("".join(
        f"{digest}  {name}\n" for name,digest in sorted(digests.items())),encoding="utf-8")
    print(f"RELEASE INTEGRITY GATE PASSED: {len(digests)} artifacts, "
          f"{evidence['gates']['matrix']['checkpoints']} real SAE checkpoints")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
