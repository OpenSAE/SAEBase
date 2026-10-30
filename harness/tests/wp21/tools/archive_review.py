"""Archive WP21 independent reviewer provenance without changing final grades.

The archive contains the review record, check assertions, process logs and the
actual independent reviewer scripts by SHA256. It excludes candidate source,
checkpoint weights, and large tensor artifacts. Keep the archive outside GitHub
until a human has checked logs and path disclosures; the original local evidence
remains authoritative. Run this only from a trusted examination checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import canonical_hash, contained, read_json, relative_name, sha256
from review import _validate_review_records

SCHEMA = "saebase.wp21.reviewer_archive.v1"


def _zip_bytes(entries: dict[str, bytes], archive: Path) -> None:
    # Stable timestamps and sorted names make archives reproducible.
    with zipfile.ZipFile(archive, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for name, payload in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            out.writestr(info, payload)


def create_archive(review_dir: Path, output: Path, *, max_total_mib: int = 256) -> dict:
    review_dir = review_dir.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_relative_to(review_dir):
        raise ValueError("review archive must be a NEW file outside the reviewer workspace")
    if not isinstance(max_total_mib, int) or max_total_mib < 1:
        raise ValueError("positive archive size budget required")
    state = read_json(contained(review_dir, "review.json", exists=True))
    final = read_json(contained(review_dir, "review_final.json", exists=True))
    if final.get("schema") != "saebase.wp21.final_review.v1" or final.get("case_id") != state.get("case_id") or final.get("run_id") != state.get("run_id"):
        raise ValueError("review final identity does not match review state")
    _validate_review_records(review_dir, state)
    files: dict[str, bytes] = {}
    for name in ("review.json", "review_final.json"):
        files[name] = contained(review_dir, name, exists=True).read_bytes()
    records = review_dir / "records"
    if records.exists():
        if records.is_symlink() or not records.is_dir():
            raise ValueError("review records must be a regular directory")
        for item in records.rglob("*"):
            relative = item.relative_to(review_dir).as_posix()
            contained(review_dir, relative, exists=True)
            if item.is_file():
                files[relative] = item.read_bytes()
    script_count = 0
    for check in state["checks"]:
        for attempt in check["attempts"]:
            proof = attempt.get("independent_script")
            if not proof:
                continue
            candidate = Path(proof["path"])
            if candidate.is_symlink() or not candidate.is_file() or sha256(candidate) != proof["sha256"]:
                raise ValueError("reviewer script no longer matches original proof")
            if candidate.resolve().is_relative_to(review_dir / "workspace") or candidate.resolve().is_relative_to(Path(state["candidate_workspace"])):
                raise ValueError("reviewer script belongs to untrusted candidate")
            name = f"reviewer_scripts/{proof['sha256']}.py"
            if name not in files:
                files[name] = candidate.read_bytes()
                script_count += 1
    # Match the exact files referenced by the reviewer check records. Merely
    # having a final.json is not sufficient to establish any check as verified.
    total = sum(len(payload) for payload in files.values())
    if total > max_total_mib * (1 << 20):
        raise ValueError("review provenance exceeds archival limit")
    index = [{"path":name,"size_bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()}
             for name,payload in sorted(files.items())]
    manifest = {"schema":SCHEMA,"case_id":state["case_id"],"run_id":state["run_id"],
                "review_final_status":final["status"],"files":index,"files_sha256":canonical_hash(index),
                "independent_script_count":script_count,
                "excludes":["candidate source/workspaces", "checkpoint weights", "large experiment tensors"],
                "warning":"Local provenance archive: check stdout/stderr and machine paths for private information before publishing. Neither the ZIP nor the hashes independently prove the computation."}
    files["archive_manifest.json"] = (json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    _zip_bytes(files, output)
    return {"status":"REVIEW_PROVENANCE_ARCHIVED_NOT_REPLAYED", "path":str(output),
            "sha256":sha256(output),"case_id":state["case_id"],"files":len(index),
            "independent_scripts":script_count}


def verify_archive(archive: Path) -> dict:
    archive = archive.resolve(strict=True)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or "archive_manifest.json" not in names:
            raise ValueError("duplicate or missing archive member")
        for name in names:
            relative_name(name)
        manifest = json.loads(z.read("archive_manifest.json"))
        if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"),list):
            raise ValueError("invalid reviewer archive manifest")
        expected = {x["path"] for x in manifest["files"]}
        if set(names) != expected | {"archive_manifest.json"}:
            raise ValueError("archive contains missing or unlisted members")
        if manifest.get("files_sha256") != canonical_hash(manifest["files"]):
            raise ValueError("archive member manifest changed")
        for item in manifest["files"]:
            payload=z.read(item["path"])
            if len(payload)!=item["size_bytes"] or hashlib.sha256(payload).hexdigest()!=item["sha256"]:
                raise ValueError("archive file checksum mismatch: "+item["path"])
        if z.testzip() is not None:
            raise ValueError("corrupted archive member")
        final = json.loads(z.read("review_final.json"))
        state = json.loads(z.read("review.json"))
        if (final.get("case_id"),final.get("run_id"),final.get("status")) != (manifest.get("case_id"),manifest.get("run_id"),manifest.get("review_final_status")):
            raise ValueError("archived final does not match archive identity")
        if (state.get("case_id"),state.get("run_id"))!=(final.get("case_id"),final.get("run_id")):
            raise ValueError("archived review and final identities differ")
        return {"status":"REVIEW_ARCHIVE_INTEGRITY_VALID_NOT_REPLAYED", "sha256":sha256(archive),
                "case_id":final["case_id"], "run_id":final["run_id"],
                "final_status":final["status"], "files":len(manifest["files"])}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    create=subs.add_parser("create")
    create.add_argument("--review",type=Path,required=True)
    create.add_argument("--output",type=Path,required=True)
    create.add_argument("--max-total-mib",type=int,default=256)
    verify=subs.add_parser("verify")
    verify.add_argument("--archive",type=Path,required=True)
    a=parser.parse_args()
    try:
        result=create_archive(a.review,a.output,max_total_mib=a.max_total_mib) if a.command=="create" else verify_archive(a.archive)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        print(json.dumps({"status":"REVIEW_ARCHIVE_ERROR","reason":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
