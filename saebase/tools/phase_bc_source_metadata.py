"""Remote source file and SHA metadata checks without downloading SAE tensors.

This is an upstream existence check; it is never a real-weight acceptance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from saebase.sae.coverage import CoverageEntry, load_coverage

CATALOG = Path(__file__).resolve().parents[2] / "conf" / "coverage" / "stage_bc.json"


def probe_file_path(entry: CoverageEntry, layer: int) -> str | None:
    p = entry.source(layer)
    key = entry.sae_key
    if p["loader"] == "qwen_resid_topk":
        return p["source_file_id"]
    if p["loader"] in ("eleutherai_topk_mlp", "sparsify_topk"):
        return p["source_file_id"].rstrip("/") + "/sae.safetensors"
    if p["loader"] == "dictionary_learning":
        return p["source_file_id"].rstrip("/") + "/ae.pt"
    if key.startswith("gemma3_4b_") or key in {
        "gemma270m_scope_resid",
        "gemma1b_scope_resid",
    }:
        return p["source_file_id"].rstrip("/") + "/params.safetensors"
    if key == "gemma2_2b_pt_l25_canonical":
        from saebase.sae.registry import get_profile

        physical = get_profile(key).resolved_physical_path
        if physical:
            return physical.rstrip("/") + "/params.npz"
    if key == "gemma2_9b_pt_scope_resid":
        return p["source_file_id"].rstrip("/") + "/params.npz"
    if key == "llama31_8b_lxr_8x":
        return p["source_file_id"].rstrip("/") + "/checkpoints/final.safetensors"
    if key.startswith("r1_distill_llama8b_llamascope"):
        return p["source_file_id"].rstrip("/") + "/sae_weights.safetensors"
    if key == "gemma2_2b_it_matryoshka_l12_32k":
        return "standard/blocks.12.hook_resid_post/sae_weights.safetensors"
    if key == "qwen35_4b_matryoshka_l15_65k":
        return "qwen-3.5-4b/btk-mat-layer-15-k-100/sae_weights.safetensors"
    if key == "mistral7b_instruct_tc_l16_131k":
        return "sae_weights.safetensors"
    # SAE Lens alias resolution may have a different physical folder and
    # filename. It will be validated by the genuine CPU SAE loader instead.
    return None


def check_sources(catalog_path: str | Path = CATALOG) -> dict:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    catalog = load_coverage(catalog_path)
    records = []
    for entry in catalog.entries:
        for layer in entry.selected_layers:
            item = entry.source(layer)
            file = probe_file_path(entry, layer)
            data = {
                "sae_key": entry.sae_key,
                "layer": layer,
                "repo_id": item["repo_id"],
                "source_file": file,
                "status": "unresolved_path" if file is None else "pending",
            }
            if file:
                try:
                    metadata = get_hf_file_metadata(
                        hf_hub_url(repo_id=item["repo_id"], filename=file)
                    )
                    data.update(
                        {
                            "status": "source_file_found",
                            "size_bytes": metadata.size,
                            "etag": metadata.etag,
                            "commit_hash": metadata.commit_hash,
                        }
                    )
                except Exception as exc:
                    data.update(
                        {
                            "status": "source_check_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:350],
                        }
                    )
            records.append(data)
    return {
        "schema": "saebase.coverage.source_metadata.v1",
        "kind": "upstream_file_presence_not_weight_validation",
        "total": len(records),
        "found": sum(x["status"] == "source_file_found" for x in records),
        "unresolved": sum(x["status"] == "unresolved_path" for x in records),
        "failed": sum(x["status"] == "source_check_failed" for x in records),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": records,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output", type=Path, default=Path("stage-bc-source-metadata.json"))
    args = parser.parse_args(argv)
    result = check_sources(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2))
    for entry in result["entries"]:
        if entry["status"] == "source_check_failed":
            print(
                f"CHECK FAILED: {entry['sae_key']} L{entry['layer']}: {entry['error_type']}: {entry['error'][:100]}"
            )
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
