"""Append QA500 as *new* configs to OpenSAE/SAEBase-Evaluation after approval.

Never edit historic results or overwrite the existing 25 HF subsets. Parent
commit guards prevent an agent from silently overwriting concurrent changes.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import re
from pathlib import Path
from typing import Any

from experiments.qa500.report import CONFIGS, DEFAULT_EXPORT
from experiments.qa500.cohort import PROTOCOL_ID
from experiments.qa500.assets import required_publication_lock

DESTINATION = "OpenSAE/SAEBase-Evaluation"


def _update_card(original: str) -> str:
    import yaml

    parts = original.split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        raise RuntimeError("Existing HF card lacks standard YAML frontmatter")
    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        raise RuntimeError("Existing HF card metadata is not a dictionary")
    old = meta.get("configs")
    if not isinstance(old, list) or len(old) < 25:
        raise RuntimeError("Expected 25 or more existing evaluation subsets")
    names = {x.get("config_name") for x in old if isinstance(x, dict)}
    if any(n in names for n in CONFIGS):
        raise RuntimeError("QA500 dataset configs already exist; refuse overwrite")
    for name in CONFIGS:
        old.append(
            {
                "config_name": name,
                "data_files": [{"split": "train", "path": f"qa_v1/{name}/data.csv"}],
            }
        )
    updated_meta = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    body = parts[2]
    addendum = (
        "\n\n## QA500 frozen zero-shot evaluation cohort\n\n"
        "This independent cohort contains ten upstream QA sources with 500 "
        "questions each, ten models and twenty SAE layer sites. It reports "
        "only exact source identifiers and verified real-model measurements "
        "from the QA500 pipeline. Previous Stage D and Research Lab cohorts "
        "remain unchanged. Some task families require separate native "
        "evaluation or have explicitly unavailable outputs. No restricted "
        "GPQA question text is redistributed.\n\n"
        "Protocol: `qa_standard_500_v1`. Software: OpenSAE/SAEBase WP22. "
        "See the QA500 publication manifest for evidence and omissions.\n"
    )
    return "---\n" + updated_meta + "---" + body + addendum


def publish(
    *,
    stage: Path,
    approved: bool,
    target: str = DESTINATION,
) -> dict[str, Any]:
    if not approved:
        raise PermissionError("HF publication requires explicit approval")
    if not required_publication_lock():
        raise RuntimeError("Project dataset/model/SAE asset revisions are not fully pinned")
    if target != DESTINATION:
        raise ValueError("Cannot publish QA500 to an unrelated HF repository")
    manifest_path = stage / "publication_manifest.json"
    evidence = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        evidence.get("protocol_id") != PROTOCOL_ID
        or evidence.get("status") != "complete"
        or evidence.get("source_text_included") is not False
        or evidence.get("number_of_subsets") != len(CONFIGS)
    ):
        raise RuntimeError("HF publication gate rejected incomplete cohort evidence")
    import hashlib

    for name in CONFIGS:
        path = stage / name / "data.csv"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != (
            evidence["file_sha256"].get(name)
        ):
            raise RuntimeError(f"Staging file changed after audit: {name}")
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

    api = HfApi()
    info = api.dataset_info(target)
    parent = info.sha
    if not re.fullmatch(r"[0-9a-f]{40}", str(parent)):
        raise RuntimeError("Cannot pin current HF target commit")
    card_file = hf_hub_download(
        repo_id=target,
        filename="README.md",
        repo_type="dataset",
        revision=parent,
    )
    original_card = Path(card_file).read_text(encoding="utf-8")
    import yaml

    original_meta = yaml.safe_load(original_card.split("---", 2)[1])
    old_names = [entry["config_name"] for entry in original_meta["configs"]]
    card = _update_card(original_card)
    operations = [
        CommitOperationAdd(
            path_in_repo=f"qa_v1/{name}/data.csv",
            path_or_fileobj=str(stage / name / "data.csv"),
        )
        for name in CONFIGS
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo="qa_v1/publication_manifest.json",
            path_or_fileobj=str(manifest_path),
        )
    )
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=BytesIO(card.encode("utf-8")),
        )
    )
    commit = api.create_commit(
        repo_id=target,
        repo_type="dataset",
        operations=operations,
        commit_message="Add independently audited QA500 evaluation cohort",
        parent_commit=parent,
    )
    return {
        "status": "published",
        "repository": target,
        "source_parent_commit": parent,
        "commit": str(commit.commit_url),
        "new_subset_names": list(CONFIGS),
        "existing_config_count_before": len(old_names),
        "new_config_count": len(CONFIGS),
        "historical_data_paths_modified": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gated QA500 HF publication")
    parser.add_argument("--stage", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(publish(stage=args.stage, approved=args.approved), indent=2))


if __name__ == "__main__":
    main()
