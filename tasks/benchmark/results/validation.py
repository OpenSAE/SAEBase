"""Purpose: Validate saved external-method results against their actual source files and scientific protocol.

Scope: the corresponding benchmark execution, result consumer or explicit component check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from saebase.extensions.metrics import validate_external_metric_source
from saebase.extensions.registry import is_external_metric
from tasks.benchmark.execution.schema import BenchmarkConfig
from tasks.benchmark.results.localization import validate_localization_result


def validate_external_sources(
    config: BenchmarkConfig,
    rows: list[dict[str, Any]],
    *,
    root: str | Path | None = None,
    localization_paper=None,
) -> None:
    """Reject stale original-estimator/native results, even on report-only runs.

    Model/SAE weights are not loaded; only the exact trusted local source files
    named in the immutable experiment configuration are fingerprinted.
    """
    from tasks.benchmark.methods.sar import source_identity

    for row in rows:
        if row.get("status") != "completed":
            continue
        name = row.get("metric_id", "")
        if name.startswith(("sar.", "rss.")):
            paper = name.split(".")[0]
            source = config.sources.get(paper)
            if not source or source_identity(source, paper) != row.get("source_sha256"):
                raise ValueError(f"external {paper} source changed after benchmark computation")
        elif name.startswith("localization."):
            if root is None:
                raise ValueError("Localize source validation requires the completed benchmark root")
            validate_localization_result(root, config, row, paper=localization_paper)
        elif is_external_metric(name):
            validate_external_metric_source(name, row, config.sources)
        elif name.startswith("saebench."):
            source = config.sources.get(name)
            if (
                not source
                or not Path(source).is_file()
                or hashlib.sha256(Path(source).read_bytes()).hexdigest() != row.get("source_sha256")
            ):
                raise ValueError(f"native {name} output changed after benchmark computation")
