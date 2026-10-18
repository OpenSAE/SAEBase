"""Two complete six-factor input designs for controlled localization experiments.

This module only creates project-authored input texts and an optional
synthetic mathematical control. It never runs a model, loads SAE weights,
or presents synthetic feature activations as genuine checkpoint results.

The exact state ordering and factor identities are compatible with the
existing SAEBase ControlledInputDesign and the original Localize estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .core import canonical_json
from .ingest import _atomic_bytes

SCHEMA = "opensae.controlled_six_factor_inputs.v1"
DESIGNS: dict[str, tuple[tuple[str, tuple[str, str]], ...]] = {
    "synthetic_known_factor_control": (
        ("factor_a", ("absent", "present")),
        ("factor_b", ("quiet", "active")),
        ("factor_c", ("low", "high")),
        ("factor_d", ("closed", "open")),
        ("factor_e", ("negative", "positive")),
        ("factor_f", ("false", "true")),
    ),
    "real_model_six_factor_design": (
        ("setting", ("coastal town", "mountain village")),
        ("season", ("spring", "autumn")),
        ("event", ("community exhibition", "local science fair")),
        ("venue", ("library hall", "school auditorium")),
        ("voice", ("first person", "third person")),
        ("format", ("one sentence", "two bullet points")),
    ),
}


def _render(name: str, values: dict[str, str]) -> str:
    """Represent every binary factor independently and without external facts."""
    if name == "synthetic_known_factor_control":
        return (
            "This is a controlled six-factor test input.\n"
            "The values below are arbitrary and have no real-world meaning.\n"
            + "".join(f"{key}: {value}\n" for key, value in values.items())
            + "\nInput record complete.\n"
        )
    if name == "real_model_six_factor_design":
        return (
            "Write a fictional event notice. Do not make factual claims "
            "about an existing place or event.\n"
            + "".join(f"{key}: {value}\n" for key, value in values.items())
            + "\nNotice:\n"
        )
    raise ValueError("Unknown controlled input design")


def make_candidate(name: str) -> dict[str, Any]:
    """Create exactly one unique sample for each bitmask, in 0..63 order."""
    if name not in DESIGNS:
        raise ValueError("Unknown controlled input design")
    factors = DESIGNS[name]
    if len(factors) != 6 or len({factor[0] for factor in factors}) != 6:
        raise ValueError("Expected six independent named binary factors")
    states: list[dict[str, Any]] = []
    for mask in range(64):
        bits = [(mask >> bit) & 1 for bit in range(6)]
        selected = {factor_name: levels[bits[i]] for i, (factor_name, levels) in enumerate(factors)}
        prompt = _render(name, selected)
        states.append(
            {
                "mask": mask,
                "bits": bits,
                "levels": selected,
                "sample_id": f"controlled6:{name}:{mask:02d}",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "token_position": -1,
            }
        )
    if len({row["prompt"] for row in states}) != 64:
        raise ValueError("Different controlled states must produce distinct texts")
    payload = {
        "schema": SCHEMA,
        "design_name": name,
        "source_type": "project_authored_controlled_input",
        "status": "input_only_no_model_execution",
        "independent_group_count": 1,
        "prompt_group_id": "controlled6:" + name,
        "n_factors": 6,
        "n_states": 64,
        "state_order": "integer_bitmask_0_to_63_factor_zero_least_significant",
        "distribution": "uniform_binary_product_when_sampled_as_defined",
        "factors": [
            {"name": factor_name, "levels": list(levels)}
            for factor_name, levels in factors
        ],
        "states": states,
        "scientific_scope": (
            "Complete enumerated input design; token positions, SAE activations, "
            "causal independence and measured localization remain unverified."
        ),
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def check_candidate(document: dict[str, Any]) -> dict[str, Any]:
    """Validate full state content against its immutable authored renderer."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("Unknown controlled design schema")
    name = document.get("design_name")
    if name not in DESIGNS:
        raise ValueError("Unexpected design name")
    expected = make_candidate(name)
    if document != expected:
        raise ValueError("Controlled design contents or manifest checksum mismatch")
    return expected


def to_saebase_design(document: dict[str, Any]) -> Any:
    """Convert to the existing original SAEBase type, without rewriting it."""
    checked = check_candidate(document)
    from saebase.data import BinaryFactor, ControlledInputDesign

    factors = tuple(
        BinaryFactor(factor["name"], tuple(factor["levels"]))
        for factor in checked["factors"]
    )
    design = ControlledInputDesign.from_prompts(
        factors=factors,
        name=checked["design_name"],
        prompts_by_state={row["mask"]: row["prompt"] for row in checked["states"]},
        sample_ids_by_state={row["mask"]: row["sample_id"] for row in checked["states"]},
        positions_by_state={row["mask"]: row["token_position"] for row in checked["states"]},
    )
    if design.n_states != 64 or design.n_factors != 6:
        raise ValueError("Conversion changed the full grid")
    return design


def synthetic_reference_codes(document: dict[str, Any]) -> dict[str, Any]:
    """Known logical feature values for a *synthetic* test, never SAE codes."""
    checked = check_candidate(document)
    if checked["design_name"] != "synthetic_known_factor_control":
        raise ValueError("Synthetic truth only belongs to the synthetic control")
    values = []
    for record in checked["states"]:
        a, b, c, d, e, f = record["bits"]
        values.append(
            [
                float(a),
                float(b + c),
                float(d * e),
                float(f),
                float(a ^ c),
                float((b * d) + e),
            ]
        )
    return {
        "schema": "opensae.synthetic_known_factor_codes.v1",
        "source_type": "synthetic_logic_not_real_sae",
        "design_manifest_sha256": checked["manifest_sha256"],
        "n_states": 64,
        "n_synthetic_features": 6,
        "row_order": "state_mask_0_to_63",
        "feature_values": values,
        "interpretation": (
            "Deterministic synthetic truth for testing original Localize math; "
            "not a learned-feature experiment."
        ),
    }


def write_candidate(name: str, output: Path) -> dict[str, Any]:
    """Create one immutable input file; never overwrite a previous candidate."""
    if output.exists():
        raise FileExistsError("Controlled input release already exists")
    item = make_candidate(name)
    _atomic_bytes(output, canonical_json(item))
    if check_candidate(json.loads(output.read_text("utf-8"))) != item:
        raise ValueError("Controlled input file failed its own verification")
    return {
        "design": name,
        "path": str(output),
        "n_states": 64,
        "manifest_sha256": item["manifest_sha256"],
        "model_inferences": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Destination for two immutable 64-state input grids"
    )
    args = parser.parse_args(argv)
    outputs = [
        args.output_dir / (name + ".json")
        for name in DESIGNS
    ]
    if any(path.exists() for path in outputs):
        parser.error("At least one design file already exists; refusing partial overwrite")
    reports = [
        write_candidate(name, path)
        for name, path in zip(DESIGNS, outputs)
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
