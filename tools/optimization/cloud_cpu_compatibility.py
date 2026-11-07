"""Select CPU dispatch from exact saved-output replays, never relax identity.

This selection only permits the expensive official check to start. It does not qualify a
build or replace any original metric, deck, output or repeat gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

VARIANT_ORDER = (
    "default",
    "blas-haswell",
    "numpy-no-avx512",
    "haswell-no-avx512",
)


def validate_environment(variant: object, environment: object) -> dict[str, str]:
    """Accept only the declared CPU-dispatch alternatives."""
    if (
        not isinstance(variant, str)
        or variant not in VARIANT_ORDER
        or not isinstance(environment, dict)
    ):
        msg = "Unknown CPU compatibility variant"
        raise ValueError(msg)
    expected = set()
    if variant in ("blas-haswell", "haswell-no-avx512"):
        expected.add("OPENBLAS_CORETYPE")
    if variant in ("numpy-no-avx512", "haswell-no-avx512"):
        expected.add("NPY_DISABLE_CPU_FEATURES")
    if set(environment) != expected:
        msg = "Compatibility selection may change only declared CPU dispatch"
        raise ValueError(msg)
    if (
        "OPENBLAS_CORETYPE" in expected
        and environment["OPENBLAS_CORETYPE"] != "Haswell"
    ):
        msg = "Only the measured Haswell BLAS alternative is permitted"
        raise ValueError(msg)
    if "NPY_DISABLE_CPU_FEATURES" in expected:
        disabled = environment["NPY_DISABLE_CPU_FEATURES"]
        if not isinstance(disabled, str) or len(disabled) > 1024:
            msg = "Invalid NumPy feature list"
            raise ValueError(msg)
        features = disabled.split(",")
        if (
            not 1 <= len(features) <= 32
            or len(set(features)) != len(features)
            or any(not re.fullmatch(r"AVX512[A-Z0-9_]*", name) for name in features)
        ):
            msg = "Only explicit AVX512 dispatch features may be disabled"
            raise ValueError(msg)
    return dict(environment)


def select_environment(summary: object) -> dict:
    """Prefer exact intermediates, then the predeclared variant order."""
    if not isinstance(summary, dict):
        msg = "Missing CPU replay summary"
        raise ValueError(msg)
    records = summary.get("records")
    requested = summary.get("requested_variants")
    if (
        not isinstance(records, list)
        or not isinstance(requested, list)
        or not 1 <= len(requested) <= 4
        or any(name not in VARIANT_ORDER for name in requested)
        or len(set(requested)) != len(requested)
        or summary.get("requested_repeats") != 2
    ):
        msg = "Expected declared variants with two fresh CPU replay processes each"
        raise ValueError(msg)
    eligible = []
    grouped = {}
    for row in records:
        if not isinstance(row, dict):
            msg = "Invalid CPU replay record"
            raise ValueError(msg)
        variant = row.get("variant")
        environment = validate_environment(variant, row.get("environment_overrides"))
        repeat = row.get("repeat_index")
        key = (variant, repeat)
        if type(repeat) is not int or repeat not in (0, 1) or key in grouped:
            msg = "Invalid or duplicate CPU replay process"
            raise ValueError(msg)
        grouped[key] = (row, environment)
    if set(grouped) != {(name, repeat) for name in requested for repeat in (0, 1)}:
        msg = "Missing or unexpected CPU replay process; no partial qualification"
        raise ValueError(msg)
    for variant in requested:
        first, environment = grouped[(variant, 0)]
        second, repeated_environment = grouped[(variant, 1)]
        if environment != repeated_environment:
            msg = "Repeat processes must use the same CPU dispatch"
            raise ValueError(msg)
        required = (
            "job_passed",
            "owned_children_reaped",
            "metric_exact",
            "equilibrium_exact",
            "source_exact",
        )
        repeated = second.get("repeat_against_first", {})
        if not isinstance(repeated, dict):
            msg = "Invalid CPU repeat comparison"
            raise ValueError(msg)
        if all(
            row.get(name) is True for row in (first, second) for name in required
        ) and all(
            repeated.get(name) is True
            for name in ("metric_exact", "equilibrium_exact", "source_exact")
        ):
            strong = all(
                row.get("intermediates_exact") is True
                for row in (first, second, repeated)
            )
            eligible.append((variant, environment, strong))
    if not eligible:
        msg = "No exact original-metric CPU compatibility replay"
        raise ValueError(msg)
    variant, environment, strong = min(
        eligible,
        key=lambda item: (
            not item[2],
            VARIANT_ORDER.index(item[0]),
        ),
    )
    return {
        "phase": "CPU_REPLAY_COMPATIBILITY_SELECTED_ONLY",
        "variant": variant,
        "environment": environment,
        "intermediates_exact": strong,
        "full_official_all_metric_identity_required": True,
        "main_table_eligible": False,
    }


def decision_from_file(summary_path: Path) -> dict:
    """Bind a deterministic selection to the actual replay receipt bytes."""
    data = summary_path.read_bytes()
    decision = select_environment(json.loads(data))
    decision["probe_summary_sha256"] = hashlib.sha256(data).hexdigest()
    return decision


def read_selection(selection_path: Path, summary_path: Path) -> dict[str, str]:
    """Refuse stale or edited decisions before starting an official process."""
    actual = json.loads(selection_path.read_text())
    if actual != decision_from_file(summary_path):
        msg = "CPU compatibility decision does not match actual replay evidence"
        raise ValueError(msg)
    return actual["environment"]
