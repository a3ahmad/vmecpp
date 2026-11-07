"""Require complete executed artifacts before qualifying a paired campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DIAGNOSTICS = ("jxbout", "mercier", "threed1_volumetrics")


def is_digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def numeric_hashes(fields) -> bool:
    return (
        isinstance(fields, dict)
        and bool(fields)
        and all(
            isinstance(value, dict)
            and isinstance(value.get("shape"), list)
            and isinstance(value.get("dtype"), str)
            and is_digest(value.get("sha256"))
            for value in fields.values()
        )
    )


def read_capture(directory: Path) -> dict:
    decks = sorted(directory.glob("actual_indata_*.json"))
    provenance = directory / "provenance.json"
    return {
        "actual_indata": [json.loads(path.read_text()) for path in decks],
        "actual_indata_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in decks
        ],
        "provenance": json.loads(provenance.read_text())
        if provenance.is_file()
        else {},
        "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()
        if provenance.is_file()
        else None,
    }


def complete_diagnostics(report: dict) -> bool:
    diagnostics = report.get("numerical_diagnostic_hashes")
    return (
        isinstance(diagnostics, list)
        and bool(diagnostics)
        and all(
            isinstance(output, dict)
            and all(numeric_hashes(output.get(section)) for section in DIAGNOSTICS)
            for output in diagnostics
        )
    )


def complete_capture(report: dict, capture: dict) -> bool:
    timings = report.get("vmec_wall_seconds", [])
    decks = capture.get("actual_indata", [])
    provenance = capture.get("provenance", {})
    return (
        report.get("status") == "SUCCESS"
        and isinstance(report.get("metrics"), dict)
        and bool(report["metrics"])
        and report.get("profiled") is False
        and bool(timings)
        and len(decks) == len(timings)
        and all(isinstance(deck, dict) and bool(deck) for deck in decks)
        and len(report.get("numerical_wout_hashes", [])) == len(timings)
        and all(numeric_hashes(fields) for fields in report["numerical_wout_hashes"])
        and complete_diagnostics(report)
        and len(report["numerical_diagnostic_hashes"]) == len(timings)
        and is_digest(provenance.get("harness_sha256"))
        and bool(provenance.get("extensions"))
        and bool(provenance.get("packages"))
        and not any(
            value is not None
            for key, value in provenance.get("experimental_runtime", {}).items()
            if key.endswith("_CAPTURE")
            or key
            in (
                "VMECPP_GPU_SOLVER_TRACE",
                "VMECPP_PROFILE_STAGES",
                "VMECPP_PROFILE_STAGES_ONLY",
            )
        )
        and is_digest(capture.get("provenance_sha256"))
        and len(capture.get("actual_indata_sha256", [])) == len(decks)
        and all(is_digest(value) for value in capture["actual_indata_sha256"])
    )


def pair_gate(
    baseline: dict,
    candidate: dict,
    baseline_capture: dict,
    candidate_capture: dict,
    *,
    require_harness: bool = True,
) -> dict:
    captures_complete = complete_capture(
        baseline, baseline_capture
    ) and complete_capture(candidate, candidate_capture)
    identity = all(
        is_digest(baseline.get(key)) and baseline.get(key) == candidate.get(key)
        for key in ("request_metrics_sha256", "metrics_sha256")
    )
    metrics = json.dumps(
        baseline.get("metrics"), sort_keys=True, allow_nan=False
    ) == json.dumps(candidate.get("metrics"), sort_keys=True, allow_nan=False)
    decks = bool(baseline_capture.get("actual_indata")) and json.dumps(
        baseline_capture.get("actual_indata"), sort_keys=True, allow_nan=False
    ) == json.dumps(
        candidate_capture.get("actual_indata"), sort_keys=True, allow_nan=False
    )
    diagnostics = (
        complete_diagnostics(baseline)
        and complete_diagnostics(candidate)
        and baseline["numerical_diagnostic_hashes"]
        == candidate["numerical_diagnostic_hashes"]
    )
    wout = bool(baseline.get("numerical_wout_hashes")) and baseline.get(
        "numerical_wout_hashes"
    ) == candidate.get("numerical_wout_hashes")
    harness = baseline_capture.get("provenance", {}).get(
        "harness_sha256"
    ) == candidate_capture.get("provenance", {}).get("harness_sha256")
    return {
        "exact": bool(
            captures_complete
            and identity
            and metrics
            and decks
            and diagnostics
            and wout
            and (harness or not require_harness)
        ),
        "captures_complete": captures_complete,
        "identity_exact": identity and metrics,
        "actual_indata_exact": decks,
        "diagnostics_exact": diagnostics,
        "wout_exact": wout,
        "harness_exact": harness,
    }


def clean_environment(environment: dict[str, str]) -> dict[str, str]:
    result = {
        key: value
        for key, value in environment.items()
        if not key.startswith("VMECPP_")
        and key
        not in (
            "LD_PRELOAD",
            "CGA_PROFILE",
            "CUDA_INJECTION64_PATH",
            "NVTX_INJECTION64_PATH",
            "CGA_OPTIMIZATION_WRAPPER_SHA256",
            "CGA_OPTIMIZATION_CONFIG_SHA256",
            "CGA_OPTIMIZATION_CONFIGURATION_SHA256",
        )
    }
    result.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    return result


def audit_case(
    records: list[dict],
    artifact_root: Path,
    *,
    require_original_reference: bool = False,
) -> dict:
    """Audit warmups and frozen configuration separately from timed samples."""
    exact = True
    warmup_complete = False
    original_checks = []
    reference_hashes = set()
    configurations = {label: set() for label in ("baseline", "candidate")}
    outputs = {label: set() for label in configurations}
    decks = {label: set() for label in configurations}
    for number in sorted({record["pair"] for record in records}):
        current = [record for record in records if record["pair"] == number]
        labels = {record["label"]: record for record in current}
        if len(current) != 2 or labels.keys() != configurations.keys():
            exact = False
            continue
        if any(record["warmup"] != (number == 0) for record in current):
            exact = False
        captured = {}
        for label, record in labels.items():
            directory = (
                artifact_root
                / label
                / f"{Path(record['seed']).stem}-{record['track']}-{number:02d}"
            )
            capture = record.get("captures") or read_capture(directory)
            captured[label] = capture
            provenance = capture.get("provenance", {})
            if "captures" in record and not all(
                is_digest(provenance.get(key))
                for key in ("campaign_wrapper_sha256", "campaign_configuration_sha256")
            ):
                exact = False
            configurations[label].add(
                json.dumps(
                    {
                        key: provenance.get(key)
                        for key in (
                            "harness_sha256",
                            "campaign_wrapper_sha256",
                            "campaign_configuration_sha256",
                            "extensions",
                            "packages",
                            "cpu_affinity",
                            "thread_environment",
                            "experimental_runtime",
                            "preloaded_libraries",
                        )
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            decks[label].add(
                json.dumps(
                    capture.get("actual_indata"), sort_keys=True, allow_nan=False
                )
            )
            report = record.get("report", {})
            outputs[label].add(
                json.dumps(
                    {
                        key: report.get(key)
                        for key in (
                            "request_metrics_sha256",
                            "metrics_sha256",
                            "metrics",
                            "numerical_wout_hashes",
                            "numerical_diagnostic_hashes",
                        )
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            reference = record.get("original_reference")
            if reference is not None:
                good = bool(reference.get("gate", {}).get("exact")) and is_digest(
                    reference.get("report_sha256")
                )
                original_checks.append(good)
                reference_hashes.add(reference.get("report_sha256"))
                exact = exact and good
            elif require_original_reference:
                original_checks.append(False)
                exact = False
        gate = pair_gate(
            labels["baseline"].get("report", {}),
            labels["candidate"].get("report", {}),
            captured["baseline"],
            captured["candidate"],
        )
        good = gate["exact"] and all(record["exit_status"] == 0 for record in current)
        exact = exact and good
        if number == 0:
            warmup_complete = good and all(record["warmup"] for record in current)
    repeated = {
        label: len(configurations[label])
        == len(decks[label])
        == len(outputs[label])
        == 1
        for label in configurations
    }
    original_exact = (
        bool(original_checks)
        and len(original_checks) == len(records)
        and all(original_checks)
        and len(reference_hashes) == 1
    )
    return {
        "exact": exact
        and warmup_complete
        and all(repeated.values())
        and (original_exact if original_checks or require_original_reference else True),
        "warmup_complete": warmup_complete,
        "repeat_configuration": repeated,
        "original_reference_checked": len(original_checks) == len(records),
        "original_reference_exact": original_exact if original_checks else None,
    }
