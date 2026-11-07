"""Summarize actual reports, preserving exact metric parity as a separate gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def metric_drift(reference: dict, candidate: dict) -> dict:
    """Keep missing/null values distinct and never hide drift behind rounding."""
    result = {}
    for name in sorted(reference.keys() | candidate.keys()):
        if name not in reference or name not in candidate:
            result[name] = {"exact": False, "reason": "missing metric"}
            continue
        baseline, actual = reference[name], candidate[name]
        if baseline is None or actual is None:
            result[name] = {"exact": baseline is actual, "relative": None}
            continue
        absolute = abs(actual - baseline)
        result[name] = {
            "exact": float(actual).hex() == float(baseline).hex(),
            "absolute": absolute,
            "relative": absolute / abs(baseline)
            if baseline
            else (0 if not actual else None),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reports = [
        json.loads(path.read_text()) for path in sorted(args.root.rglob("report.json"))
    ]
    references = {}
    reference_reports = (
        [
            json.loads(path.read_text())
            for path in sorted(args.reference.rglob("report.json"))
        ]
        if args.reference
        else reports
    )
    for report in reference_reports:
        if report["status"] == "SUCCESS":
            references.setdefault((report["seed"], report["track"]), report)
    rows = []
    for key in sorted({(r["seed"], r["track"]) for r in reports}):
        samples = [r for r in reports if (r["seed"], r["track"]) == key]
        good = [r for r in samples if r["status"] == "SUCCESS"]
        reference = references.get(key)
        drifts = (
            [metric_drift(reference["metrics"], r["metrics"]) for r in good]
            if reference is not None
            else []
        )
        maximum_drift = {}
        if reference is not None:
            for name in reference["metrics"]:
                numeric = [d[name] for d in drifts if "absolute" in d[name]]
                maximum_drift[name] = {
                    "absolute": max((d["absolute"] for d in numeric), default=None),
                    "relative": max(
                        (d["relative"] for d in numeric if d["relative"] is not None),
                        default=None,
                    ),
                }
        rows.append(
            {
                "seed": key[0],
                "track": key[1],
                "executed": len(samples),
                "successful": len(good),
                "measurement": "derived from measured reports; warmups included",
                "adapter_median_seconds": statistics.median(
                    r["adapter_wall_seconds"] for r in good
                )
                if good
                else None,
                "metrics_exact": reference is not None
                and bool(good)
                and len(good) == len(samples)
                and all(
                    r["metrics_sha256"] == reference["metrics_sha256"] for r in good
                ),
                "request_metrics_exact": reference is not None
                and bool(good)
                and len(good) == len(samples)
                and all(
                    r["request_metrics_sha256"] == reference["request_metrics_sha256"]
                    for r in good
                ),
                "per_sample_drift": drifts,
                "maximum_drift_per_metric": maximum_drift,
            }
        )
    print(json.dumps(rows, indent=2, allow_nan=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
