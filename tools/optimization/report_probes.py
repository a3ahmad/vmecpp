"""Summarize experimental parity without promoting unpaired smoke timings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from summarize import metric_drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    samples = json.loads(args.samples.read_text())
    groups = defaultdict(list)
    for sample in samples:
        label = sample.get("label", f"fourier_threads_{sample.get('threads')}")
        groups[label].append(sample)
    result = []
    for label, group in groups.items():
        reports = [r["report"] for r in group if r["exit_status"] == 0]
        result.append(
            {
                "configuration": label,
                "classification": "experimental probe; no qualified speedup",
                "executed_samples": len(group),
                "successful_samples": len(reports),
                "metrics_exact_vs_baseline": bool(reports)
                and all(
                    r["metrics_sha256"] == baseline["metrics_sha256"] for r in reports
                ),
                "all_numerical_fields_exact_vs_baseline": bool(reports)
                and all(
                    r["numerical_wout_hashes"] == baseline["numerical_wout_hashes"]
                    for r in reports
                ),
                "repeat_deterministic": len(reports) >= 2
                and len(
                    {
                        json.dumps(
                            [r["metrics"], r["numerical_wout_hashes"]], sort_keys=True
                        )
                        for r in reports
                    }
                )
                == 1,
                "metric_drift_each_sample": [
                    metric_drift(baseline["metrics"], r["metrics"]) for r in reports
                ],
                "measured_adapter_seconds": [
                    r["adapter_wall_seconds"] for r in reports
                ],
                "timing_limit": "Unpaired probes; contention and order not controlled",
            }
        )
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
