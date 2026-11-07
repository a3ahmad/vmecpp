"""Report paired timings with warmups excluded and exact-parity gates."""

from __future__ import annotations

# CLI progress is intentionally emitted.
# ruff: noqa: T201
import argparse
import json
import random
import statistics
from pathlib import Path

if __package__:
    from .campaign_gates import (
        audit_case,
        complete_diagnostics,
        pair_gate,
        read_capture,
    )
    from .summarize import metric_drift
else:
    from campaign_gates import (
        audit_case,
        complete_diagnostics,
        pair_gate,
        read_capture,
    )
    from summarize import metric_drift


def interval(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    generator = random.Random(0)
    ratios = []
    for _ in range(2000):
        draw = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        ratios.append(
            statistics.median(p[0] for p in draw)
            / statistics.median(p[1] for p in draw)
        )
    ratios.sort()
    return ratios[49], ratios[1949]


def summarize_pairs(
    samples: list[dict],
    artifact_root: Path,
    *,
    require_original_reference: bool = False,
) -> list[dict]:
    rows = []
    for seed, track in sorted({(r["seed"], r["track"]) for r in samples}):
        case = [row for row in samples if row["seed"] == seed and row["track"] == track]
        audit = audit_case(
            case, artifact_root, require_original_reference=require_original_reference
        )
        selected = [
            r
            for r in samples
            if r["seed"] == seed and r["track"] == track and not r["warmup"]
        ]
        pairs = []
        vmec_pairs = []
        exact = audit["exact"]
        decks_exact = True
        captures_complete = True
        drifts = []
        identities = {"baseline": set(), "candidate": set()}
        numerical_identities = {"baseline": set(), "candidate": set()}
        for number in sorted({r["pair"] for r in selected}):
            records = {r["label"]: r for r in selected if r["pair"] == number}
            if (
                records.keys() != {"baseline", "candidate"}
                or sum(r["pair"] == number for r in selected) != 2
            ):
                exact = False
                continue
            a, b = records["baseline"], records["candidate"]
            if a["exit_status"] != 0 or b["exit_status"] != 0:
                exact = False
                continue
            captured = {}
            for label, record in records.items():
                directory = (
                    artifact_root / label / f"{Path(seed).stem}-{track}-{number:02d}"
                )
                captured[label] = record.get("captures") or read_capture(directory)
            a, b = a["report"], b["report"]
            gate = pair_gate(a, b, captured["baseline"], captured["candidate"])
            exact = exact and gate["exact"]
            decks_exact = decks_exact and gate["actual_indata_exact"]
            captures_complete = captures_complete and gate["captures_complete"]
            for label, report in (("baseline", a), ("candidate", b)):
                identities[label].add(report["request_metrics_sha256"])
                numerical_identities[label].add(
                    json.dumps(
                        {
                            "wout": report["numerical_wout_hashes"],
                            "diagnostics": report.get("numerical_diagnostic_hashes"),
                        },
                        sort_keys=True,
                    )
                )
            exact = exact and a["request_metrics_sha256"] == b["request_metrics_sha256"]
            exact = exact and a["numerical_wout_hashes"] == b["numerical_wout_hashes"]
            exact = exact and a.get("numerical_diagnostic_hashes") == b.get(
                "numerical_diagnostic_hashes"
            )
            drifts.append(metric_drift(a["metrics"], b["metrics"]))
            pairs.append((a["adapter_wall_seconds"], b["adapter_wall_seconds"]))
            vmec_pairs.append(
                (sum(a["vmec_wall_seconds"]), sum(b["vmec_wall_seconds"]))
            )
        maximum = {}
        for drift in drifts:
            for name, value in drift.items():
                if "absolute" not in value:
                    maximum.setdefault(name, {"absolute": None, "relative": None})
                    continue
                old = maximum.setdefault(name, {"absolute": 0, "relative": 0})
                old["absolute"] = max(old["absolute"], value["absolute"])
                old["relative"] = (
                    max(old["relative"], value["relative"])
                    if old["relative"] is not None and value["relative"] is not None
                    else None
                )
        repeated = {
            label: len(identities[label]) == 1
            and len(numerical_identities[label]) == 1
            and audit["repeat_configuration"][label]
            for label in identities
        }
        qualified = len(pairs) >= 5 and exact and all(repeated.values())
        rows.append(
            {
                "seed": seed,
                "track": track,
                "measured_pairs": len(pairs),
                "main_table_eligible": qualified,
                "exact": exact,
                "repeat_deterministic": repeated,
                "warmup_audit": audit,
                "additional_diagnostics_captured": bool(selected)
                and all(complete_diagnostics(r.get("report", {})) for r in selected),
                "actual_indata_exact": decks_exact and bool(pairs),
                "complete_artifacts_captured": captures_complete and bool(pairs),
                "measurement": "derived from executed paired runs; warmups excluded",
                "baseline_seconds": statistics.median(p[0] for p in pairs)
                if pairs
                else None,
                "candidate_seconds": statistics.median(p[1] for p in pairs)
                if pairs
                else None,
                "baseline_vmec_seconds": statistics.median(p[0] for p in vmec_pairs)
                if pairs
                else None,
                "candidate_vmec_seconds": statistics.median(p[1] for p in vmec_pairs)
                if pairs
                else None,
                "speedup": statistics.median(p[0] for p in pairs)
                / statistics.median(p[1] for p in pairs)
                if qualified
                else None,
                "paired_bootstrap_95_interval": interval(pairs) if qualified else None,
                "maximum_drift_per_metric": maximum,
                "raw_adapter_pairs": pairs,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-original-reference", action="store_true")
    args = parser.parse_args()
    rows = summarize_pairs(
        json.loads(args.samples.read_text()),
        args.samples.parent,
        require_original_reference=args.require_original_reference,
    )
    args.out.write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(
        f"Cases: {len(rows)}; qualified: {sum(row['main_table_eligible'] for row in rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
