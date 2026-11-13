"""Read saved research outputs; preserve metric/array drift and failed solves."""

# Optional scientific readers load only for saved CPU comparisons.
# ruff: noqa: PLC0415

import argparse
import csv
import json
import math
from pathlib import Path


def write(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def arrays(reference, actual):
    import numpy as np
    from scipy.io import netcdf_file  # pyright: ignore[reportMissingImports]

    rows = []
    with (
        netcdf_file(reference, "r", mmap=False) as left,
        netcdf_file(actual, "r", mmap=False) as right,
    ):
        for name in sorted(left.variables.keys() | right.variables.keys()):
            if name not in left.variables or name not in right.variables:
                rows.append(
                    {
                        "field": name,
                        "status": "missing",
                        "bits_exact": False,
                        "maxabs": None,
                        "relative_L2": None,
                        "worst_relative_nonzero": None,
                        "zero_reference_count": None,
                        "finite_actual": None,
                    }
                )
                continue
            a, b = (
                np.asarray(left.variables[name].data),
                np.asarray(right.variables[name].data),
            )
            if a.dtype.kind not in "biufc" or b.dtype.kind not in "biufc":
                continue
            row = {
                "field": name,
                "status": "numeric",
                "bits_exact": a.shape == b.shape
                and a.dtype == b.dtype
                and a.tobytes() == b.tobytes(),
                "maxabs": None,
                "relative_L2": None,
                "worst_relative_nonzero": None,
                "zero_reference_count": int(np.count_nonzero(a == 0)),
                "finite_actual": bool(np.isfinite(b).all()),
            }
            if a.shape == b.shape and np.isfinite(a).all() and row["finite_actual"]:
                x, y = a.astype(np.longdouble), b.astype(np.longdouble)
                delta = np.abs(x - y)
                nonzero = x != 0
                scale = (
                    max(
                        float(np.max(np.abs(x), initial=0)),
                        float(np.max(np.abs(y), initial=0)),
                    )
                    or 1.0
                )
                denominator = np.sum((x / scale) ** 2, dtype=np.longdouble)
                row.update(
                    maxabs=float(np.max(delta, initial=0)),
                    relative_L2=float(
                        np.sqrt(
                            np.sum((delta / scale) ** 2, dtype=np.longdouble)
                            / denominator
                        )
                    )
                    if denominator
                    else None,
                    worst_relative_nonzero=float(
                        np.max(delta[nonzero] / np.abs(x[nonzero]), initial=0)
                    ),
                )
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    base = repo / "build/optimization"
    authority = json.loads((base / "gb202-packed-matrix1/matrix1.json").read_text())
    state = json.loads((args.run / "status.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    metrics, physical, calls = [], [], []
    for record in state["rows"]:
        case = authority["cases"][record["case_index"]]
        reference = json.loads(Path(case["original_report"]).read_text())
        path = Path(record["report"])
        actual = json.loads(path.read_text()) if path.exists() else {}
        prefix = {
            "case_index": record["case_index"],
            "boundary": Path(case["seed"]).name,
            "track": case["track"],
            "role": record["role"],
            "call": record["call"],
        }
        values = actual.get("metrics") or {}
        for name in sorted(reference["metrics"].keys() | values.keys()):
            before, after = reference["metrics"].get(name), values.get(name)
            missing = name not in reference["metrics"] or name not in values
            numeric = (
                not missing
                and before is not None
                and after is not None
                and math.isfinite(before)
                and math.isfinite(after)
            )
            absolute = abs(after - before) if numeric else None
            hex_exact = None
            if numeric:
                assert before is not None
                assert after is not None
                hex_exact = float(before).hex() == float(after).hex()
            metrics.append(
                {
                    **prefix,
                    "metric": name,
                    "CPU_original": before,
                    "actual": after,
                    "status": "missing"
                    if missing
                    else ("numeric" if numeric else "null_or_nonfinite"),
                    "absolute": absolute,
                    "relative": absolute / abs(before)
                    if numeric and before
                    else (0 if numeric and after == 0 else None),
                    "hex_exact": hex_exact,
                    "unchanged_null": not missing and before is None and after is None,
                }
            )
        directory = path.parent.parent
        originals = sorted(Path(case["original_directory"]).glob("raw_wout_*.nc"))
        outputs = sorted((directory / "worker").glob("saved_wout_*.nc"))
        for index, (left, right) in enumerate(zip(originals, outputs, strict=False)):
            physical.extend(
                {**prefix, "output_index": index, **row} for row in arrays(left, right)
            )
        calls.append(
            {
                **prefix,
                "status": actual.get("status", "REPORT_MISSING"),
                "direct_exit_status": record["process"].get("direct_exit_status"),
                "expected_vmec_outputs": len(originals),
                "saved_vmec_outputs": len(outputs),
                "FP64_terminal_certificate_checked": record.get(
                    "FP64_terminal_certificate_checked"
                ),
                "FP64_terminal_certificate_passed": record.get(
                    "FP64_terminal_certificate_passed"
                ),
                "trace": record.get("precision_trace"),
                "report": str(path),
            }
        )
    write(args.out / "metric-drift.csv", metrics)
    write(args.out / "wout-array-errors.csv", physical)
    write(args.out / "calls.csv", calls)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "scope": "ALTERNATIVE_RESEARCH_NOT_QUALIFIED",
                "execution_complete": state["complete"],
                "calls": len(calls),
                "metric_observations": len(metrics),
                "numeric_wout_fields_compared": len(physical),
                "originals_unmodified": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
