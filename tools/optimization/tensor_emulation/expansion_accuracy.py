"""CPU-only expansion diagnostics and public synthetic TEGEMM stress inputs.

No GPU library is imported. Supplied C, exact mathematical A*B, native GEMM, and the
full original transform are distinct reference scopes.
"""

import argparse
import importlib.util
import json
import math
import struct
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np


def helpers():
    path = Path(__file__).with_name("analyze_suite.py")
    spec = importlib.util.spec_from_file_location("saved_gemm_analysis", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ordered_dot(left, right):
    """Independent separate RN multiply/add, including IEEE zero signs."""
    value = 0.0
    for raw_a, raw_b in zip(left, right, strict=True):
        a, b = float(raw_a), float(raw_b)
        product = float(Fraction(a) * Fraction(b))
        if product == 0:
            product = math.copysign(0.0, math.copysign(1.0, a) * b)
        result = float(Fraction(value) + Fraction(product))
        if result == 0 and value == 0 and product == 0:
            result = (
                -0.0
                if math.copysign(1.0, value) < 0 and math.copysign(1.0, product) < 0
                else 0.0
            )
        value = result
    return value


def stress_inputs():
    """Small finite operands; range cases intentionally challenge digit scaling."""
    patterns = {
        "dyadic": [i / 16 for i in range(-8, 8)],
        "cancellation": [2.0**53, 1.0, -(2.0**53), 0.0] * 4,
        "nearzero": [1.0, 2.0**-100, -1.0, 0.0] * 4,
        "signedzero": [0.0, -0.0, 0.0, -0.0] * 4,
        "subnormal": [2.0**-1022, 2.0**-1050, 2.0**-1074, -(2.0**-1074)] * 4,
        "lowtail": [1.0 + 2.0**-52, -1.0, 2.0**-104, 0.0] * 4,
        "ties": [1.0, 2.0**-53, 2.0**-52, -1.0] * 4,
        "balanced_range": [
            math.ldexp(1.0, e) for e in (-500, -150, -24, 0, 24, 150, 500, 0)
        ]
        * 2,
    }
    for name, pattern in patterns.items():
        a = np.array([pattern] * 8, dtype="<f8")
        b = np.ones((16, 4), dtype="<f8")
        b[:, 1] = -1.0
        b[:, 2] = 0.5
        b[:, 3] = np.tile([1.0, -1.0], 8)
        if name == "balanced_range":
            b *= np.array([1.0 / x for x in pattern])[:, None]
        for row in range(8):
            a[row] = np.roll(a[row], 4 * (row % 4))
        if name == "balanced_range":
            # Keep products finite even when row rotations change pairing.
            a[:] = pattern
        c = np.array(
            [[ordered_dot(a[i], b[:, j]) for j in range(4)] for i in range(8)],
            dtype="<f8",
        )
        yield name, a, b, c


def generate(out):
    out.mkdir(parents=True, exist_ok=False)
    h = helpers()
    jobs = []
    for name, a, b, c in stress_inputs():
        path = out / (name + ".bin")
        with path.open("wb") as stream:
            stream.write(b"TEGEMM1\0" + struct.pack("<QQQ", 8, 4, 16))
            for values in (a, b, c):
                stream.write(values.tobytes(order="F"))
        jobs.append(
            {
                "name": name,
                "file": path.name,
                "sha256": h.digest(path),
                "M": 8,
                "N": 4,
                "K": 16,
            }
        )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "jobs": jobs,
                "reference_scope": "synthetic separate RN FP64 multiply/add in flat K order; not original VMEC Fourier",
                "range_cases_may_require_fallback": True,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


def cancellation_error(a, b, actual, reference):
    # longdouble is a diagnostic denominator; selected Fraction dots are exact.
    denominator = (
        np.abs(a.astype(np.longdouble)) @ np.abs(b.astype(np.longdouble))
    ).ravel(order="F")
    error = np.abs(actual.astype(np.longdouble) - reference.astype(np.longdouble))
    valid = denominator != 0
    return {
        "max_cancellation_scaled_error_vs_C": float(
            np.max(error[valid] / denominator[valid], initial=0)
        ),
        "zero_absolute_product_sum_count": int(np.count_nonzero(~valid)),
        "max_absolute_error_zero_product_sum": float(np.max(error[~valid], initial=0)),
        "cancellation_denominator_scope": "longdouble sum(abs(A*B)); selected Fraction dots exact",
    }


def analyze(args):
    h = helpers()
    manifest = json.loads(args.manifest.read_text())
    jobs = {row["name"]: row for row in manifest["jobs"]}
    status = json.loads((args.suite / "status.json").read_text())
    assert status["complete"]
    args.out.mkdir(parents=True, exist_ok=False)
    rows, repeats, dots, comparisons = [], [], [], []
    groups = defaultdict(list)
    for run in status["rows"]:
        assert run["exit_code"] == 0
        case = run["case"]
        source = args.manifest.parent / jobs[case]["file"]
        assert h.digest(source) == run["input_sha256"] == jobs[case]["sha256"]
        a, b, reference = h.load_input(source)
        directory = args.suite / run["label"]
        reports = sorted(directory.glob("report.*.json"))
        assert reports
        native_path = directory / "output.native.bin"
        native = np.fromfile(native_path, dtype="<f8")
        assert native.size == reference.size
        native_report = json.loads((directory / "report.native.json").read_text())
        ordered_report_path = directory / "report.ordered.json"
        ordered_report = (
            json.loads(ordered_report_path.read_text())
            if ordered_report_path.exists()
            else {}
        )
        ordered_path = directory / "output.ordered.bin"
        ordered = (
            np.fromfile(ordered_path, dtype="<f8") if ordered_path.exists() else None
        )
        for report_path in reports:
            report = json.loads(report_path.read_text())
            mode = report_path.name[len("report.") : -len(".json")]
            assert report["mode"] == mode
            assert report["complete"]
            output = directory / ("output." + mode + ".bin")
            values = np.fromfile(output, dtype="<f8")
            assert values.size == reference.size
            record = {
                "case": case,
                "repetition": run["repetition"],
                "mode": mode,
                "version": "13.5",
                "array": values,
                "path": output,
            }
            groups[case].append(record)
            finite = bool(np.isfinite(values).all())
            row = {
                "case": case,
                "repetition": run["repetition"],
                "mode": mode,
                "finite_output": finite,
                "output_sha256": h.digest(output),
                "report_sha256": h.digest(report_path),
                "original_GPU_comparator_qualified": False,
                "reference_scope": "supplied C, not proven actual original GPU algorithm",
            }
            for key in (
                "type",
                "components",
                "digit_bits",
                "Kblock",
                "exact_FP32_subdot_bound",
                "graph_captured",
                "graph_batch",
                "split",
                "reuse_B",
                "terms",
                "range_failures",
                "workspace_bytes",
                "steady_event_median_ms",
                "steady_RAW_diagnostic_median_ms",
                "first_direct_event_ms",
                "first_direct_RAW_ms",
                "mode_allocation_RAW_ms",
                "B_presplit_event_ms",
                "capture_and_first_graph_RAW_ms",
                "backend",
                "cached_operands_active",
                "fused_A_split_active",
                "quantized_split_A_active",
                "operations_per_sample",
                "quantized_split_active",
                "quantized_split_recurrence_fallback",
                "fallback_count",
                "pair_count",
                "group_diagonals",
                "parallel_group_count",
                "group_pair_capacity",
                "digit_integer_abs_bound",
                "omitted_cross_term_abs_bound",
                "bound_scope",
                "actual_tensor_path_proven",
                "max_split_residual_A",
                "max_split_residual_B",
            ):
                row[key] = report.get(key)
            if finite:
                row.update(h.errors(values, reference)[0])
                row.update(cancellation_error(a, b, values, reference))
                row.update(
                    {
                        "vs_native_" + key: value
                        for key, value in h.errors(values, native)[0].items()
                    }
                )
            if finite and ordered is not None:
                row.update(
                    {
                        "vs_flat_ordered_GPU_" + key: value
                        for key, value in h.errors(values, ordered)[0].items()
                    }
                )
            event = report.get("steady_event_median_ms")
            baseline = native_report.get("steady_event_median_ms")
            row["same_run_native_event_ratio"] = (
                baseline / event if event and baseline else None
            )
            ordered_event = ordered_report.get("steady_event_median_ms")
            row["same_run_flat_ordered_GPU_event_ratio"] = (
                ordered_event / event if event and ordered_event else None
            )
            rows.append(row)
    for case, records in groups.items():
        source = args.manifest.parent / jobs[case]["file"]
        a, b, reference = h.load_input(source)
        finite_records = [r for r in records if np.isfinite(r["array"]).all()]
        if finite_records:
            selected, checked = h.rational_checks(
                case, a, b, reference, finite_records, 32
            )
            dots.extend(selected)
            comparisons.extend(checked)
        for mode in sorted({r["mode"] for r in records}):
            runs = sorted(
                [r for r in records if r["mode"] == mode], key=lambda r: r["repetition"]
            )
            repeats.append(
                {
                    "case": case,
                    "mode": mode,
                    "repeats": len(runs),
                    "repeat_check_available": len(runs) >= 2,
                    "all_repeat_bytes_exact": all(
                        h.same_bytes(runs[0]["path"], r["path"]) for r in runs[1:]
                    )
                    if len(runs) >= 2
                    else None,
                }
            )
    h.write_csv(args.out / "accuracy.csv", rows)
    h.write_csv(args.out / "repeat-identity.csv", repeats)
    h.write_csv(args.out / "exact-comparisons.csv", comparisons)
    (args.out / "exact-dots.json").write_text(
        json.dumps(dots, indent=2) + "\n", encoding="ascii"
    )
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "reports": len(rows),
                "repeat_checks": repeats,
                "exact_selected_dots": len(dots),
                "original_GPU_comparator_qualified": False,
                "solver_qualified": False,
                "scope": "component only; native DGEMM and full18-output inverse control are different algorithms/scopes; no full transform speed ratio",
                "error_bound_scope": "exact FP32 subdots do not bound finite-q split residual, triangular omitted products, scaling or reconstruction",
                "RAW_scope": "diagnostic only; no WSL host-wall authority",
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


def checks():
    assert ordered_dot([2.0**53, 1.0, -(2.0**53)], [1.0] * 3) == 0
    assert sum((Fraction(x) for x in [2.0**53, 1.0, -(2.0**53)]), Fraction()) == 1
    assert ordered_dot([1.0, 2.0**-53], [1.0, 1.0]) == 1.0
    assert ordered_dot([2.0**-1074], [0.5]) == 0
    cases = list(stress_inputs())
    assert len(cases) == 8
    for _, a, b, c in cases:
        assert a.shape == (8, 16)
        assert b.shape == (16, 4)
        assert c.shape == (8, 4)
        assert np.isfinite(a).all()
        assert np.isfinite(b).all()
        assert np.isfinite(c).all()
    h = helpers()
    result = h.errors(np.array([1.0, 1.0]), np.array([0.0, 1.0]))[0]
    assert result["max_absolute_error"] == 1.0
    assert (
        cancellation_error(
            np.ones((1, 2)), np.array([[1.0], [-1.0]]), np.array([1.0]), np.array([0.0])
        )["max_cancellation_scaled_error_vs_C"]
        == 0.5
    )
    return {"complete": True, "synthetic_cases": 8, "GPU_calls": 0}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--generate", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(checks()))  # noqa: T201
    elif args.generate:
        generate(args.generate)
    else:
        assert args.manifest
        assert args.suite
        assert args.out
        analyze(args)


if __name__ == "__main__":
    main()
