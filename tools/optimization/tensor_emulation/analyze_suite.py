"""Analyze saved GEMM outputs with independent exact-rational selected dots.

No solver, GPU library, profiler, or provider is loaded by this CPU-only reader.
"""

# Plotting dependencies are optional until --plots is requested.
# ruff: noqa: PLC0415

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np

MODES = (
    "native",
    "dynamic-eager",
    "dynamic-performant",
    "reduced-4",
    "reduced-8",
    "reduced-12",
)


def mode_order(available):
    if set(available) == set(MODES):
        return MODES
    assert "native" in available
    fixed = set(available) - {"native"}
    assert fixed
    assert all(name.startswith("fixed-") for name in fixed)
    return ("native", *sorted(fixed, key=lambda name: -int(name.split("-", 1)[1])))


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def same_bytes(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while chunk := a.read(1024 * 1024):
            if chunk != b.read(len(chunk)):
                return False
    return True


def load_input(path):
    with path.open("rb") as stream:
        assert stream.read(8) == b"TEGEMM1\0"
        m, n, k = struct.unpack("<QQQ", stream.read(24))
    assert m > 0
    assert n > 0
    assert k > 0
    assert path.stat().st_size == 32 + 8 * (m * k + k * n + m * n)
    values = np.memmap(path, mode="r", dtype="<f8", offset=32)
    a = values[: m * k].reshape((m, k), order="F")
    b = values[m * k : m * k + k * n].reshape((k, n), order="F")
    reference = values[m * k + k * n :]
    assert np.isfinite(values).all()
    return cast(np.ndarray, a), cast(np.ndarray, b), cast(np.ndarray, reference)


def ordered_bits(values):
    bits = values.view(np.uint64)
    return np.where(bits >> np.uint64(63), ~bits, bits | np.uint64(1 << 63))


def errors(actual, reference):
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all()
    delta = np.abs(actual - reference)
    left, right = ordered_bits(actual), ordered_bits(reference)
    ulps = np.where(left >= right, left - right, right - left)
    ulps[(actual == 0) & (reference == 0)] = 0
    mismatch = actual.view(np.uint64) != reference.view(np.uint64)
    nonzero = reference != 0
    scale = (
        max(
            float(np.max(np.abs(reference), initial=0)),
            float(np.max(np.abs(actual), initial=0)),
        )
        or 1.0
    )
    scaled_reference = np.asarray(reference, dtype=np.longdouble) / scale
    scaled_actual = np.asarray(actual, dtype=np.longdouble) / scale
    numerator = np.sum((scaled_actual - scaled_reference) ** 2, dtype=np.longdouble)
    denominator = np.sum(scaled_reference**2, dtype=np.longdouble)
    relative_l2 = float(np.sqrt(numerator / denominator)) if denominator else None
    result = {
        "different_ordered_bits": int(np.count_nonzero(mismatch)),
        "signed_zero_differences": int(
            np.count_nonzero(mismatch & (actual == 0) & (reference == 0))
        ),
        "max_absolute_error": float(np.max(delta, initial=0)),
        "max_relative_error_nonzero_reference": float(
            np.max(delta[nonzero] / np.abs(reference[nonzero]), initial=0)
        ),
        "zero_reference_count": int(np.count_nonzero(~nonzero)),
        "max_absolute_error_zero_reference": float(np.max(delta[~nonzero], initial=0)),
        "max_ULP_zero_sign_separate": int(np.max(ulps, initial=0)),
        "relative_L2": relative_l2,
    }
    return result, delta, ulps


def select_entries(a, b, reference, records, count):
    worst_abs = np.zeros(reference.size)
    worst_ulp = np.zeros(reference.size, dtype=np.uint64)
    for record in records:
        _, absolute, ulps = errors(record["array"], reference)
        np.maximum(worst_abs, absolute, out=worst_abs)
        np.maximum(worst_ulp, ulps, out=worst_ulp)
    # This FP64 scale is a SELECTION HEURISTIC, not the accuracy authority.
    absolute_terms = np.zeros((a.shape[0], b.shape[1]), order="F")
    for inner in range(a.shape[1]):
        absolute_terms += np.abs(a[:, inner, None] * b[inner, None, :])
    absolute_terms = absolute_terms.ravel(order="F")
    cancellation = np.zeros(reference.size)
    np.divide(absolute_terms, np.abs(reference), out=cancellation, where=reference != 0)
    cancellation[(reference == 0) & (absolute_terms > 0)] = np.inf
    buckets = (
        ("worst_absolute", np.argsort(-worst_abs, kind="stable")),
        ("worst_ULP", np.argsort(-worst_ulp.astype(np.longdouble), kind="stable")),
        ("near_zero_reference", np.argsort(np.abs(reference), kind="stable")),
        ("cancellation", np.argsort(-cancellation, kind="stable")),
    )
    target = min(count, reference.size)
    quota = math.ceil(target / len(buckets))
    selected = {}
    for reason, order in buckets:
        added = 0
        for raw_index in order:
            index = int(raw_index)
            if index not in selected:
                selected[index] = reason
                added += 1
            if added >= quota or len(selected) >= target:
                break
        if len(selected) >= target:
            break
    assert len(selected) == target
    return selected


def rational_checks(case, a, b, reference, records, count):
    dots, comparisons = [], []
    for index, reason in select_entries(a, b, reference, records, count).items():
        row, column = index % a.shape[0], index // a.shape[0]
        products = [
            Fraction(float(a[row, inner])) * Fraction(float(b[inner, column]))
            for inner in range(a.shape[1])
        ]
        exact = sum(products, Fraction(0))
        absolute_sum = sum(map(abs, products), Fraction(0))
        rounded = float(exact)
        dots.append(
            {
                "case": case,
                "flat_column_major_index": index,
                "row": row,
                "column": column,
                "selected_for": reason,
                "terms": a.shape[1],
                "exact_sum_numerator": str(exact.numerator),
                "exact_sum_denominator": str(exact.denominator),
                "rounded_exact_GEMM_hex": rounded.hex(),
                "original_ordered_hex": float(reference[index]).hex(),
                "exact_sum_abs_products_numerator": str(absolute_sum.numerator),
                "exact_sum_abs_products_denominator": str(absolute_sum.denominator),
                "reference_scope": "exact mathematical A*B of stored binary64 operands; not non-GEMM physics epilogues",
            }
        )
        rounded_bits = struct.pack("<d", rounded)
        for record in records:
            value = float(record["array"][index])
            difference = abs(Fraction(value) - exact)
            comparisons.append(
                {
                    "case": case,
                    "version": record["version"],
                    "repetition": record["repetition"],
                    "requested_mode": record["mode"],
                    "row": row,
                    "column": column,
                    "selected_for": reason,
                    "actual_hex": value.hex(),
                    "correctly_rounded_GEMM_bits": struct.pack("<d", value)
                    == rounded_bits,
                    "absolute_error_vs_exact_GEMM": float(difference),
                    "relative_error_vs_exact_GEMM": float(difference / abs(exact))
                    if exact
                    else None,
                    "relative_status": "defined"
                    if exact
                    else "zero_exact_reference_NA",
                    "cancellation_aware_error": float(difference / absolute_sum)
                    if absolute_sum
                    else None,
                    "error_exact_numerator": str(difference.numerator),
                    "error_exact_denominator": str(difference.denominator),
                }
            )
    return dots, comparisons


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out, rows):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    modes = mode_order({r["requested_mode"] for r in rows})
    for case in sorted({r["case"] for r in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for version in sorted({r["version"] for r in rows if r["case"] == case}):
            color = {"13.1": "#2166ac", "13.5": "#b35806"}[version]
            selected = [
                r for r in rows if r["case"] == case and r["version"] == version
            ]
            for ax, field in zip(
                axes, ("steady_event_ms", "max_absolute_error"), strict=True
            ):
                medians = []
                for index, mode in enumerate(modes):
                    values = [r[field] for r in selected if r["requested_mode"] == mode]
                    medians.append(float(np.median(values)))
                    ax.scatter(
                        [index] * len(values), values, color=color, alpha=0.45, s=18
                    )
                ax.plot(
                    range(len(modes)),
                    medians,
                    "o-",
                    color=color,
                    label="CUDA " + version,
                )
                ax.set_xticks(
                    range(len(modes)), modes, rotation=30, ha="right", fontsize=8
                )
                ax.grid(alpha=0.2)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Full library GEMM call event time (ms)")
        positive = [
            r["max_absolute_error"]
            for r in rows
            if r["case"] == case and r["max_absolute_error"] > 0
        ]
        axes[1].set_yscale(
            "symlog", linthresh=min(positive) / 10 if positive else 1e-18
        )
        axes[1].set_ylabel("Maximum absolute error vs original ordered output")
        axes[0].legend()
        fig.suptitle(
            case + " - requested modes, not proven dispatch labels", fontsize=12
        )
        fig.text(
            0.5,
            0.02,
            "RTX5090 / SM120. Dots: two repeats; lines: medians. External packing excluded. Not solver metrics or throughput.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(out / (case + "-time-error.png"), dpi=160)
        fig.savefig(out / (case + "-time-error.pdf"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--entries", type=int, default=32)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    assert (
        args.entries == 32
    ), "This bounded screen uses32 independent exact dots per case"
    status_path = args.suite / "status.json"
    status = json.loads(status_path.read_text())
    assert status["complete"]
    assert all(row["exit_code"] == 0 for row in status["rows"])
    grouped = defaultdict(list)
    sources = {str(status_path): digest(status_path)}
    for row in status["rows"]:
        grouped[row["case"]].append(row)
    rows, repeat_checks, cross_version, all_dots, all_exact_errors = [], [], [], [], []
    modes = None
    for case, runs in grouped.items():
        versions = sorted({r["version"] for r in runs})
        assert set(versions) <= {"13.1", "13.5"}
        assert {(r["version"], r["repetition"]) for r in runs} == {
            (v, r) for v in versions for r in (0, 1)
        }
        assert len(runs) == len(versions) * 2
        source = Path(runs[0]["command"][runs[0]["command"].index("--input") + 1])
        manifest_path = source.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        job = next(job for job in manifest["jobs"] if job["name"] == case)
        assert digest(source) == job["sha256"]
        sources[str(source)] = job["sha256"]
        sources[str(manifest_path)] = digest(manifest_path)
        a, b, reference = load_input(source)
        records = []
        for run in runs:
            destination = Path(run["output"])
            available = {
                p.name.removeprefix("report.").removesuffix(".json")
                for p in destination.glob("report.*.json")
            }
            current_modes = mode_order(available)
            if modes is None:
                modes = current_modes
            assert current_modes == modes
            native_ms = json.loads((destination / "report.native.json").read_text())[
                "steady_event_median_ms"
            ]
            for mode in modes:
                report_path = destination / f"report.{mode}.json"
                output_path = destination / f"output.{mode}.bin"
                report = json.loads(report_path.read_text())
                assert report["complete"]
                assert not report["profile_one"]
                assert report["mode"] == mode
                assert (report["m"], report["n"], report["k"]) == (
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                )
                assert Path(report["input"]).resolve() == source.resolve()
                assert output_path.stat().st_size == reference.size * 8
                actual = np.memmap(output_path, mode="r", dtype="<f8")
                error, _, _ = errors(actual, reference)
                row = {
                    "case": case,
                    "family": job["family"],
                    "version": run["version"],
                    "repetition": run["repetition"],
                    "requested_mode": mode,
                    "accuracy_label": report["accuracy_label"],
                    "device": report["device"],
                    "SMs": report["SMs"],
                    "CC_major": report["CC_major"],
                    "CC_minor": report["CC_minor"],
                    "runtime_version": report["runtime_version"],
                    "driver_version": report["driver_version"],
                    "M": a.shape[0],
                    "N": b.shape[1],
                    "K": a.shape[1],
                    "graph_captured": report.get("graph_captured", False),
                    "capture_RAW_ms_diagnostic": report.get("capture_RAW_ms"),
                    "instantiate_RAW_ms_diagnostic": report.get("instantiate_RAW_ms"),
                    "first_graph_event_ms": report.get("first_graph_event_ms"),
                    "steady_event_ms": report["steady_event_median_ms"],
                    "native_over_requested_event_ratio": native_ms
                    / report["steady_event_median_ms"],
                    "first_event_ms": report["first_GEMM_event_ms"],
                    "RAW_diagnostic_ms": report["steady_RAW_diagnostic_median_ms"],
                    "mantissa_offset": report["mantissa_offset"],
                    "mantissa_control": report["mantissa_control"],
                    "max_mantissa_bits_readback": report["max_mantissa_bits_readback"],
                    "compute_type": report["compute_type"],
                    "cublas_version": report["cublas_version"],
                    "cublas_owner": report["cublas_owner"],
                    "actual_tensor_path_proven_by_this_report": report[
                        "actual_tensor_path_proven"
                    ],
                    "output_sha256": digest(output_path),
                    **error,
                }
                rows.append(row)
                records.append(
                    {
                        "version": run["version"],
                        "repetition": run["repetition"],
                        "mode": mode,
                        "path": output_path,
                        "array": actual,
                        "row": row,
                    }
                )
                sources[str(report_path)] = digest(report_path)
        assert modes is not None
        for version in versions:
            for mode in modes:
                left, right = sorted(
                    (
                        r
                        for r in records
                        if r["version"] == version and r["mode"] == mode
                    ),
                    key=lambda r: r["repetition"],
                )
                repeat_checks.append(
                    {
                        "case": case,
                        "version": version,
                        "requested_mode": mode,
                        "full_output_bytes_equal": same_bytes(
                            left["path"], right["path"]
                        ),
                        "repeat0_sha256": left["row"]["output_sha256"],
                        "repeat1_sha256": right["row"]["output_sha256"],
                    }
                )
        for repetition in (0, 1) if len(versions) == 2 else ():
            for mode in modes:
                left, right = sorted(
                    (
                        r
                        for r in records
                        if r["repetition"] == repetition and r["mode"] == mode
                    ),
                    key=lambda r: r["version"],
                )
                cross_version.append(
                    {
                        "case": case,
                        "repetition": repetition,
                        "requested_mode": mode,
                        "CUDA13_1_vs13_5_bytes_equal": same_bytes(
                            left["path"], right["path"]
                        ),
                    }
                )
        dots, comparisons = rational_checks(
            case, a, b, reference, records, args.entries
        )
        all_dots.extend(dots)
        all_exact_errors.extend(comparisons)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "time-error.csv", rows)
    write_csv(args.out / "repeat-byte-identity.csv", repeat_checks)
    write_csv(args.out / "cross-version-byte-identity.csv", cross_version)
    write_csv(args.out / "selected-exact-errors.csv", all_exact_errors)
    (args.out / "exact-rational-dots.json").write_text(
        json.dumps(all_dots, indent=2) + "\n"
    )
    summary = {
        "scope": "GEMM-only alternatives on stored operands; no solver metrics or qualification",
        "suite_complete": True,
        "analyzer_sha256": digest(Path(__file__)),
        "cases": len(grouped),
        "mode_reports": len(rows),
        "graph_captured_reports": sum(r["graph_captured"] for r in rows),
        "repeat_comparisons": len(repeat_checks),
        "repeat_byte_exact": sum(r["full_output_bytes_equal"] for r in repeat_checks),
        "independent_exact_dots": len(all_dots),
        "selected_arm_comparisons": len(all_exact_errors),
        "exact_reference": "Python Fraction exact binary64 products and sum, alpha1 beta0; one final binary64 rounding",
        "selection": "32 unique per case balanced worst absolute/ULP, nearzero, cancellation; cancellation scale is only a selection heuristic",
        "dispatch_labels": "requested modes only; separate target-specific profile proof required, mixed fallback permitted",
        "timing_scope": "event time includes internal library work, excludes external operand packing; RAW is WSL diagnostic, not wall-speed authority",
        "norm_arithmetic_bits": int(np.finfo(np.longdouble).nmant + 1),
        "sources": sources,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plots:
        make_plots(args.out, rows)
    sys.stdout.write(
        json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
