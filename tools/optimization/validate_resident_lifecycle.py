"""Exercise capped-stage GPU lifecycle with deliberately noncomparable decks.

These changed-niter experiments test lifecycle correctness only. They produce no
official metrics and must never enter a fidelity benchmark or speedup table.
"""

from __future__ import annotations

# Optional oracle imports belong only in the isolated worker process.
# ruff: noqa: PLC0415, T201
import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.optimization.validate_resident_solver import (
    compare_traces,
    first_difference,
    read_trace,
    run_process,
)

EXPERIMENT_FLAGS = {
    "VMECPP_GPU_SOLVER_RESIDENT_CONTROL": (0, 1),
    "VMECPP_GPU_SOLVER_GRAPH": (0, 1),
    "VMECPP_GPU_SOLVER_PINNED_SCALARS": (0, 1),
    "VMECPP_GPU_RESIDUAL_ROUNDS": (0, 32),
    "VMECPP_GPU_RESIDUAL_GRAPH": (0, 1),
    "VMECPP_GPU_DIAGNOSTICS_ROUNDS": (0, 32),
    "VMECPP_GPU_DIAGNOSTICS_GRAPH": (0, 1),
}


def parse_experiment_overrides(assignments: list[str]) -> dict[str, str]:
    """Accept only allowlisted numeric GPU options; never accept credentials."""
    overrides = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator or key not in EXPERIMENT_FLAGS:
            message = "--env accepts only allowlisted numeric GPU experiment flags"
            raise ValueError(message)
        if not value.isascii() or not value.isdecimal():
            message = f"Invalid numeric experiment flag: {key}"
            raise ValueError(message)
        minimum, maximum = EXPERIMENT_FLAGS[key]
        number = int(value)
        if not minimum <= number <= maximum:
            message = f"Experiment flag out of range: {key}"
            raise ValueError(message)
        overrides[key] = str(number)
    return overrides


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(deck: Path, out: Path) -> int:
    """Run one standalone capped solve in the explicitly selected oracle Python."""
    import importlib.metadata
    import traceback

    import numpy as np

    import vmecpp

    out.mkdir(parents=True, exist_ok=False)
    report = {
        "configuration": "noncomparable lifecycle-only changed-niter deck",
        "measurement": "executed correctness probe; not speedup evidence",
        "deck_sha256": file_digest(deck),
        "vmecpp_version": importlib.metadata.version("vmecpp"),
        "extension_sha256": {
            str(path): file_digest(path)
            for path in Path(vmecpp.__file__).parent.rglob("_vmecpp*.so")
        },
        "success": False,
    }
    try:
        indata = vmecpp.VmecInput.from_file(deck)
        indata.save(out / "actual_indata_0.json")
        result = vmecpp.run(indata, max_threads=1, verbose=False)
        hashes = {}
        arrays = {}
        for section in ("input", "wout", "jxbout", "mercier", "threed1_volumetrics"):
            fields = {}
            for name, value in getattr(result, section).model_dump().items():
                if isinstance(value, int | float | bool | np.ndarray | np.number):
                    array = np.ascontiguousarray(np.asarray(value))
                    fields[name] = {
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    }
                    arrays[f"{section}.{name}"] = array
            hashes[section] = fields
        np.savez_compressed(out / "numerical.npz", **arrays)
        report["numerical_hashes"] = hashes
        report["success"] = True
    except Exception:
        report["exception"] = traceback.format_exc()
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--preload", type=Path)
    parser.add_argument("--reference-extension", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--env", action="append", default=[], metavar="VMECPP_NAME=VALUE"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.deck, args.out)
    if not all((args.python, args.preload, args.reference_extension)):
        parser.error("--python, --preload and --reference-extension are required")
    try:
        overrides = parse_experiment_overrides(args.env)
    except ValueError as error:
        parser.error(str(error))
    experimental_environment = {
        "VMECPP_GPU_RESIDUAL_ROUNDS": "12",
        "VMECPP_GPU_RESIDUAL_GRAPH": "1",
        **overrides,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {args.cpu})
    original = json.loads(args.deck.read_text())
    summary = {
        "configuration": "noncomparable lifecycle-only changed-niter decks",
        "measurement": "correctness probes only; no speedup evidence",
        "original_deck": str(args.deck.resolve()),
        "original_deck_sha256": file_digest(args.deck),
        "original_deck_fields": original,
        "preload_sha256": file_digest(args.preload),
        "reference_extension_sha256": file_digest(args.reference_extension),
        "experimental_environment": experimental_environment,
        "repeats": 2,
        "samples": [],
        "exact": False,
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("VMECPP_") and key != "LD_PRELOAD"
    }
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LD_PRELOAD=str(args.preload.resolve()),
        VMECPP_FORWARD_REFERENCE_SO=str(args.reference_extension.resolve()),
    )
    environment.update(experimental_environment)
    for niter in (1, 2, 3):
        deck = dict(original)
        deck["niter_array"] = [niter] * len(original["niter_array"])
        deck["return_outputs_even_if_not_converged"] = True
        deck_path = args.out / f"noncomparable-niter-{niter}.json"
        deck_path.write_text(json.dumps(deck, indent=2, allow_nan=False))
        references = {}
        for repeat in range(2):
            for label in ("baseline", "candidate"):
                destination = args.out / f"niter-{niter}-{label}-{repeat}"
                trace = args.out / f"{destination.name}.trace.jsonl"
                environment["VMECPP_GPU_SOLVER_TRACE"] = str(trace.resolve())
                if label == "candidate":
                    environment["VMECPP_GPU_SOLVER"] = "1"
                else:
                    environment.pop("VMECPP_GPU_SOLVER", None)
                command = [
                    str(args.python),
                    "-m",
                    "tools.optimization.validate_resident_lifecycle",
                    "--worker",
                    "--deck",
                    str(deck_path),
                    "--out",
                    str(destination),
                ]
                status, _elapsed = run_process(
                    command, environment, destination.with_suffix(".log"), args.timeout
                )
                report_path = destination / "report.json"
                report = (
                    json.loads(report_path.read_text()) if report_path.exists() else {}
                )
                references[label, repeat] = report
                baseline = references["baseline", 0]
                difference = first_difference(
                    baseline.get("numerical_hashes"), report.get("numerical_hashes")
                )
                expected_bad_jacobian = (
                    niter == 1
                    and "Vmec status code: BAD_JACOBIAN"
                    in baseline.get("exception", "")
                    and report.get("exception") == baseline.get("exception")
                )
                events, trace_error = read_trace(trace)
                gpu_updates = sum(bool(event.get("gpu")) for event in events)
                trace_comparison = compare_traces(
                    args.out / f"niter-{niter}-baseline-0.trace.jsonl", trace
                )
                passed = (
                    (
                        status == 0
                        and report.get("success") is True
                        and bool(report.get("numerical_hashes"))
                        or status == 1
                        and expected_bad_jacobian
                    )
                    and difference is None
                    and trace_error is None
                    and trace_comparison["exact"]
                    and (label != "candidate" or niter == 1 or gpu_updates > 0)
                )
                summary["samples"].append(
                    {
                        "niter_array": deck["niter_array"],
                        "deck_sha256": file_digest(deck_path),
                        "label": label,
                        "repeat": repeat,
                        "exit_status": status,
                        "passed": passed,
                        "expected_bad_jacobian_failure_path": expected_bad_jacobian,
                        "report": str(report_path),
                        "first_numerical_difference": difference,
                        "executed_gpu_updates": gpu_updates,
                        "trace_error": trace_error,
                        "trace_comparison": trace_comparison,
                    }
                )
                (args.out / "lifecycle-validation.json").write_text(
                    json.dumps(summary, indent=2, allow_nan=False)
                )
                print(niter, label, repeat, status, passed, flush=True)
                if not passed:
                    return 1
    summary["exact"] = True
    (args.out / "lifecycle-validation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
