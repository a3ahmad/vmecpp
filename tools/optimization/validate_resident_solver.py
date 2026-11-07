"""Gate resident solver runs against executed baseline output, with zero tolerance.

This is a correctness gate, not a paired speedup campaign. Run smoke first (two fresh
processes), then matrix after smoke succeeds. No solve runs in --check-only. Only
explicitly supplied experimental variables are recorded; credentials are never read or
included in provenance.
"""

from __future__ import annotations

# CLI progress is intentionally emitted.
# ruff: noqa: T201
import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tools.optimization.summarize import metric_drift

IDENTITY_FIELDS = (
    "request_metrics_sha256",
    "metrics_sha256",
    "numerical_wout_hashes",
    "numerical_diagnostic_hashes",
)


def first_difference(baseline, candidate, path=""):
    """Return the first structural/value difference without losing missing keys."""
    if type(baseline) is not type(candidate):
        return {"path": path, "baseline": baseline, "candidate": candidate}
    if isinstance(baseline, dict):
        for key in sorted(baseline.keys() | candidate.keys()):
            child = f"{path}.{key}" if path else key
            if key not in baseline or key not in candidate:
                return {
                    "path": child,
                    "baseline_present": key in baseline,
                    "candidate_present": key in candidate,
                }
            difference = first_difference(baseline[key], candidate[key], child)
            if difference is not None:
                return difference
    elif isinstance(baseline, list):
        for index in range(min(len(baseline), len(candidate))):
            difference = first_difference(
                baseline[index], candidate[index], f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        if len(baseline) != len(candidate):
            return {
                "path": f"{path}.length",
                "baseline": len(baseline),
                "candidate": len(candidate),
            }
    elif (
        isinstance(baseline, float) and baseline.hex() != candidate.hex()
    ) or baseline != candidate:
        return {"path": path, "baseline": baseline, "candidate": candidate}
    return None


def read_trace(path: Path) -> tuple[list, dict | None]:
    """Preserve partial trajectories and identify malformed/unfinished events."""
    if not path.is_file():
        return [], {"missing_trace": str(path)}
    events = []
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return events, {"malformed_trace_line": index, "text": line[:512]}
        if not isinstance(event, dict):
            return events, {"non_object_trace_line": index}
        events.append(event)
    return events, None


def compare_traces(baseline: Path, candidate: Path) -> dict:
    """Locate the first differing scalar/control event in opt-in JSONL traces."""
    traces = []
    undefined_residual_events = []
    for path in (baseline, candidate):
        events, error = read_trace(path)
        if error is not None:
            return {"exact": False, "path": str(path), "error": error}
        previous_ns = None
        residuals_initialized = False
        ignored = []
        for index, event in enumerate(events):
            event.pop("gpu", None)
            if event.get("ns") != previous_ns:
                residuals_initialized = False
                previous_ns = event.get("ns")
            if event.get("bad_jacobian"):
                # The original update exits before assigning force residuals.
                # At a grid's first failed update these scalars are undefined.
                # After one successful update they are valid cached values and
                # must still be compared on every subsequent bad-Jacobian exit.
                if not residuals_initialized and "residuals" in event:
                    event.pop("residuals")
                    ignored.append(index)
            else:
                residuals_initialized = True
        traces.append(events)
        undefined_residual_events.append(ignored)
    difference = first_difference(*traces)
    return {
        "exact": bool(traces[0]) and difference is None,
        "event_counts": [len(events) for events in traces],
        "undefined_initial_bad_jacobian_residual_events": undefined_residual_events,
        "first_difference": difference,
    }


def compare_reports(baseline: dict, candidate: dict) -> dict:
    """Reject incomplete capture, output drift, and unsuccessful solves."""
    missing = [
        f"{label}.{key}"
        for label, report in (("baseline", baseline), ("candidate", candidate))
        for key in IDENTITY_FIELDS
        if not report.get(key)
    ]
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        wout = report.get("numerical_wout_hashes", [])
        diagnostics = report.get("numerical_diagnostic_hashes", [])
        if not all(wout) or len(wout) != len(diagnostics):
            missing.append(f"{label}.complete_numerical_capture")
        for index, diagnostic in enumerate(diagnostics):
            for section in ("jxbout", "mercier", "threed1_volumetrics"):
                if not diagnostic.get(section):
                    missing.append(f"{label}.diagnostics[{index}].{section}")
    differences = {
        key: difference
        for key in IDENTITY_FIELDS
        if (difference := first_difference(baseline.get(key), candidate.get(key)))
        is not None
    }
    statuses = [baseline.get("status"), candidate.get("status")]
    drift = metric_drift(baseline.get("metrics") or {}, candidate.get("metrics") or {})
    return {
        "exact": not missing
        and not differences
        and statuses == ["SUCCESS"] * 2
        and bool(drift)
        and all(row["exact"] for row in drift.values()),
        "missing_required_capture": missing,
        "statuses": statuses,
        "first_difference_per_identity_field": differences,
        "metric_drift": drift,
    }


def read_references(path: Path) -> dict:
    """Load one report or all baseline cases from an executed samples file."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = [{"label": "baseline", "exit_status": 0, "report": data}]
    references = {}
    for sample in data:
        if sample.get("label") != "baseline" or sample.get("exit_status") != 0:
            continue
        report = sample["report"]
        if not compare_reports(report, report)["exact"]:
            message = f"Incomplete or unsuccessful baseline in {path}"
            raise ValueError(message)
        key = (report["seed"], report["track"])
        if key in references and not compare_reports(references[key], report)["exact"]:
            message = f"Non-deterministic baseline case: {key}"
            raise ValueError(message)
        references[key] = report
    if not references:
        message = f"No executed baseline references in {path}"
        raise ValueError(message)
    return references


def frozen_boundary_hash(seed: Path) -> str:
    """Hash the frozen boundary payload, matching canonical zero normalization."""

    def normalize(value):
        if isinstance(value, float) and value == 0.0:
            return 0.0
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    boundary = json.loads(json.loads(seed.read_text())["boundary_json"])
    payload = json.dumps(
        normalize(boundary), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_input_decks(reference: Path, candidate: Path) -> dict:
    """Require unchanged actual solver settings, including every multigrid stage."""
    reference_paths = sorted(reference.glob("actual_indata_*.json"))
    candidate_paths = sorted(candidate.glob("actual_indata_*.json"))
    if not reference_paths or not candidate_paths:
        return {"exact": False, "missing_actual_indata": True}
    baseline = [json.loads(path.read_text()) for path in reference_paths]
    actual = [json.loads(path.read_text()) for path in candidate_paths]
    difference = first_difference(baseline, actual)
    return {"exact": difference is None, "first_difference": difference}


def run_process(command: list[str], environment: dict, log: Path, timeout: int):
    """Bound the whole process group so timeout cannot leave orphaned solvers."""
    started = time.perf_counter()
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            status = "timeout"
    return status, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--stella", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "matrix"), default="smoke")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--preload", type=Path, action="append", default=[])
    parser.add_argument(
        "--env", action="append", default=[], metavar="VMECPP_NAME=VALUE"
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--reference-trace", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two independent processes are required")
    if args.timeout < 1:
        parser.error("timeout must be positive")
    if args.reference_trace and args.mode != "smoke":
        parser.error("--reference-trace is for the single-case smoke gate")
    overrides = {}
    for assignment in args.env:
        key, separator, value = assignment.partition("=")
        if not separator or not key.startswith("VMECPP_"):
            parser.error("--env accepts only explicit VMECPP_ experiment variables")
        overrides[key] = value
    require_gpu = "VMECPP_GPU_SOLVER" in overrides
    tracing = args.trace or bool(args.reference_trace) or require_gpu
    references = read_references(args.reference)
    fixtures = json.loads(args.manifest.read_text())
    cases = [
        (Path(fixture["path"]), track)
        for fixture in fixtures
        for track in ("P1", "P3")
        if args.mode == "matrix"
        or (Path(fixture["path"]).name == "public_p3_seed.json" and track == "P3")
    ]
    if len(cases) != (6 if args.mode == "matrix" else 1):
        parser.error("expected three frozen boundaries, or the public P3 smoke case")
    for seed, track in cases:
        if (seed.name, track) not in references:
            parser.error(f"missing baseline reference for {seed.name} {track}")
        fixture = next(item for item in fixtures if Path(item["path"]) == seed)
        if frozen_boundary_hash(seed) != fixture["boundary_sha256"]:
            parser.error(f"frozen boundary hash changed: {seed.name}")
    if args.check_only:
        print(json.dumps({"check_only": True, "cases": len(cases), "solves": 0}))
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {args.cpu})
    environment = dict(os.environ)
    # Avoid silently inheriting earlier experimental configurations.
    for key in list(environment):
        if key.startswith("VMECPP_") or key == "LD_PRELOAD":
            environment.pop(key)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    environment.update(overrides)
    if args.preload:
        environment["LD_PRELOAD"] = ":".join(str(p.resolve()) for p in args.preload)
    summary = {
        "purpose": "strict correctness gate; not a paired speedup campaign",
        "tolerance": 0,
        "mode": args.mode,
        "diagnostic_tracing": tracing,
        "require_gpu_updates": require_gpu,
        "reference": str(args.reference.resolve()),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "fixture_file_sha256": {
            str(seed): hashlib.sha256(seed.read_bytes()).hexdigest()
            for seed, _ in cases
        },
        "experimental_environment": overrides,
        "preload_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.preload
        },
        "samples": [],
        "exact": False,
    }
    first_reports = {}
    for seed, track in cases:
        for repeat in range(args.repeats):
            destination = args.out / f"{seed.stem}-{track}-{repeat:02d}"
            command = [
                str(args.python),
                "-m",
                "tools.optimization.benchmark",
                "--stella",
                str(args.stella),
                "--seed",
                str(seed),
                "--track",
                track,
                "--out",
                str(destination),
            ]
            trace = args.out / f"{destination.name}.trace.jsonl"
            if tracing:
                environment["VMECPP_GPU_SOLVER_TRACE"] = str(trace.resolve())
            status, elapsed = run_process(
                command, environment, destination.with_suffix(".log"), args.timeout
            )
            report_path = destination / "report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            reference = references[seed.name, track]
            comparison = compare_reports(reference, report)
            reference_directory = (
                args.reference.parent
                if args.reference.name == "report.json"
                else args.reference.parent / "baseline" / f"{seed.stem}-{track}-00"
            )
            input_comparison = compare_input_decks(reference_directory, destination)
            first = first_reports.setdefault((seed.name, track), report)
            repeat_comparison = compare_reports(first, report)
            sample = {
                "measurement": "measured",
                "seed": seed.name,
                "track": track,
                "repeat": repeat,
                "exit_status": status,
                "process_wall_seconds": elapsed,
                "report": str(report_path),
                "baseline_comparison": comparison,
                "input_deck_comparison": input_comparison,
                "repeat_comparison": repeat_comparison,
            }
            trace_comparison = (
                compare_traces(args.reference_trace, trace)
                if args.reference_trace
                else None
            )
            sample["trace_comparison"] = trace_comparison
            events, trace_error = read_trace(trace) if tracing else ([], None)
            gpu_updates = sum(bool(event.get("gpu")) for event in events)
            sample["executed_gpu_updates"] = gpu_updates
            sample["trace_error"] = trace_error
            summary["samples"].append(sample)
            failed = (
                status != 0
                or not comparison["exact"]
                or not input_comparison["exact"]
                or not repeat_comparison["exact"]
                or (trace_comparison is not None and not trace_comparison["exact"])
                or (require_gpu and gpu_updates == 0)
                or (tracing and trace_error is not None)
            )
            (args.out / "validation.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False)
            )
            print(seed.name, track, repeat, status, comparison["exact"], flush=True)
            if failed:
                return 1
    summary["exact"] = True
    (args.out / "validation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
