"""Check an extra public P1 fixture without changing the frozen main suite.

Run baseline first with the original oracle. Run candidates only after the GPU slot is
coordinated. This is extra correctness coverage, not speedup evidence.
"""

from __future__ import annotations

# CLI progress is intentionally emitted.
# ruff: noqa: T201
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from tools.optimization.campaign_gates import clean_environment
from tools.optimization.validate_resident_lifecycle import parse_experiment_overrides
from tools.optimization.validate_resident_solver import (
    compare_input_decks,
    compare_reports,
    compare_traces,
    frozen_boundary_hash,
    read_trace,
    run_process,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(
    python: Path, root: Path, destination: Path, environment: dict, timeout: int
):
    command = [
        str(python),
        str(root / "benchmark-snapshot.py"),
        "--stella",
        json.loads((root / "fixture.json").read_text())["stella"],
        "--seed",
        str(root / "public_p1_seed.json"),
        "--track",
        "P1",
        "--out",
        str(destination),
    ]
    status, elapsed = run_process(
        command, environment, destination.with_suffix(".log"), timeout
    )
    path = destination / "report.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    return status, elapsed, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "candidates"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--stella", type=Path)
    parser.add_argument("--public-seed", type=Path)
    parser.add_argument("--main-manifest", type=Path)
    parser.add_argument("--preload", type=Path)
    parser.add_argument("--reference-extension", type=Path)
    parser.add_argument("--cpu", type=int, default=22)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    root = args.out.resolve()
    os.sched_setaffinity(0, {args.cpu})
    environment = clean_environment(dict(os.environ))
    if args.phase == "baseline":
        if not all((args.stella, args.public_seed, args.main_manifest)):
            parser.error("baseline requires stella, public-seed and main-manifest")
        root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.public_seed, root / "public_p1_seed.json")
        shutil.copy2(
            Path(__file__).with_name("benchmark.py"), root / "benchmark-snapshot.py"
        )
        metadata = {
            "purpose": "extra public P1/P1 correctness coverage; outside main suite",
            "measurement": "diagnostic only; no speedup claim",
            "stella": str(args.stella.resolve()),
            "source": str(args.public_seed.resolve()),
            "source_sha256": digest(args.public_seed),
            "copied_fixture_sha256": digest(root / "public_p1_seed.json"),
            "boundary_sha256": frozen_boundary_hash(root / "public_p1_seed.json"),
            "main_manifest": str(args.main_manifest.resolve()),
            "main_manifest_sha256": digest(args.main_manifest),
            "benchmark_snapshot_sha256": digest(root / "benchmark-snapshot.py"),
        }
        (root / "fixture.json").write_text(json.dumps(metadata, indent=2))
        status, elapsed, report = execute(
            args.python, root, root / "baseline-0", environment, args.timeout
        )
        decks = [
            json.loads(path.read_text())
            for path in sorted((root / "baseline-0").glob("actual_indata_*.json"))
        ]
        resolved_twelve = bool(decks) and all(
            deck.get("mpol") == deck.get("ntor") == 12 for deck in decks
        )
        comparison = compare_reports(report, report)
        summary = {
            "purpose": metadata["purpose"],
            "measurement": metadata["measurement"],
            "baseline_exit_status": status,
            "baseline_process_wall_seconds": elapsed,
            "baseline_complete": comparison,
            "default_preset_resolves_mpol_ntor_12": resolved_twelve,
            "resolved_mode_numbers": [
                {key: deck.get(key) for key in ("mpol", "ntor", "ns_array")}
                for deck in decks
            ],
            "main_manifest_unchanged": digest(args.main_manifest)
            == metadata["main_manifest_sha256"],
            "candidate_samples": [],
            "exact": False,
        }
        (root / "extra-coverage.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        return 0 if status == 0 and comparison["exact"] and resolved_twelve else 1
    if not args.preload or not args.reference_extension:
        parser.error("candidates require preload and reference-extension")
    metadata = json.loads((root / "fixture.json").read_text())
    summary = json.loads((root / "extra-coverage.json").read_text())
    if (
        summary["baseline_exit_status"] != 0
        or not summary["baseline_complete"]["exact"]
    ):
        parser.error("a successful original baseline is required")
    if not summary["default_preset_resolves_mpol_ntor_12"]:
        parser.error("the default preset did not resolve to mpol=ntor=12")
    if digest(root / "public_p1_seed.json") != metadata["source_sha256"]:
        parser.error("the extra public fixture changed")
    overrides = parse_experiment_overrides(args.env)
    environment.update(overrides)
    environment.update(
        LD_PRELOAD=str(args.preload.resolve()),
        VMECPP_FORWARD_REFERENCE_SO=str(args.reference_extension.resolve()),
        VMECPP_GPU_SOLVER="1",
    )
    summary["experimental_environment"] = overrides
    summary["preload_sha256"] = digest(args.preload)
    baseline = json.loads((root / "baseline-0/report.json").read_text())
    first = None
    for repeat in range(2):
        trace = root / f"candidate-{repeat}.trace.jsonl"
        environment["VMECPP_GPU_SOLVER_TRACE"] = str(trace)
        status, elapsed, report = execute(
            args.python, root, root / f"candidate-{repeat}", environment, args.timeout
        )
        comparison = compare_reports(baseline, report)
        decks = compare_input_decks(root / "baseline-0", root / f"candidate-{repeat}")
        first = report if first is None else first
        repeated = compare_reports(first, report)
        events, error = read_trace(trace)
        gpu_updates = sum(bool(event.get("gpu")) for event in events)
        sample = {
            "repeat": repeat,
            "exit_status": status,
            "process_wall_seconds": elapsed,
            "baseline_comparison": comparison,
            "actual_decks_comparison": decks,
            "repeat_comparison": repeated,
            "executed_gpu_updates": gpu_updates,
            "trace_error": error,
        }
        summary["candidate_samples"].append(sample)
        (root / "extra-coverage.json").write_text(json.dumps(summary, indent=2))
        print(repeat, status, comparison["exact"], gpu_updates, flush=True)
        if status != 0 or not all(
            (
                comparison["exact"],
                decks["exact"],
                repeated["exact"],
                gpu_updates > 0,
                error is None,
            )
        ):
            return 1
    summary["repeat_trajectory_comparison"] = compare_traces(
        root / "candidate-0.trace.jsonl", root / "candidate-1.trace.jsonl"
    )
    summary["main_manifest_unchanged"] = (
        digest(Path(metadata["main_manifest"])) == metadata["main_manifest_sha256"]
    )
    summary["exact"] = (
        summary["repeat_trajectory_comparison"]["exact"]
        and summary["main_manifest_unchanged"]
    )
    (root / "extra-coverage.json").write_text(json.dumps(summary, indent=2))
    return 0 if summary["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
