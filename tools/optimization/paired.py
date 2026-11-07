"""Run serial, alternating baseline/candidate measurements on frozen fixtures."""

from __future__ import annotations

# CLI progress is intentionally emitted.
# ruff: noqa: T201
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

if __package__:
    from .campaign_gates import (
        clean_environment,
        complete_capture,
        pair_gate,
        read_capture,
    )
else:
    from campaign_gates import (
        clean_environment,
        complete_capture,
        pair_gate,
        read_capture,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--stella", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument(
        "--tracks", nargs="+", choices=("P1", "P3"), default=("P1", "P3")
    )
    parser.add_argument(
        "--candidate-cpus",
        help="Optional comma-separated candidate CPU affinity; baseline stays on --cpu",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-directory", type=Path)
    args = parser.parse_args()
    candidate_cpus = (
        {int(value) for value in args.candidate_cpus.split(",")}
        if args.candidate_cpus
        else {args.cpu}
    )
    os.sched_setaffinity(0, {args.cpu})
    args.out.mkdir(parents=True, exist_ok=args.resume)
    fixtures = json.loads(args.manifest.read_text())
    environment = clean_environment(dict(os.environ))
    samples = json.loads((args.out / "samples.json").read_text()) if args.resume else []
    start_pair = max((row["pair"] for row in samples), default=-1) + 1
    if samples:
        previous = [row for row in samples if row["pair"] == start_pair - 1]
        if len(previous) != len(fixtures) * len(args.tracks) * 2 or any(
            row["exit_status"] != 0 for row in previous
        ):
            parser.error(
                "resume requires a fully completed successful pair across all fixtures"
            )
        actual_cases = {(row["seed"], row["track"]) for row in previous}
        expected_cases = {
            (Path(f["path"]).name, track) for f in fixtures for track in args.tracks
        }
        if actual_cases != expected_cases:
            parser.error("resume fixture matrix differs")
        for seed, track in expected_cases:
            records = [
                row for row in previous if row["seed"] == seed and row["track"] == track
            ]
            labels = {row["label"]: row for row in records}
            if len(records) != 2 or labels.keys() != {"baseline", "candidate"}:
                parser.error("resume requires unique baseline/candidate records")
            captures = {
                label: record.get("captures")
                or read_capture(
                    args.out / label / f"{Path(seed).stem}-{track}-{start_pair - 1:02d}"
                )
                for label, record in labels.items()
            }
            if not pair_gate(
                labels["baseline"]["report"],
                labels["candidate"]["report"],
                captures["baseline"],
                captures["candidate"],
            )["exact"]:
                parser.error("resume requires exact complete prior pair artifacts")
            if args.reference_directory and any(
                not (row.get("original_reference") or {}).get("gate", {}).get("exact")
                for row in records
            ):
                parser.error("resume requires original-reference parity")
    for pair in range(start_pair, args.pairs):
        for fixture in fixtures:
            seed = Path(fixture["path"])
            for track in args.tracks:
                reports = {}
                captures = {}
                labels = (
                    ("baseline", "candidate")
                    if pair % 2 == 0
                    else ("candidate", "baseline")
                )
                for label in labels:
                    interpreter = getattr(args, f"{label}_python")
                    destination = args.out / label / f"{seed.stem}-{track}-{pair:02d}"
                    destination.parent.mkdir(exist_ok=True)
                    command = [
                        str(interpreter),
                        str(Path(__file__).with_name("benchmark.py")),
                        "--stella",
                        str(args.stella),
                        "--seed",
                        str(seed),
                        "--track",
                        track,
                        "--out",
                        str(destination),
                    ]
                    started = time.perf_counter()
                    with (destination.parent / f"{destination.name}.log").open(
                        "w"
                    ) as log:
                        try:
                            completed = subprocess.run(
                                command,
                                env=environment,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=1800,
                                check=False,
                                preexec_fn=(
                                    lambda: os.sched_setaffinity(0, candidate_cpus)
                                )
                                if label == "candidate"
                                else None,
                            )
                            status = completed.returncode
                        except subprocess.TimeoutExpired:
                            status = "timeout"
                    path = destination / "report.json"
                    report = json.loads(path.read_text()) if path.exists() else {}
                    reports[label] = report
                    capture = read_capture(destination)
                    captures[label] = capture
                    reference_gate = None
                    if args.reference_directory:
                        reference_directory = (
                            args.reference_directory / f"{seed.stem}-{track}"
                        )
                        reference_path = reference_directory / "report.json"
                        reference_gate = {
                            "report_sha256": hashlib.sha256(
                                reference_path.read_bytes()
                            ).hexdigest(),
                            "gate": pair_gate(
                                json.loads(reference_path.read_text()),
                                report,
                                read_capture(reference_directory),
                                capture,
                                require_harness=False,
                            ),
                        }
                    samples.append(
                        {
                            "label": label,
                            "pair": pair,
                            "warmup": pair == 0,
                            "seed": seed.name,
                            "track": track,
                            "exit_status": status,
                            "process_wall_seconds": time.perf_counter() - started,
                            "report": report,
                            "captures": capture,
                            "captures_complete": complete_capture(report, capture),
                            "original_reference": reference_gate,
                        }
                    )
                    temporary = args.out / "samples.json.tmp"
                    temporary.write_text(json.dumps(samples, indent=2, allow_nan=False))
                    temporary.replace(args.out / "samples.json")
                    print(pair, seed.name, track, label, status, flush=True)
                    if status != 0:
                        return 1
                    if not complete_capture(report, capture):
                        return 5
                    if (
                        reference_gate is not None
                        and not reference_gate["gate"]["exact"]
                    ):
                        return 6
                baseline, candidate = reports["baseline"], reports["candidate"]
                gate = pair_gate(
                    baseline, candidate, captures["baseline"], captures["candidate"]
                )
                if not gate["identity_exact"]:
                    return 2
                if not gate["wout_exact"]:
                    return 3
                if not gate["diagnostics_exact"]:
                    return 4
                if (
                    not gate["actual_indata_exact"]
                    or not gate["captures_complete"]
                    or not gate["harness_exact"]
                ):
                    return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
