"""Execute a serial fixture matrix and compare exact metrics across samples."""

from __future__ import annotations

# This command-line tool emits progress for long-running measurements.
# ruff: noqa: T201
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stella", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seeds", type=Path, nargs="+")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("at least two samples required")
    args.out.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    # Pin every child to the same currently allowed logical processor.
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    results = []
    seeds = args.seeds or [
        args.stella / "artifacts" / "seeds" / f"public_{name}_seed.json"
        for name in ("p3", "p1", "p2")
    ]
    for sample in range(args.samples):
        for seed_path in seeds:
            seed = seed_path.stem
            for track in ("P1", "P3"):
                destination = args.out / f"{seed}-{track}-{sample:02d}"
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("benchmark.py")),
                    "--stella",
                    str(args.stella),
                    "--seed",
                    str(seed_path),
                    "--track",
                    track,
                    "--out",
                    str(destination),
                ]
                started = time.perf_counter()
                with (args.out / f"{destination.name}.log").open("w") as log:
                    try:
                        completed = subprocess.run(
                            command,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=1800,
                            check=False,
                        )
                        status = completed.returncode
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                report_path = destination / "report.json"
                report = (
                    json.loads(report_path.read_text()) if report_path.exists() else {}
                )
                results.append(
                    {
                        "seed": seed,
                        "track": track,
                        "sample": sample,
                        "warmup": sample == 0,
                        "exit_status": status,
                        "process_wall_seconds": time.perf_counter() - started,
                        "report": report,
                    }
                )
                (args.out / "samples.json").write_text(json.dumps(results, indent=2))
                print(seed, track, sample, status, flush=True)
    return 0 if all(row["exit_status"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
