"""Run an isolated SM copy calibration with optional actual GPU telemetry."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

TELEMETRY_COLUMNS = (
    "timestamp",
    "uuid",
    "pci.bus_id",
    "name",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
)


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".temporary")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def telemetry(path: Path, stop: threading.Event) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(TELEMETRY_COLUMNS),
        "--format=csv,noheader,nounits",
    ]
    with path.open("w", encoding="ascii") as log:
        while not stop.is_set():
            started_at = time.time()
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=3, check=False
                )
                record = {
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "exit_status": result.returncode,
                    "raw_csv": result.stdout,
                    "stderr": result.stderr[:400],
                }
            except (OSError, subprocess.TimeoutExpired) as error:
                record = {
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "available": False,
                    "error": str(error),
                }
            log.write(json.dumps(record, ensure_ascii=True) + "\n")
            log.flush()
            stop.wait(0.2)


def audit(report: dict, passes: int, samples: int) -> dict:
    assert report["passes_per_sample"] == passes
    assert report["sample_rounds"] == samples
    assert len(report["samples"]) == 6 * samples
    regimes = {regime["name"]: regime for regime in report["regimes"]}
    assert len(regimes) == 3
    assert regimes["tiny"]["working_set_bytes"] == 1 << 20
    assert regimes["l2_candidate"]["working_set_bytes"] <= (
        report["device"]["l2_bytes"] / 2
    )
    assert regimes["dram_candidate"]["working_set_bytes"] >= (
        4 * report["device"]["l2_bytes"]
    )
    counts = Counter()
    seen = set()
    groups = {}
    for regime in regimes.values():
        assert regime["allocated_input_bytes"] == regime["allocated_output_bytes"]
        assert regime["working_set_bytes"] == 2 * regime["allocated_input_bytes"]
        assert regime["logical_bytes_per_full_pass"] == regime["working_set_bytes"]
        assert (
            regime["logical_bytes_per_sample"] == regime["working_set_bytes"] * passes
        )
        assert regime["graph_kernel_nodes"] == passes
    for sample in report["samples"]:
        regime = regimes[sample["regime"]]
        assert sample["all_words_verified"] is True
        assert sample["checksum_u64"] == regime["expected_checksum_u64"]
        assert math.isfinite(sample["cuda_event_ms"])
        assert sample["cuda_event_ms"] > 0
        assert sample["mode"] in ("direct", "graph")
        assert 0 <= sample["round"] < samples
        assert 0 <= sample["order"] < 6
        position = (sample["round"], sample["order"])
        assert position not in seen
        seen.add(position)
        expected_bandwidth = regime["logical_bytes_per_sample"] / (
            sample["cuda_event_ms"] * 1e6
        )
        assert math.isclose(
            sample["logical_gigabytes_per_second"], expected_bandwidth, rel_tol=1e-12
        )
        key = (sample["regime"], sample["mode"])
        counts[key] += 1
        groups.setdefault(key, []).append(sample)
    assert len(counts) == 6
    assert all(count == samples for count in counts.values())
    for round_index in range(samples):
        assert (
            len(
                {
                    (sample["regime"], sample["mode"])
                    for sample in report["samples"]
                    if sample["round"] == round_index
                }
            )
            == 6
        )
    return {
        "scope": "Derived medians of measured logical SM copy samples; "
        "not physical bandwidth or achieved solver roofline",
        "all_outputs_exact": True,
        "groups": [
            {
                "regime": key[0],
                "mode": key[1],
                "measured_samples": len(values),
                "median_cuda_event_ms": statistics.median(
                    sample["cuda_event_ms"] for sample in values
                ),
                "median_logical_gigabytes_per_second": statistics.median(
                    sample["logical_gigabytes_per_second"] for sample in values
                ),
            }
            for key, values in sorted(groups.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--passes", type=int, default=128)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    command = [
        str(args.binary.resolve()),
        "--device",
        str(args.device),
        "--passes",
        str(args.passes),
        "--samples",
        str(args.samples),
        "--warmups",
        str(args.warmups),
    ]
    plan = {
        "phase": "plan",
        "command": command,
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timeout_seconds": args.timeout,
        "gpu_execution_requested": args.execute,
        "telemetry_columns": TELEMETRY_COLUMNS,
        "telemetry_scope": "All nvidia-smi devices, mapped by UUID/PCI to actual "
        "CUDA device in report; raw actual readings, not advertised specifications",
    }
    if not args.execute:
        sys.stdout.write(json.dumps(plan, indent=2) + "\n")
        return 0
    if args.timeout < 1:
        parser.error("timeout must be positive")
    if os.name != "posix":
        parser.error("execution requires the Linux CUDA build")
    args.out.mkdir(parents=True, exist_ok=False)
    plan.update(phase="running", started_at=time.time())
    save(args.out / "status.json", plan)
    stop = threading.Event()
    monitor = threading.Thread(
        target=telemetry, args=(args.out / "telemetry.jsonl", stop)
    )
    monitor.start()
    code: int | str = "not_started"
    try:
        with (
            (args.out / "report.json").open("w") as output,
            (args.out / "stderr.log").open("w") as error,
        ):
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=error,
                start_new_session=True,
            )
            plan["pid"] = process.pid
            save(args.out / "status.json", plan)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = "timeout"
            except BaseException:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = "interrupted"
                raise
        if code == 0:
            report = json.loads((args.out / "report.json").read_text())
            save(args.out / "summary.json", audit(report, args.passes, args.samples))
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        AssertionError,
        KeyError,
        TypeError,
        ArithmeticError,
    ) as error:
        plan["error"] = str(error)
        if code == 0:
            code = "audit_failed"
    finally:
        stop.set()
        monitor.join(timeout=5)
        readings = [
            json.loads(line)
            for line in (args.out / "telemetry.jsonl").read_text().splitlines()
        ]
        plan.update(
            phase="complete" if code == 0 else "failed",
            exit_status=code,
            finished_at=time.time(),
            telemetry_thread_stopped=not monitor.is_alive(),
            telemetry_queries=len(readings),
            telemetry_successful_queries=sum(
                reading.get("exit_status") == 0 and bool(reading.get("raw_csv"))
                for reading in readings
            ),
        )
        save(args.out / "status.json", plan)
    sys.stdout.write(json.dumps(plan) + "\n")
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
