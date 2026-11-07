"""Build/check a strict GPU replay of actually observed damping arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(number: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", number))[0]


def value(word: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", word))[0]


def log_pairs(path: Path) -> list[tuple[int, int]]:
    data = path.read_bytes()
    require(len(data) % 448 == 0, "Truncated actual observer records")
    result = []
    for offset in range(0, len(data), 448):
        row = struct.unpack_from("<8Q", data, offset)
        require(row[0] == 0x564D45434C4F4731, "Observer magic mismatch")
        symbol = data[offset + 320 : offset + 448].split(b"\0")[0]
        if symbol == b"_ZN6vmecpp4Vmec6EvolveENS_14VmecCheckpointEidiRb":
            result.append((row[4], row[5]))
    require(bool(result), "Actual Evolve log corpus absent")
    return result


def verify_cpu(events: list[dict], logs: list[tuple[int, int]]) -> dict:
    history: list[float] = []
    previous = 0.0
    index = 0
    resets = zeros = 0
    for event in events:
        fac, b1, dt_bits, fsq_bits = event["step_bits"]
        dt, fsq = value(dt_bits), value(fsq_bits)
        require(
            math.isfinite(dt) and dt > 0 and math.isfinite(fsq),
            "Nonfinite/invalid observed input",
        )
        require(event["iter2"] >= event["iter1"], "Invalid observed iteration counters")
        if event["iter2"] == event["iter1"]:
            history = [0.15 / dt] * 10
            resets += 1
        require(len(history) == 10, "Observed history has no initialization")
        history = history[1:] + history[:1]
        if event["iter2"] > event["iter1"]:
            numerator = 0.0
            if fsq != 0.0:
                require(
                    index < len(logs) and previous != 0.0,
                    "Missing observed log or zero denominator",
                )
                argument, logarithm = logs[index]
                require(
                    bits(fsq / previous) == argument,
                    "Observed ratio/log correspondence mismatch",
                )
                numerator = min(abs(value(logarithm)), 0.15)
                index += 1
            else:
                zeros += 1
            history[9] = numerator / dt
        previous = fsq
        require(
            [bits(x) for x in history] == event["inv_tau_bits"],
            "CPU history reconstruction mismatch",
        )
        total = 0.0
        for item in history:
            total += item
        dtau = dt * (total / 10.0) / 2.0
        require(
            (bits(1.0 / (1.0 + dtau)), bits(1.0 - dtau)) == (fac, b1),
            "CPU coefficient reconstruction mismatch",
        )
    require(index == len(logs), "Unused observed log records")
    return {
        "original_history_and_fac_b1_bit_exact": True,
        "events": len(events),
        "actual_Evolve_logs": index,
        "history_resets": resets,
        "zero_fsq_events": zeros,
    }


def build(args) -> None:
    require(
        len(args.controller) == len(args.log_records),
        "Pair each controller with its actual log",
    )
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / "source"
    source.mkdir()
    for name in ("gpu_controller_replay.cu", "build_controller_replay.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    cases = []
    for index, (controller, log) in enumerate(
        zip(args.controller, args.log_records, strict=True)
    ):
        events = [
            json.loads(line) for line in controller.read_text().splitlines() if line
        ]
        logs = log_pairs(log)
        require(0 < len(events) <= 100000, "Observed corpus exceeds bounded replay")
        cpu = verify_cpu(events, logs)
        binary = args.out / f"case{index}.bin"
        with binary.open("wb") as output:
            output.write(b"VMECCTR1" + struct.pack("<QQ", len(events), len(logs)))
            for event in events:
                fac, b1, dt, fsq = event["step_bits"]
                output.write(
                    struct.pack(
                        "<4i14Q",
                        event["ns"],
                        event["iter1"],
                        event["iter2"],
                        int(event["iter1"] == event["iter2"]),
                        fsq,
                        dt,
                        fac,
                        b1,
                        *event["inv_tau_bits"],
                    )
                )
            for row in logs:
                output.write(struct.pack("<QQ", *row))
        cases.append(
            {
                "controller": str(controller),
                "controller_sha256": digest(controller),
                "log_records": str(log),
                "log_records_sha256": digest(log),
                "input": str(binary),
                "input_sha256": digest(binary),
                "CPU_preflight": cpu,
            }
        )
    command = [
        str(args.cuda / "bin/nvcc"),
        "-std=c++20",
        "-O3",
        "--fmad=false",
        "-gencode",
        "arch=compute_80,code=sm_80",
        "-gencode",
        "arch=compute_120,code=sm_120",
        str(source / "gpu_controller_replay.cu"),
        "-o",
        str(args.out / "controller-replay"),
    ]
    (args.out / "build-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (args.out / "build.log").open("w") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    manifest = {
        "scope": "Fixed observed damping arithmetic, not autonomous solver or all SolveLoop branches",
        "build_exit_code": completed.returncode,
        "GPU_executions_by_builder": 0,
        "cases": cases,
        "source_sha256": {p.name: digest(p) for p in source.iterdir()},
        "limits": [
            "No generic CPU/CUDA log error bound established",
            "No live solver integration",
            "All six-case coverage and generic certificate/fallback remain future gates",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    require(
        completed.returncode == 0, "GPU replay compilation failed; preserve build.log"
    )
    manifest["binary_sha256"] = digest(args.out / "controller-replay")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def check(args) -> None:
    data, observed = args.input.read_bytes(), args.output.read_bytes()
    require(
        data[:8] == b"VMECCTR1" and observed[:8] == b"VMECGCR1", "Replay magic mismatch"
    )
    count, log_count = struct.unpack_from("<QQ", data, 8)
    out_count, modes = struct.unpack_from("<QQ", observed, 8)
    require(count == out_count and modes == 2, "Output count/mode mismatch")
    require(len(data) == 24 + count * 128 + log_count * 16, "Input framing mismatch")
    require(len(observed) == 24 + 2 * count * 128, "Output framing mismatch")
    events = [struct.unpack_from("<4i14Q", data, 24 + i * 128) for i in range(count)]
    outputs = [
        [
            struct.unpack_from("<15Q2I", observed, 24 + (mode * count + i) * 128)
            for i in range(count)
        ]
        for mode in range(2)
    ]
    changed_steps, changed_history, changed_logs, changed_numerators = [], [], [], []
    log_calls = 0
    for i, (event, reference, candidate) in enumerate(
        zip(events, *outputs, strict=True)
    ):
        require(reference[16] == 0 and candidate[16] == 0, "Device error flags nonzero")
        require(
            reference[:2] == event[6:8] and reference[2:12] == event[8:18],
            "GPU literal CPU-log arithmetic changed original bits",
        )
        require(reference[15] == candidate[15], "Different log control flow")
        if candidate[:2] != reference[:2]:
            changed_steps.append(
                {
                    "event": i,
                    "original_fac_b1_bits": reference[:2],
                    "native_fac_b1_bits": candidate[:2],
                }
            )
        if candidate[2:12] != reference[2:12]:
            changed_history.append(i)
        if reference[15]:
            log_calls += 1
            require(reference[12] == candidate[12], "GPU ratio argument changed")
            if reference[13] != candidate[13]:
                changed_logs.append(
                    {
                        "event": i,
                        "argument_bits": reference[12],
                        "CPU_log_bits": reference[13],
                        "CUDA_log_bits": candidate[13],
                    }
                )
            if reference[14] != candidate[14]:
                changed_numerators.append(i)
    require(log_calls == log_count, "Output log count mismatch")
    result = {
        "scope": "Actual GPU fixed observed damping replay; not an autonomous solver or performance evidence",
        "events": count,
        "log_calls": log_calls,
        "GPU_CPU_log_arithmetic_all_bits_exact": True,
        "native_log_raw_differences": changed_logs,
        "native_log_numerator_differing_events": changed_numerators,
        "native_log_history_differing_events": changed_history,
        "native_log_changed_step_events": changed_steps,
        "native_log_changed_step_count": len(changed_steps),
        "history_differing_event_count": len(changed_history),
        "input_sha256": digest(args.input),
        "output_sha256": digest(args.output),
        "limitations": [
            "Observed trajectory inputs held fixed",
            "No all-case or generic log interval certificate",
            "If a coefficient changes, subsequent observed physics is counterfactual",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build_parser = modes.add_parser("build")
    build_parser.add_argument("--controller", type=Path, action="append", required=True)
    build_parser.add_argument(
        "--log-records", type=Path, action="append", required=True
    )
    build_parser.add_argument("--cuda", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, required=True)
    check_parser = modes.add_parser("check")
    for name in ("input", "output", "out"):
        check_parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    (build if args.mode == "build" else check)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
