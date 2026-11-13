"""Bounded exclusive-GPU expansion screening on SHA-bound exported operands."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--shared-owner", type=Path, required=True)
    parser.add_argument("--driver-libs", type=Path, required=True)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--terms", choices=("full", "triangular"), default="full")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--backend", default="wmma")
    parser.add_argument("--group-diagonals", action="store_true")
    parser.add_argument("--split", choices=("row", "warp", "quantized"))
    parser.add_argument("--graph-batch", type=int, default=0)
    parser.add_argument("--digits", type=int, default=0)
    parser.add_argument("--k-block", type=int, default=0)
    parser.add_argument("--reuse-b", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.repeats <= 4
    assert 1 <= args.calls <= 100
    jobs = {j["name"]: j for j in json.loads(args.manifest.read_text())["jobs"]}
    assert all(name in jobs for name in args.job)
    assert digest(args.binary) == args.binary_sha256
    if not args.execute:
        print(json.dumps({"phase": "PLAN", "jobs": args.job, "GPU_calls": 0}))  # noqa: T201
        return
    args.out.mkdir(parents=True, exist_ok=False)
    expected = json.loads(args.owner.read_text())
    rows = []
    status = {"complete": False, "rows": rows, "solver_qualified": False}
    try:
        for repetition in range(args.repeats):
            names = args.job if repetition % 2 == 0 else args.job[::-1]
            for name in names:
                live = json.loads(args.shared_owner.read_text())
                assert live["claim_token"] == expected["claim_token"]
                assert live["project"] == "vmecpp"
                assert live["gpu_uuid"] == expected["gpu_uuid"]
                source = args.manifest.parent / jobs[name]["file"]
                assert digest(source) == jobs[name]["sha256"]
                assert digest(args.binary) == args.binary_sha256
                label = f"r{repetition}-{name}"
                output = args.out / label
                output.mkdir()
                command = [
                    str(args.binary),
                    "--execute",
                    "--input",
                    str(source),
                    "--mode",
                    args.mode,
                    "--calls",
                    str(args.calls),
                    "--backend",
                    args.backend,
                    "--digits",
                    str(args.digits),
                    "--k-block",
                    str(args.k_block),
                    "--samples",
                    "7",
                    "--warmups",
                    "3",
                    "--terms",
                    args.terms,
                    "--expect-cublas",
                    "/usr/local/cuda-13.5/lib64/libcublas.so",
                    "--report",
                    str(output / "report"),
                    "--output",
                    str(output / "output"),
                ]
                if args.reuse_b:
                    command.append("--reuse-b")
                if args.group_diagonals:
                    command.append("--group-diagonals")
                if args.split:
                    command.extend(["--split", args.split])
                if args.graph_batch:
                    command.extend(["--graph-batch", str(args.graph_batch)])
                env = {
                    "PATH": "/usr/bin:/bin",
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "LD_LIBRARY_PATH": f"{args.driver_libs}:/usr/lib/wsl/lib:/usr/local/cuda-13.5/lib64",
                }
                print(  # noqa: T201
                    "VMECPP_QPC_MARKER "
                    + json.dumps({"phase": "case_begin", "case": label}),
                    flush=True,
                )
                with (
                    (output / "stdout.log").open("w") as stdout,
                    (output / "stderr.log").open("w") as stderr,
                ):
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    try:
                        code = process.wait(timeout=120)
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=8)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                        raise
                print(  # noqa: T201
                    "VMECPP_QPC_MARKER "
                    + json.dumps(
                        {"phase": "case_end", "case": label, "exit_code": code}
                    ),
                    flush=True,
                )
                rows.append(
                    {
                        "label": label,
                        "case": name,
                        "repetition": repetition,
                        "exit_code": code,
                        "input_sha256": digest(source),
                        "binary_sha256": args.binary_sha256,
                        "command": command,
                        "output": str(output),
                    }
                )
                assert code == 0, output
        status["complete"] = True
    finally:
        (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
