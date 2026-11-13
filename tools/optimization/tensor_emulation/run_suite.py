"""Run a bounded, exclusive-GPU component comparison; PLAN by default."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--shared-owner", type=Path, required=True)
    parser.add_argument("--driver-libs", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument(
        "--build-suffix", choices=("build1", "build2", "build3"), default="build1"
    )
    parser.add_argument(
        "--versions", choices=("13.1", "13.5"), nargs="+", default=["13.1", "13.5"]
    )
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.repeats <= 3
    manifest = json.loads(args.manifest.read_text())
    jobs = {job["name"]: job for job in manifest["jobs"]}
    assert all(name in jobs for name in args.job)
    if not args.execute:
        print(  # noqa: T201
            json.dumps(
                {
                    "phase": "PLAN",
                    "jobs": args.job,
                    "repeats": args.repeats,
                    "GPU_calls": 0,
                }
            )
        )
        return
    args.out.mkdir(parents=True, exist_ok=False)
    expected = json.loads(args.owner.read_text())

    def ownership():
        live = json.loads(args.shared_owner.read_text())
        assert live["project"] == "vmecpp"
        assert live["claim_token"] == expected["claim_token"]
        assert live["gpu_uuid"] == expected["gpu_uuid"]

    rows = []
    status = {
        "complete": False,
        "rows": rows,
        "scope": "component GEMM alternative, not solver metrics",
    }
    try:
        for repetition in range(args.repeats):
            versions = args.versions if repetition % 2 == 0 else args.versions[::-1]
            for name in args.job:
                job = jobs[name]
                source = args.manifest.parent / job["file"]
                assert hashlib.sha256(source.read_bytes()).hexdigest() == job["sha256"]
                for version in versions:
                    ownership()
                    label = f"r{repetition}-cuda{version}-{name}"
                    output = args.out / label
                    output.mkdir()
                    toolkit = Path("/usr/local") / ("cuda-" + version)
                    binary = (
                        args.build_root
                        / ("tensor-emulation-" + version + "-" + args.build_suffix)
                        / "bench"
                    )
                    receipt = json.loads(
                        (binary.parent / "build-receipt.json").read_text()
                    )
                    assert (
                        hashlib.sha256(binary.read_bytes()).hexdigest()
                        == receipt["binary_sha256"]
                    )
                    env = {
                        "PATH": "/usr/bin:/bin",
                        "HOME": "/tmp",
                        "LANG": "C.UTF-8",
                        "CUDA_VISIBLE_DEVICES": "0",
                        "LD_LIBRARY_PATH": f"{args.driver_libs}:/usr/lib/wsl/lib:{toolkit / 'lib64'}",
                    }
                    command = [
                        str(binary),
                        "--execute",
                        "--input",
                        str(source),
                        "--mode",
                        "all",
                        "--calls",
                        "50",
                        "--samples",
                        "7",
                        "--warmups",
                        "5",
                        "--workspace-mib",
                        "8192",
                        "--expect-cublas",
                        str(toolkit / "lib64/libcublas.so"),
                        "--report",
                        str(output / "report"),
                        "--output",
                        str(output / "output"),
                    ]
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
                            code = process.wait(timeout=90)
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
                            "version": version,
                            "repetition": repetition,
                            "exit_code": code,
                            "output": str(output),
                            "command": command,
                            "finished_epoch": time.time(),
                        }
                    )
                    assert code == 0, output
        status["complete"] = True
    finally:
        (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
