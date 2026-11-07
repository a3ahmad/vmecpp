"""Run an already deployed replica, allowing only an idle desktop GPU context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from tools.optimization.deploy_secondary_solver import (
    CUDART_SHA256,
    REFERENCE_SHA256,
    REMOTE_PARENT,
    digest,
    save,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != REMOTE_PARENT / "resident-solver4":
        message = "Unexpected secondary replica directory"
        raise ValueError(message)
    relocation = json.loads((root / "relocation.json").read_text())
    for item in relocation:
        if digest(root / item["path"]) != item["after"]:
            message = "Relocated binary changed: " + item["path"]
            raise ValueError(message)
    extension = (
        root
        / "candidate-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    )
    if digest(extension) != REFERENCE_SHA256:
        message = "Reference extension changed"
        raise ValueError(message)
    if digest(REMOTE_PARENT / "lib/libcudart.so.12") != CUDART_SHA256:
        message = "Pinned CUDA runtime changed"
        raise ValueError(message)
    gpu_status = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    desktop_contexts = []
    other_contexts = []
    for line in processes.splitlines():
        pid, name, memory = (part.strip() for part in line.split(","))
        target = (
            desktop_contexts
            if (
                name == "/usr/libexec/gnome-remote-desktop-daemon"
                and int(memory) <= 512
            )
            else other_contexts
        )
        target.append({"pid": int(pid), "name": name, "memory_mib": int(memory)})
    save(
        root / "resume-gpu-status.json",
        {
            "status": gpu_status,
            "idle_desktop_contexts": desktop_contexts,
            "other_contexts": other_contexts,
        },
    )
    if other_contexts or any(
        int(line.split(",")[1]) for line in gpu_status.splitlines()
    ):
        message = "Secondary GPU has active work; no GPU job started"
        raise RuntimeError(message)
    source_comparison = []
    for path in sorted((root / "stella-snapshot").rglob("*")):
        if path.is_file():
            relative = path.relative_to(root / "stella-snapshot")
            existing = REMOTE_PARENT / "stella" / relative
            source_comparison.append(
                {
                    "path": str(relative),
                    "expected_sha256": digest(path),
                    "existing_sha256": digest(existing) if existing.is_file() else None,
                }
            )
    save(root / "resume-source-comparison.json", source_comparison)
    stella = REMOTE_PARENT / "stella"
    if any(x["expected_sha256"] != x["existing_sha256"] for x in source_comparison):
        stella = root / "stella-snapshot"
    commands = []
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("VMECPP_") or key == "LD_PRELOAD":
            environment.pop(key)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1", LD_LIBRARY_PATH=str(REMOTE_PARENT / "lib")
    )

    def run(command: list[str], name: str) -> int:
        commands.append({"name": name, "command": command})
        save(root / "resume-commands.json", commands)
        with (root / "logs" / (name + ".log")).open("x") as output:
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            message = "Secondary validation failed: " + name
            raise RuntimeError(message)
        return result.returncode

    run([str(root / "bin/solver-replay"), str(root / "captures")], "ordered-replay")
    flags = {
        "VMECPP_GPU_SOLVER": "1",
        "VMECPP_FORWARD_REFERENCE_SO": str(extension),
        "VMECPP_GPU_SOLVER_RESIDENT_CONTROL": "1",
        "VMECPP_GPU_RESIDUAL_ROUNDS": "12",
        "VMECPP_GPU_RESIDUAL_GRAPH": "1",
        "VMECPP_GPU_DIAGNOSTICS_ROUNDS": "16",
        "VMECPP_GPU_DIAGNOSTICS_GRAPH": "1",
    }
    for key in (
        "VMECPP_GPU_RESIDUAL_ROUNDS",
        "VMECPP_GPU_RESIDUAL_GRAPH",
        "VMECPP_GPU_DIAGNOSTICS_ROUNDS",
        "VMECPP_GPU_DIAGNOSTICS_GRAPH",
    ):
        environment[key] = flags[key]
    run([str(root / "bin/solver-replay"), str(root / "captures")], "optimistic-replay")
    command = [
        str(REMOTE_PARENT / "baseline/bin/python"),
        "-B",
        "-m",
        "tools.optimization.validate_resident_solver",
        "--python",
        str(root / "candidate-env/bin/python"),
        "--stella",
        str(stella),
        "--manifest",
        str(root / "fixtures/remote-manifest.json"),
        "--reference",
        str(root / "reference/report.json"),
        "--reference-trace",
        str(root / "reference/cpu.trace.jsonl"),
        "--out",
        str(root / "smoke"),
        "--preload",
        str(root / "lib/solver-shim.so"),
        "--cpu",
        str(min(os.sched_getaffinity(0))),
    ]
    for key, value in flags.items():
        command.extend(("--env", key + "=" + value))
    status = run(command, "strict-smoke")
    save(
        root / "deployment-status.json",
        {
            "smoke_exit_status": status,
            "stella": str(stella),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
