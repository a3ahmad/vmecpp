"""Replicate frozen solver 4 in a new, isolated secondary-host subtree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

REMOTE_PARENT = Path("/bulk/vmecpp-optimization-20261103")
REFERENCE_SHA256 = "144e4f204cbb0569edd25710cf8ef4c0e618b0e7a60e33cf3a556066e8a06da8"
CUDART_SHA256 = "381c0faae2678098eb2aef376ff04ef7129d86e28d0cc0fd2c83d4b6d75c0098"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(workspace: Path, stella: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    optimization = workspace / "build/optimization"

    def copy(source: Path, relative: str) -> None:
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for name, source in {
        "solver-shim.so": "resident-solver4/solver-shim.so",
        "solver-backend.so": "resident-solver4/solver-backend.so",
        "resident.so": "resident-device-inverse/resident.so",
        "iteration.so": "resident-iteration17/iteration.so",
    }.items():
        copy(optimization / source, "lib/" + name)
    copy(optimization / "resident-solver4/solver-replay", "bin/solver-replay")
    for path in (optimization / "resident-solver4/source").iterdir():
        if path.is_file():
            copy(path, "source/" + path.name)
    for name in ("commands.json", "source-hashes.json", "replay-status.json"):
        copy(optimization / "resident-solver4" / name, "provenance/" + name)
    for name in (
        "benchmark.py",
        "summarize.py",
        "validate_resident_solver.py",
        "validate_resident_lifecycle.py",
        "deploy_secondary_solver.py",
    ):
        copy(workspace / "tools/optimization" / name, "tools/optimization/" + name)
    for path in (optimization / "fixtures-frozen").glob("*.json"):
        copy(path, "fixtures/" + path.name)
    for path in (optimization / "combined-local/baseline-0").glob("*.json"):
        if path.name == "report.json" or path.name.startswith("actual_indata_"):
            copy(path, "reference/" + path.name)
    copy(
        optimization / "resident-solver1/cpu-trace/public_p3_seed-P3-00.trace.jsonl",
        "reference/cpu.trace.jsonl",
    )
    package = optimization / "gpu-env/lib/python3.12/site-packages/vmecpp"
    for path in package.rglob("*"):
        if (
            path.is_file()
            and path.suffix not in (".so", ".pyc")
            and "__pycache__" not in path.parts
        ):
            copy(path, "python/vmecpp/" + str(path.relative_to(package)))
    for path in (stella / "src/cga").rglob("*"):
        if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts:
            copy(path, "stella-snapshot/" + str(path.relative_to(stella)))
    copy(stella / "configs/oracle.yaml", "stella-snapshot/configs/oracle.yaml")
    captures = (
        "real-transform.bin",
        "resident-fields/real-fields.bin",
        "resident-constants/real-fields.bin.constants.bin",
        "resident-constraints-capture/real-fields.bin.constraints.bin",
        "a100-fused-results/fused/real-pipeline.bin",
        "resident-iteration10/real-preparation.bin",
        "resident-iteration10/real-iteration.bin",
        "resident-iteration10/real-step.bin",
        "resident-iteration10/real-diagnostics.bin",
        "resident-preconditioner-capture-retry/real-preconditioner.bin",
    )
    for name in captures:
        copy(optimization / name, "captures/" + name)
    manifest = {
        "purpose": "strict cross-host correctness gate; no speedup claim",
        "reference_sha256": REFERENCE_SHA256,
        "cudart_sha256": CUDART_SHA256,
        "files": {
            str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    }
    save(out / "payload-manifest.json", manifest)
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=str(path.relative_to(out)), recursive=False)
    save(
        out.with_suffix(".archive.json"),
        {
            "archive": str(archive),
            "sha256": digest(archive),
            "bytes": archive.stat().st_size,
        },
    )


def deploy(root: Path) -> None:
    if root.resolve() != REMOTE_PARENT / "resident-solver4":
        msg = "Unexpected secondary deployment directory"
        raise ValueError(msg)
    manifest = json.loads((root / "payload-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if root not in path.resolve().parents or digest(path) != expected:
            raise ValueError("Payload path or hash mismatch: " + name)
    logs = root / "logs"
    logs.mkdir(exist_ok=False)
    commands = []
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["LD_LIBRARY_PATH"] = str(REMOTE_PARENT / "lib")
    environment["PIP_CONFIG_FILE"] = "/dev/null"

    def run(command: list[str], name: str, *, check: bool = True) -> int:
        commands.append({"name": name, "command": command})
        save(root / "deployment-commands.json", commands)
        with (logs / (name + ".log")).open("w") as output:
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if check and result.returncode != 0:
            raise RuntimeError("Deployment command failed: " + name)
        return result.returncode

    reference = REMOTE_PARENT / "gpu-extension.so"
    if digest(reference) != REFERENCE_SHA256:
        msg = "Existing reference extension changed"
        raise ValueError(msg)
    if digest(REMOTE_PARENT / "lib/libcudart.so.12") != CUDART_SHA256:
        msg = "Existing pinned CUDA runtime changed"
        raise ValueError(msg)
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
    save(root / "stella-source-comparison.json", source_comparison)
    stella = REMOTE_PARENT / "stella"
    if any(x["expected_sha256"] != x["existing_sha256"] for x in source_comparison):
        stella = root / "stella-snapshot"
    python = str(REMOTE_PARENT / "baseline/bin/python")
    run([python, "-B", "-m", "venv", str(root / "tool-env")], "create-tool-env")
    run(
        [
            str(root / "tool-env/bin/python"),
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            "--index-url",
            "https://pypi.org/simple",
            "--quiet",
            "patchelf",
        ],
        "install-isolated-patchelf",
    )
    run(
        [python, "-B", "-m", "venv", str(root / "candidate-env")],
        "create-candidate-env",
    )
    site = root / "candidate-env/lib/python3.12/site-packages"
    shutil.copytree(root / "python/vmecpp", site / "vmecpp")
    (site / "oracle.pth").write_text(
        str(REMOTE_PARENT / "baseline/lib/python3.12/site-packages") + "\n"
    )
    extension = site / "vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    shutil.copy2(reference, extension)
    patchelf = str(root / "tool-env/bin/patchelf")
    relocation = []
    binaries = [*sorted((root / "lib").glob("*.so")), root / "bin/solver-replay"]
    for path in binaries:
        before = digest(path)
        dynamic = subprocess.run(
            ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
        ).stdout
        for needed in re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic):
            if needed.startswith("/mnt/f/"):
                replacement = (
                    str(extension) if "_vmecpp" in needed else Path(needed).name
                )
                run(
                    [patchelf, "--replace-needed", needed, replacement, str(path)],
                    "relocate-" + path.name + "-" + Path(needed).name,
                )
        origin = "$ORIGIN" if path.parent.name == "lib" else "$ORIGIN/../lib"
        rpath = ":".join((origin, str(extension.parent), str(REMOTE_PARENT / "lib")))
        run([patchelf, "--set-rpath", rpath, str(path)], "rpath-" + path.name)
        relocation.append(
            {
                "path": str(path.relative_to(root)),
                "before": before,
                "after": digest(path),
            }
        )
        dependencies = subprocess.run(
            ["ldd", str(path)],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        (logs / (path.name + ".ldd.log")).write_text(
            dependencies.stdout + dependencies.stderr
        )
        if "not found" in dependencies.stdout:
            raise RuntimeError("Unresolved copied library: " + path.name)
    save(root / "relocation.json", relocation)
    if digest(extension) != REFERENCE_SHA256:
        msg = "Candidate extension changed during deployment"
        raise ValueError(msg)
    fixtures = json.loads((root / "fixtures/manifest.json").read_text())
    for fixture in fixtures:
        fixture["path"] = str(root / "fixtures" / Path(fixture["path"]).name)
    save(root / "fixtures/remote-manifest.json", fixtures)
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
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    save(
        root / "pre-run-gpu.json",
        {"status": gpu_status, "compute_processes": processes},
    )
    if processes or any(
        int(line.split(",")[1]) != 0 for line in gpu_status.splitlines()
    ):
        msg = "Secondary GPU is busy; no GPU job started"
        raise RuntimeError(msg)
    for key in tuple(environment):
        if key.startswith("VMECPP_") or key == "LD_PRELOAD":
            environment.pop(key)
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
    environment.update(
        {
            key: value
            for key, value in flags.items()
            if key
            not in (
                "VMECPP_GPU_SOLVER",
                "VMECPP_FORWARD_REFERENCE_SO",
                "VMECPP_GPU_SOLVER_RESIDENT_CONTROL",
            )
        }
    )
    run([str(root / "bin/solver-replay"), str(root / "captures")], "optimistic-replay")
    command = [
        python,
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
    status = run(command, "strict-smoke", check=False)
    save(
        root / "deployment-status.json",
        {"smoke_exit_status": status, "stella": str(stella)},
    )
    if status:
        msg = "Secondary strict smoke did not pass"
        raise RuntimeError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "deploy"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--stella", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.workspace is None or args.stella is None or args.out is None:
            parser.error("prepare requires --workspace, --stella and --out")
        prepare(args.workspace.resolve(), args.stella.resolve(), args.out.resolve())
    else:
        if args.root is None:
            parser.error("deploy requires --root")
        deploy(args.root.resolve())


if __name__ == "__main__":
    main()
