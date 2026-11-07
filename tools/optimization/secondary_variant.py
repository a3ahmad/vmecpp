"""Prepare isolated frozen solver variants and reproducible secondary campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

from tools.optimization.campaign_gates import clean_environment
from tools.optimization.deploy_secondary_solver import (
    REFERENCE_SHA256,
    REMOTE_PARENT,
    digest,
    save,
)

BASE = REMOTE_PARENT / "resident-solver4"
ALLOWED = ("resident-solver4", "resident-solver5-packed", "resident-solver8-capture")


def package(workspace: Path, snapshot: Path, out: Path) -> None:
    if (
        snapshot.name not in ALLOWED
        or snapshot.parent != workspace / "build/optimization"
    ):
        message = "Expected an explicitly frozen solver snapshot"
        raise ValueError(message)
    out.mkdir(parents=True, exist_ok=False)
    optimization = workspace / "build/optimization"

    def copy(source: Path, name: str) -> None:
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for name, path in {
        "solver-shim.so": snapshot / "solver-shim.so",
        "solver-backend.so": snapshot / "solver-backend.so",
        "resident.so": optimization / "resident-device-inverse/resident.so",
        "iteration.so": optimization / "resident-iteration17/iteration.so",
    }.items():
        copy(path, "lib/" + name)
    copy(snapshot / "solver-replay", "bin/solver-replay")
    for path in (snapshot / "source").iterdir():
        if path.is_file():
            copy(path, "source/" + path.name)
    for name in ("commands.json", "source-hashes.json"):
        copy(snapshot / name, "provenance/" + name)
    for name in (
        "benchmark.py",
        "campaign_gates.py",
        "paired.py",
        "report_pairs.py",
        "summarize.py",
        "validate_resident_solver.py",
        "secondary_variant.py",
        "deploy_secondary_solver.py",
    ):
        copy(workspace / "tools/optimization" / name, "tools/optimization/" + name)
    fixtures = json.loads((optimization / "fixtures-frozen/manifest.json").read_text())
    for fixture in fixtures:
        seed = Path(fixture["path"])
        for track in ("P1", "P3"):
            original = (
                optimization
                / "a100-fused-results/paired-combined/baseline"
                / f"{seed.stem}-{track}-00"
            )
            for path in original.glob("*.json"):
                if path.name in (
                    "report.json",
                    "provenance.json",
                ) or path.name.startswith("actual_indata_"):
                    copy(path, f"references/{seed.stem}-{track}/{path.name}")
    original_payload = json.loads(
        (optimization / "secondary-solver4-payload/payload-manifest.json").read_text()
    )
    original_files = {
        name: sha
        for name, sha in original_payload["files"].items()
        if name.startswith("stella-snapshot/")
        or name.startswith("fixtures/")
        and name != "fixtures/manifest.json"
    }
    save(
        out / "variant-manifest.json",
        {
            "snapshot": snapshot.name,
            "original_files": original_files,
            "files": {
                str(path.relative_to(out)): digest(path)
                for path in sorted(out.rglob("*"))
                if path.is_file()
            },
        },
    )
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


def frozen_flags(extension: Path, graph: bool) -> dict[str, str]:
    flags = {
        "VMECPP_GPU_SOLVER": "1",
        "VMECPP_FORWARD_REFERENCE_SO": str(extension),
        "VMECPP_GPU_SOLVER_RESIDENT_CONTROL": "1",
        "VMECPP_GPU_RESIDUAL_ROUNDS": "12",
        "VMECPP_GPU_RESIDUAL_GRAPH": "1",
        "VMECPP_GPU_DIAGNOSTICS_ROUNDS": "16",
        "VMECPP_GPU_DIAGNOSTICS_GRAPH": "1",
    }
    if graph:
        flags.update(VMECPP_GPU_SOLVER_GRAPH="1", VMECPP_GPU_SOLVER_PINNED_SCALARS="1")
    return flags


def verify_original(manifest: dict) -> None:
    for name, expected in manifest["original_files"].items():
        if digest(BASE / name) != expected:
            message = "Frozen source or public fixture changed: " + name
            raise ValueError(message)


def stage(root: Path, *, smoke: bool) -> None:
    manifest = json.loads((root / "variant-manifest.json").read_text())
    expected_name = manifest["snapshot"] + "-replica"
    if root != REMOTE_PARENT / expected_name:
        message = "Unexpected fresh replica directory"
        raise ValueError(message)
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            message = "Variant payload changed: " + name
            raise ValueError(message)
    verify_original(manifest)
    extension = (
        BASE
        / "candidate-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    )
    if digest(extension) != REFERENCE_SHA256:
        message = "Matching CPU reference changed"
        raise ValueError(message)
    (root / "logs").mkdir(exist_ok=False)
    environment = clean_environment(dict(os.environ))
    environment["PYTHONPATH"] = str(root)
    environment["LD_LIBRARY_PATH"] = str(REMOTE_PARENT / "lib")
    commands = []

    def run(command: list[str], name: str) -> None:
        commands.append({"name": name, "command": command})
        save(root / "stage-commands.json", commands)
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
            message = "Variant command failed: " + name
            raise RuntimeError(message)

    relocated = []
    patchelf = str(BASE / "tool-env/bin/patchelf")
    for path in [*sorted((root / "lib").glob("*.so")), root / "bin/solver-replay"]:
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
        relocated.append(
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
        (root / "logs" / (path.name + ".ldd.log")).write_text(
            dependencies.stdout + dependencies.stderr
        )
        if "not found" in dependencies.stdout:
            message = "Variant dependency missing: " + path.name
            raise RuntimeError(message)
    save(root / "relocation.json", relocated)
    fixtures = json.loads((BASE / "fixtures/remote-manifest.json").read_text())
    save(root / "fixtures.json", fixtures)
    save(
        root / "selection-fixture.json",
        [
            fixture
            for fixture in fixtures
            if Path(fixture["path"]).name == "public_p3_seed.json"
        ],
    )
    source_files = {
        str(BASE / name): expected
        for name, expected in manifest["original_files"].items()
    }
    source_files.update(
        {
            str(root / name): sha
            for name, sha in manifest["files"].items()
            if not name.startswith(("lib/", "bin/"))
        }
    )
    candidate_files = {
        str(root / item["path"]): item["after"]
        for item in relocated
        if item["path"].startswith("lib/")
    }
    candidate_files[str(extension)] = REFERENCE_SHA256
    original_extension = (
        REMOTE_PARENT
        / "baseline/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    )
    old_relocation = json.loads((BASE / "relocation.json").read_text())
    variants = {
        "baseline": {
            "python": str(REMOTE_PARENT / "baseline/bin/python"),
            "flags": {},
            "preload": [],
            "files": {str(original_extension): digest(original_extension)},
        },
        "candidate": {
            "python": str(BASE / "candidate-env/bin/python"),
            "flags": frozen_flags(
                extension, manifest["snapshot"] == "resident-solver8-capture"
            ),
            "preload": [str(root / "lib/solver-shim.so")],
            "files": candidate_files,
        },
        "variant4": {
            "python": str(BASE / "candidate-env/bin/python"),
            "flags": frozen_flags(extension, False),
            "preload": [str(BASE / "lib/solver-shim.so")],
            "files": {
                str(BASE / item["path"]): item["after"]
                for item in old_relocation
                if item["path"].startswith("lib/")
            },
        },
    }
    for label, configuration in variants.items():
        wrapper = root / "bin" / (label + "-python")
        config_path = root / (label + "-config.json")
        configuration.update(
            wrapper=str(wrapper),
            files={**source_files, **configuration["files"]},
            ld_library_path=str(REMOTE_PARENT / "lib"),
        )
        save(config_path, configuration)
        command = [
            "env",
            "PYTHONPATH=" + str(root),
            str(REMOTE_PARENT / "baseline/bin/python"),
            "-B",
            "-m",
            "tools.optimization.secondary_variant",
            "exec",
            "--config",
            str(config_path),
            "--",
        ]
        wrapper.write_text("#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n')
        wrapper.chmod(0o755)
    python = str(REMOTE_PARENT / "baseline/bin/python")
    common = [
        "--stella",
        str(BASE / "stella-snapshot"),
        "--cpu",
        str(min(os.sched_getaffinity(0))),
        "--reference-directory",
        str(root / "references"),
    ]
    paired = str(root / "tools/optimization/paired.py")
    full = [
        python,
        "-B",
        paired,
        "--baseline-python",
        str(root / "bin/baseline-python"),
        "--candidate-python",
        str(root / "bin/candidate-python"),
        "--manifest",
        str(root / "fixtures.json"),
        "--out",
        str(root / "paired"),
        "--pairs",
        "6",
        *common,
    ]
    selection = [
        python,
        "-B",
        paired,
        "--baseline-python",
        str(root / "bin/variant4-python"),
        "--candidate-python",
        str(root / "bin/candidate-python"),
        "--manifest",
        str(root / "selection-fixture.json"),
        "--out",
        str(root / "selection"),
        "--pairs",
        "3",
        "--tracks",
        "P3",
        *common,
    ]
    smoke_command = [
        python,
        "-B",
        "-m",
        "tools.optimization.validate_resident_solver",
        "--python",
        str(BASE / "candidate-env/bin/python"),
        "--stella",
        str(BASE / "stella-snapshot"),
        "--manifest",
        str(root / "fixtures.json"),
        "--reference",
        str(BASE / "reference/report.json"),
        "--reference-trace",
        str(BASE / "reference/cpu.trace.jsonl"),
        "--out",
        str(root / "smoke"),
        "--preload",
        str(root / "lib/solver-shim.so"),
    ]
    for key, value in variants["candidate"]["flags"].items():
        smoke_command.extend(("--env", key + "=" + value))
    save(
        root / "campaign-plan.json",
        {
            "full_campaign": full,
            "selection": selection,
            "smoke": smoke_command,
            "full_evaluations": 72,
            "warmup_pairs": 1,
            "measured_pairs": 5,
            "selection_evaluations": 6,
            "full_campaign_launched": False,
        },
    )
    if smoke:
        status = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if any(int(value) for value in status.split()) or any(
            name != "/usr/libexec/gnome-remote-desktop-daemon"
            for name in processes.splitlines()
            if name
        ):
            message = "Secondary GPU is busy; no variant smoke started"
            raise RuntimeError(message)
        run(smoke_command, "strict-smoke")


def execute(config: Path, arguments: list[str]) -> None:
    configuration = json.loads(config.read_text())
    for name, expected in configuration["files"].items():
        if digest(Path(name)) != expected:
            message = "Frozen campaign dependency changed: " + name
            raise ValueError(message)
    environment = clean_environment(dict(os.environ))
    environment.update(configuration["flags"])
    environment["LD_LIBRARY_PATH"] = configuration["ld_library_path"]
    if configuration["preload"]:
        environment["LD_PRELOAD"] = ":".join(configuration["preload"])
    wrapper = Path(configuration["wrapper"])
    environment["CGA_OPTIMIZATION_WRAPPER_SHA256"] = hashlib.sha256(
        wrapper.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    environment["CGA_OPTIMIZATION_CONFIG_SHA256"] = digest(config)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    os.execve(
        configuration["python"],
        [configuration["python"], "-B", *arguments],
        environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("package", "stage", "exec"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path)
    args, remaining = parser.parse_known_args()
    if args.mode == "package":
        if args.workspace is None or args.snapshot is None or args.out is None:
            parser.error("package requires workspace, snapshot and out")
        package(args.workspace.resolve(), args.snapshot.resolve(), args.out.resolve())
    elif args.mode == "stage":
        if args.root is None:
            parser.error("stage requires root")
        stage(args.root.resolve(), smoke=args.smoke)
    else:
        if args.config is None:
            parser.error("exec requires config")
        execute(args.config, remaining)


if __name__ == "__main__":
    main()
