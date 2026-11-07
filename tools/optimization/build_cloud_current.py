"""Assemble an isolated current A100 pilot payload locally; no GPU/cloud actions."""

from __future__ import annotations

# The output is an explicit local build recipe/result.
# ruff: noqa: T201
import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from tools.optimization.cloud_pilot_current import REFERENCE_SHA, REQUIRED_FLAGS

ROOT = Path(__file__).resolve().parents[2]
OPT = ROOT / "build/optimization"
OLD = OPT / "validation-agent/cloud-pilot8-bundle"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with destination.open("rb") as file:
        is_elf = file.read(4) == b"\x7fELF"
    if is_elf:
        destination.chmod(0o755)


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {16, 17, 18, 19})
    os.nice(19)
    for name in ("oracle-packages.tar.gz", "nsys-cli.tar.gz", "ncu-cli.tar.gz"):
        copy_file(OPT / name, out / "archives" / name)
    for name in ("fixtures", "references", "stella-snapshot"):
        shutil.copytree(OLD / name, out / name)
    inputs = json.loads(
        (OPT / "native-seed-forward-payload2/original-library-inputs.json").read_text()
    )
    for name, record in inputs.items():
        assert digest(record["path"]) == record["sha256"]
        copy_file(record["path"], out / "lib" / name)
    copy_file(OPT / "profile-tiles-full1/resident.so", out / "lib/baseline-resident.so")
    for source, name in (
        (OPT / "roofline-matched1/roofline-matched", "roofline-matched"),
        (OPT / "forward-groups1/groups-benchmark", "groups-benchmark"),
        (
            OPT / "resident-solver-seed-forward1/control/solver-control-replay",
            "solver-control-replay",
        ),
        (OLD / "bin/patchelf", "patchelf"),
    ):
        copy_file(source, out / "bin" / name)
    guard_source = ROOT / "tools/optimization/pilot_cpu_guard.cc"
    copy_file(guard_source, out / "source/pilot_cpu_guard.cc")
    nvtx_headers = Path(
        "/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nvtx/include"
    )
    shutil.copytree(nvtx_headers / "nvtx3", out / "source/nvtx3")
    guard_command = [
        "g++",
        "-std=c++20",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
        "-I" + str(out / "source"),
        str(out / "source/pilot_cpu_guard.cc"),
        "-o",
        str(out / "bin/pilot-cpu-guard.so"),
    ]
    subprocess.run(guard_command, check=True)
    (out / "source/pilot-cpu-guard-build-command.json").write_text(
        json.dumps(guard_command, indent=2) + "\n"
    )
    for source, name in (
        (OPT / "roofline-matched1/source/roofline_matched.cu", "roofline_matched.cu"),
        (OPT / "forward-groups1/source/groups_benchmark.cc", "groups_benchmark.cc"),
    ):
        copy_file(source, out / "source" / name)
    shutil.copytree(OPT / "resident-solver-seed-forward1/source", out / "source/solver")
    shutil.copytree(OPT / "forward-groups1/source", out / "source/forward")
    copy_file(
        OPT / "forward-prototype/real-forward.bin", out / "captures/real-forward.bin"
    )
    tools = out / "tools/optimization"
    tools.mkdir(parents=True)
    (out / "tools/__init__.py").write_text("")
    (tools / "__init__.py").write_text("")
    for name in (
        "benchmark.py",
        "campaign_gates.py",
        "summarize.py",
        "validate_resident_solver.py",
        "paired.py",
        "report_pairs.py",
        "cloud_pilot_controller.py",
        "cloud_pilot_current_controller.py",
        "analyze_full_vmec_profile.py",
        "analyze_batch_round.py",
        "analyze_batch_timeline.py",
    ):
        copy_file(ROOT / "tools/optimization" / name, tools / name)
    copy_file(
        ROOT / "tools/optimization/cloud_pilot.py", tools / "cloud_pilot_legacy.py"
    )
    copy_file(
        ROOT / "tools/optimization/cloud_pilot_current.py", tools / "cloud_pilot.py"
    )
    python_site = OPT / "gpu-env/lib/python3.12/site-packages"
    shutil.copytree(
        python_site / "vmecpp",
        out / "python/vmecpp",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    assert (
        digest(out / "python/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so")
        == REFERENCE_SHA
    )
    metadata = "vmecpp-0.4.11.dist-info"
    with tarfile.open(OPT / "oracle-packages.tar.gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if Path(member.name).parts and Path(member.name).parts[0] == metadata
        ]
        assert members
        archive.extractall(out / "python", members=members, filter="data")
    record = out / "python" / metadata / "RECORD"
    copy_file(record, out / "provenance/original-vmecpp-RECORD.csv")
    rows = []
    for path in sorted((out / "python").rglob("*")):
        if not path.is_file() or path == record:
            continue
        hashed = (
            base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
            .decode()
            .rstrip("=")
        )
        rows.append(
            [
                str(path.relative_to(out / "python")),
                "sha256=" + hashed,
                str(path.stat().st_size),
            ]
        )
    rows.append([metadata + "/RECORD", "", ""])
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    record.write_text(stream.getvalue())
    # Only private non-glibc dependency copies; never replace system libc/driver.
    readiness = json.loads(
        (OPT / "a100-current-readiness1/transitive-files.json").read_text()
    )
    excluded = (
        "libc.so",
        "libm.so",
        "ld-linux",
        "libpthread.so",
        "libdl.so",
        "librt.so",
        "libutil.so",
        "libresolv.so",
    )
    runtime_sources = {}
    for source, facts in readiness["files"].items():
        path = Path(source)
        if "/opt/nvidia/nsight-systems/" in source:
            copy_file(path, out / "profiling/nsys-host" / path.name)
        elif path.name in inputs or path.name.startswith("_vmecpp"):
            # Their relocated canonical copies live in work/lib/current package.
            continue
        elif not path.name.startswith(excluded):
            destination = out / "runtime" / path.name
            if destination.exists():
                assert digest(destination) == facts["sha256"]
            else:
                copy_file(path, destination)
            runtime_sources[str(destination.relative_to(out))] = facts
    copy_file(
        "/usr/local/cuda-12.9/lib64/libcudart.so.12", out / "runtime/libcudart.so.12"
    )
    # Preserve executable mode for loader-invoked binaries across Windows/WSL copies.
    for path in (out / "profiling/nsys-host").iterdir():
        if path.name == "QdstrmImporter":
            path.chmod(0o755)
    proof = out / "provenance"
    proof.mkdir(exist_ok=True)
    for source, name in (
        (
            OPT / "native-seed-forward-crosshost-results2/independent-audit.json",
            "native12-correctness.json",
        ),
        (
            OPT / "seed-forward-qualification1/final-summary.json",
            "local-correctness.json",
        ),
        (
            OPT / "resident-solver-seed-forward1/commands.json",
            "candidate-build-commands.json",
        ),
        (OPT / "cloud-image-compatibility1/metadata.json", "image-metadata.json"),
        (OPT / "a100-current-readiness1/base-image-abi.json", "base-image-abi.json"),
    ):
        copy_file(source, proof / name)
    checkpoints = sorted(
        path
        for path in (OPT / "native-seed-forward-qualification-checkpoints").glob(
            "c072-complete-*"
        )
        if path.is_dir()
    )
    assert checkpoints
    for name in (
        "full-campaign-status.json",
        "full-guard-status.json",
        "candidate-config.json",
        "verification.json",
    ):
        copy_file(checkpoints[-1] / name, proof / ("native72-" + name))
    (proof / "runtime-sources.json").write_text(
        json.dumps(runtime_sources, indent=2) + "\n"
    )
    config = {
        "candidate": "FULL72-qualified SparseSeed+N3/Thomas8 original CPU controller",
        "CPU_baseline_label": "matched current CPU build144e, no GPUpreload",
        "original_stock3358_scope": "immutable global reference only; no live stockwheel latency measured by this pilot",
        "flags": REQUIRED_FLAGS,
        "reference_extension_sha256": REFERENCE_SHA,
        "controller_pool_GIL_queue": "OFF",
        "max_hourly_usd": 1.0,
        "maximum_lease_seconds": 3600,
        "image_manifest": "sha256:d0a2357cee2a6f1bfb5274d59e3c701cae7b73890ad0d0ab6c018ba4b2274a63",
        "permission_denied_policy": "stop after exact anchor/repeat and actual permission probe; retrieve/destroy",
        "nsys_injection_sha256": digest(
            "/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/libToolsInjection64.so"
        ),
        "expected_core_sha256": {
            name: digest("/lib/x86_64-linux-gnu/" + name)
            for name in ("libc.so.6", "libm.so.6", "libstdc++.so.6")
        },
        "required_assets": [
            "archives/oracle-packages.tar.gz",
            "archives/nsys-cli.tar.gz",
            "archives/ncu-cli.tar.gz",
            "bin/roofline-matched",
            "bin/groups-benchmark",
            "bin/patchelf",
            "bin/pilot-cpu-guard.so",
            "lib/solver-shim.so",
            "lib/solver-backend.so",
            "lib/resident.so",
            "lib/rhs.so",
            "lib/iteration.so",
            "lib/baseline-resident.so",
            "captures/real-forward.bin",
            "profiling/nsys-host/QdstrmImporter",
            "runtime/libcudart.so.12",
            "fixtures/manifest.json",
            "python/vmecpp-0.4.11.dist-info/RECORD",
        ],
    }
    (out / "pilot-config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "matching_extension_sha256": REFERENCE_SHA,
        "original_extension_sha256": REFERENCE_SHA,
        "scope": "Current canonical candidate; excludes optional fused/unqualified device-dispatch work",
        "vmecpp_metadata_RECORD": "derived truthfully from current overlay; original preserved",
        "files": {
            str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    }
    (out / "payload-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "root": str(out),
        "files": len(manifest["files"]),
        "bytes": sum(path.stat().st_size for path in out.rglob("*") if path.is_file()),
        "GPU_runs": 0,
        "lease_actions": 0,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        print(json.dumps({"plan_only": True, "GPU_runs": 0, "lease_actions": 0}))
        return
    print(json.dumps(build(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
