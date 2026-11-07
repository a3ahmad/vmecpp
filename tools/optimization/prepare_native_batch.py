"""Package and CPU-stage an isolated native solver18 batch campaign."""

# Preparation emits reviewable manifest/recipe paths.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tarfile
from pathlib import Path

from tools.optimization.native_batch_campaign import (
    REMOTE_ROOT,
    digest,
    physical_affinity,
    save,
    service_launch,
)

REFERENCE_SHA256 = "144e4f204cbb0569edd25710cf8ef4c0e618b0e7a60e33cf3a556066e8a06da8"
CUDART_SHA256 = "381c0faae2678098eb2aef376ff04ef7129d86e28d0cc0fd2c83d4b6d75c0098"


def text_digest(path, section_name=".text"):
    value = Path(path).read_bytes()
    if value[:6] != b"\x7fELF\x02\x01":
        message = "Expected a little-endian ELF64 file"
        raise ValueError(message)
    offset = struct.unpack_from("<Q", value, 40)[0]
    stride, count, names_index = struct.unpack_from("<HHH", value, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", value, offset + index * stride)
        for index in range(count)
    ]
    names = sections[names_index]
    strings = value[names[4] : names[4] + names[5]]
    for section in sections:
        name = strings[section[0] :].split(b"\0", 1)[0]
        if name == section_name.encode():
            return hashlib.sha256(
                value[section[4] : section[4] + section[5]]
            ).hexdigest()
    message = "ELF has no requested payload section"
    raise ValueError(message)


def package(workspace, out):
    out.mkdir(parents=True, exist_ok=False)
    optimization = workspace / "build/optimization"

    def copy(source, destination):
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for name, source in {
        "solver-shim.so": "resident-solver18-lattice-coordinated/solver-shim.so",
        "solver-backend.so": "resident-solver18-lattice-coordinated/solver-backend.so",
        "resident.so": "resident-device-inverse/resident.so",
        "iteration.so": "resident-iteration-factor1/iteration.so",
        "gil-release-shim.so": "validation-agent/gil-release-prototype/build4/gil-release-shim.so",
        "gil-release-test.so": "validation-agent/gil-release-prototype/build4/gil-release-test.so",
    }.items():
        copy(optimization / source, "lib/" + name)
    for directory, target in (
        (
            optimization / "resident-solver18-lattice-coordinated/source",
            "source/solver18",
        ),
        (optimization / "resident-device-inverse/source", "source/resident"),
        (optimization / "resident-iteration-factor1/source", "source/iteration"),
    ):
        for path in directory.iterdir():
            if path.is_file():
                copy(path, target + "/" + path.name)
    for name in ("commands.json", "source-hashes.json"):
        copy(
            optimization / "resident-solver18-lattice-coordinated" / name,
            "provenance/solver18-" + name,
        )
    for name in (
        "gil_release_shim.cc",
        "build-commands.json",
        "backend-guard-tests.json",
    ):
        copy(
            optimization / "validation-agent/gil-release-prototype/build4" / name,
            "provenance/gil4-" + name,
        )
    for name in (
        "batch_throughput.py",
        "campaign_gates.py",
        "summarize.py",
        "native_batch_campaign.py",
        "prepare_native_batch.py",
    ):
        copy(workspace / "tools/optimization" / name, "tools/optimization/" + name)
    copy(
        optimization / "secondary-solver4-payload/stella-snapshot/configs/oracle.yaml",
        "stella/configs/oracle.yaml",
    )
    for path in (optimization / "fixtures-frozen").glob("*.json"):
        copy(path, "fixtures/" + path.name)
    for directory in (
        optimization / "validation-agent/cloud-pilot8-bundle/references"
    ).iterdir():
        for path in directory.glob("*.json"):
            copy(path, "references/" + directory.name + "/" + path.name)
    python = optimization / "gpu-env/lib/python3.12/site-packages/vmecpp"
    for path in python.rglob("*"):
        if (
            path.is_file()
            and path.suffix not in (".so", ".pyc")
            and "__pycache__" not in path.parts
        ):
            copy(path, "python/vmecpp/" + str(path.relative_to(python)))
    historical = json.loads(
        (
            optimization / "validation-agent/native72-final-independent-summary.json"
        ).read_text()
    )
    measured_sum = sum(row["candidate_seconds"] for row in historical["rows"])
    plan = {
        "scope": "Prepared only; no native solver18 batch timing measured",
        "remote_root": str(REMOTE_ROOT),
        "workload_cases": 6,
        "workers": [1, 4],
        "persistent_pools": True,
        "warmup_pairs": 1,
        "measured_pairs": 5,
        "full_evaluations": 72,
        "estimated_minutes": [20, 30],
        "estimate_basis": {
            "measured_native_solver8_six_case_median_seconds_sum": measured_sum,
            "assumptions_not_measurements": "Solver18 no slower than historical8, four-worker workload throughput1.5..2x serial; initial excluded warmup will update ETA. No native18 batch result exists yet.",
        },
        "independent_deadline_seconds": 2100,
        "optional_two_worker_arm": "Excluded to keep the fixed72-evaluation primary campaign within the time estimate",
        "no_profiler": True,
        "main_single_eval_table_eligible": False,
    }
    save(out / "campaign-plan.json", plan)
    manifest = {
        "files": {
            str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
        "reference_sha256": REFERENCE_SHA256,
        "cudart_sha256": CUDART_SHA256,
        "no_credentials_or_submission_artifacts": True,
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


def stage(root):
    if root.resolve() != REMOTE_ROOT:
        message = "Unexpected native batch stage root"
        raise ValueError(message)
    if (root / "stage-ready.json").exists() or (root / "stage-started.json").exists():
        message = "Stage already attempted; never patch/restart implicitly"
        raise RuntimeError(message)
    manifest = json.loads((root / "payload-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError("Payload path/hash mismatch: " + name)
    save(
        root / "stage-started.json",
        {"scope": "CPU-only deployment; GPU execution forbidden in this action"},
    )
    parent = root.parent
    topology = []
    for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        if not (cpu / "topology/core_id").is_file():
            continue
        nodes = sorted(path.name for path in cpu.glob("node[0-9]*"))
        topology.append(
            {
                "cpu": int(cpu.name[3:]),
                "core": int((cpu / "topology/core_id").read_text()),
                "package": int((cpu / "topology/physical_package_id").read_text()),
                "node": nodes[0] if nodes else None,
                "siblings": (cpu / "topology/thread_siblings_list").read_text().strip(),
            }
        )
    affinity = physical_affinity(topology)
    extension = parent / "gpu-extension.so"
    cudart = parent / "lib/libcudart.so.12"
    if digest(extension) != REFERENCE_SHA256 or digest(cudart) != CUDART_SHA256:
        message = "Pinned existing reference/runtime changed"
        raise RuntimeError(message)
    target_extension = (
        root / "python/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    )
    target_extension.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(extension, target_extension)
    shutil.copy2(cudart, root / "lib/libcudart.so.12")
    shutil.copytree(
        parent / "stella/src/cga",
        root / "stella/src/cga",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # The approved shared source tree contains src/cga only. The payload keeps
    # the original frozen oracle configuration from the initial replica.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in ("HOME", "PATH", "LANG", "LC_ALL")
    }
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LD_LIBRARY_PATH=str(root / "lib"),
    )
    patch = parent / "resident-solver4/tool-env/bin/patchelf"
    relocation = []
    (root / "logs").mkdir()
    for path in sorted((root / "lib").glob("*.so")):
        before, code_before = digest(path), text_digest(path)
        payload_sections = {}
        for name in (".rodata", ".nv_fatbin", ".nvFatBinSegment"):
            with contextlib.suppress(ValueError):
                payload_sections[name] = text_digest(path, name)
        dynamic = subprocess.run(
            ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
        ).stdout
        changes = []
        for needed in re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic):
            if needed.startswith("/mnt/f/"):
                replacement = (
                    str(target_extension) if "_vmecpp" in needed else Path(needed).name
                )
                subprocess.run(
                    [str(patch), "--replace-needed", needed, replacement, str(path)],
                    check=True,
                )
                changes.append({"old": needed, "new": replacement})
            if path.name.startswith("gil-release") and needed == "libpython3.12.so.1.0":
                subprocess.run(
                    [str(patch), "--remove-needed", needed, str(path)], check=True
                )
                changes.append(
                    {
                        "removed": needed,
                        "reason": "Native interpreter exports its own static Python runtime; never load a second runtime",
                    }
                )
        subprocess.run(
            [
                str(patch),
                "--set-rpath",
                "$ORIGIN:" + str(target_extension.parent),
                str(path),
            ],
            check=True,
        )
        code_after = text_digest(path)
        if code_before != code_after:
            message = "Relocation modified machine-code .text"
            raise RuntimeError(message)
        if any(
            text_digest(path, name) != sha for name, sha in payload_sections.items()
        ):
            message = "Relocation modified constants or embedded GPU binary payload"
            raise RuntimeError(message)
        linked = subprocess.run(
            ["ldd", str(path)],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        ).stdout
        (root / "logs" / (path.name + ".ldd.log")).write_text(linked)
        if "not found" in linked:
            message = "Copied native dependency unresolved: " + path.name
            raise RuntimeError(message)
        relocation.append(
            {
                "path": str(path.relative_to(root)),
                "before": before,
                "after": digest(path),
                "text_sha256_unchanged": code_after,
                "constant_and_gpu_payload_sections_unchanged": payload_sections,
                "changes": changes,
            }
        )
    save(root / "relocation.json", relocation)
    fixtures = json.loads((root / "fixtures/manifest.json").read_text())
    for item in fixtures:
        item["path"] = str(root / "fixtures" / Path(item["path"]).name)
    save(root / "fixtures/remote-manifest.json", fixtures)
    service_environment = dict(environment)
    service_environment.update(
        PYTHONPATH=str(root / "python") + ":" + str(root),
        LD_PRELOAD=str(root / "lib/gil-release-shim.so")
        + ":"
        + str(root / "lib/solver-shim.so"),
        VMECPP_RELEASE_GIL="1",
        VMECPP_FORWARD_REFERENCE_SO=str(target_extension),
        VMECPP_GPU_SOLVER="1",
        VMECPP_GPU_SOLVER_RESIDENT_CONTROL="1",
        VMECPP_GPU_RESIDUAL_ROUNDS="32",
        VMECPP_GPU_RESIDUAL_GRAPH="1",
        VMECPP_GPU_DIAGNOSTICS_ROUNDS="16",
        VMECPP_GPU_DIAGNOSTICS_GRAPH="1",
        VMECPP_GPU_SOLVER_GRAPH="1",
        VMECPP_GPU_SOLVER_PINNED_SCALARS="1",
    )
    python = str(parent / "baseline/bin/python")
    probe = "import ctypes,json;from pathlib import Path;from types import SimpleNamespace;from tools.optimization.batch_throughput import preflight_runtime;v=preflight_runtime(SimpleNamespace(backend='gpu'));m=Path('/proc/self/maps').read_text();assert 'libpython3.12' not in m;assert v['activity']()==(0,0,0);print(json.dumps({'ready':True,'counts':v['activity'](),'backend':str(v['backend_library']),'no_second_python_runtime':True}))"
    command, probe_environment = service_launch(
        service_environment,
        sorted(os.sched_getaffinity(0)),
        python,
        ["-B", "-c", probe],
    )
    result = subprocess.run(
        command,
        cwd=root,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    (root / "logs/cpu-production-guard.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        message = "Native production GIL/backend CPU guard failed"
        raise RuntimeError(message)
    test_environment = dict(service_environment)
    test_environment["LD_PRELOAD"] = (
        str(root / "lib/gil-release-test.so") + ":" + str(root / "lib/solver-shim.so")
    )
    test = """
import ctypes,json,threading
from pathlib import Path
g=ctypes.PyDLL(None)
g.VmecppGilGuardExercise.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int]
g.VmecppGilGuardExercise.restype=ctypes.c_int
active=ctypes.CDLL(None).VmecppGilGuardActive
active.restype=ctypes.c_int
done=threading.Event()
progress=[]
def spin():
    while not done.is_set():
        if active(): progress.append(1)
thread=threading.Thread(target=spin)
thread.start()
try:
    assert g.VmecppGilGuardExercise(0,0,100)==7
finally:
    done.set()
    thread.join()
assert progress
assert g.VmecppGilGuardExercise(0,1,0)==15
assert g.VmecppGilGuardExercise(1,0,0)==16
assert 'libpython3.12' not in Path('/proc/self/maps').read_text()
print(json.dumps({'normal_release_reacquire':True,'exception_reacquire':True,'verbose_rejection':True,'thread_progress_while_released':len(progress),'no_second_python_runtime':True,'GPU_executed':False}))
"""
    command, probe_environment = service_launch(
        test_environment, sorted(os.sched_getaffinity(0)), python, ["-B", "-c", test]
    )
    result = subprocess.run(
        command,
        cwd=root,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    (root / "logs/cpu-gil-exercise.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        message = "Native GIL CPU mock failed"
        raise RuntimeError(message)
    files = {
        str(path): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "stage-started.json"
        and "logs" not in path.parts
    }
    files[python] = digest(python)
    save(
        root / "stage-ready.json",
        {
            "scope": "CPU-only staged; no GPU evaluation executed",
            "GPU_executed": False,
            "files": files,
            "python": python,
            "service_environment": service_environment,
            "relocation_text_unchanged": True,
            "native_python_runtime_guard_pass": True,
            "affinity": affinity,
        },
    )
    print(
        json.dumps(
            {
                "stage_ready": True,
                "GPU_executed": False,
                "root": str(root),
                "files": len(files),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "stage"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.action == "package":
        if args.workspace is None or args.out is None:
            parser.error("package requires --workspace and --out")
        package(args.workspace.resolve(), args.out.resolve())
    else:
        if args.root is None:
            parser.error("stage requires --root")
        stage(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
