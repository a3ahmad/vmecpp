"""Profile-only diagnostic: one anchor and one profiled GPU equilibrium.

All execution requires an existing external <=1h lease watchdog. Default is plan. The
payload installs this file as cloud_pilot.py with a frozen legacy helper copy.
"""

from __future__ import annotations

# CLI reporting and delayed oracle imports are intentional.
# ruff: noqa: T201, PLC0415, BLE001
import argparse
import contextlib
import ctypes
import hashlib
import importlib
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import zipfile
from importlib import metadata
from pathlib import Path

sys.dont_write_bytecode = True

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_helper = (
    "cloud_pilot_legacy"
    if Path(__file__).with_name("cloud_pilot_legacy.py").exists()
    else "cloud_pilot"
)
legacy = importlib.import_module("tools.optimization." + _helper)
gates = importlib.import_module("tools.optimization.campaign_gates")
REFERENCE_SHA = "144e4f204cbb0569edd25710cf8ef4c0e618b0e7a60e33cf3a556066e8a06da8"
BASELINE_LABEL = "matched current CPU build144e, no GPUpreload"
REQUIRED_FLAGS = {
    "VMECPP_GPU_SOLVER": "1",
    "VMECPP_GPU_SOLVER_RESIDENT_CONTROL": "1",
    "VMECPP_GPU_RESIDUAL_ROUNDS": "32",
    "VMECPP_GPU_RESIDUAL_GRAPH": "1",
    "VMECPP_GPU_DIAGNOSTICS_ROUNDS": "16",
    "VMECPP_GPU_DIAGNOSTICS_GRAPH": "1",
    "VMECPP_GPU_SOLVER_GRAPH": "1",
    "VMECPP_GPU_SOLVER_PINNED_SCALARS": "1",
    "VMECPP_FORWARD_REFERENCE_SO": "{extension}",
}
RESERVE = 600

compatibility = importlib.import_module("tools.optimization.cloud_cpu_compatibility")
PRIOR_CANONICAL_SHA256 = (
    "46a087e109be6e1d975ed2ee1c7198710fb288332f472c65491b483704650b43"
)


def compatibility_paths(root):
    output = root / "results/cpu-compatibility"
    return output / "selection.json", output / "probe/summary.json"


def selected_cpu_environment(root):
    selected, summary = compatibility_paths(root)
    return compatibility.read_selection(selected, summary)


def verify_worker_dispatch(root):
    selected = selected_cpu_environment(root)
    for key in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES"):
        if os.environ.get(key) != selected.get(key):
            message = "Actual worker CPU dispatch differs from verified replay"
            raise RuntimeError(message)


def run_compatibility_probe(root, started, expires):
    probe = root / "profiling/cpu-compatibility"
    output = root / "results/cpu-compatibility"
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(root / "work/baseline/bin/python"),
        str(probe / "run_probe.py"),
        "--execute",
        "--wout",
        str(probe / "reference/raw_wout_0.nc"),
        "--reference",
        str(probe / "reference/replay.json"),
        "--source-root",
        str(probe / "source/constellaration"),
        "--out",
        str(output / "probe"),
        "--repeats",
        "2",
        "--child-seconds",
        "30",
        "--total-seconds",
        "180",
        "--expires-at",
        str(expires),
        "--reserve-seconds",
        str(RESERVE),
    ]
    job = run_owned(
        root,
        command,
        "cpu-compatibility-probe",
        time_budget(started, expires, 200),
        environment(root, "baseline", allow_unselected=True),
    )
    if not job["passed"]:
        message = "CPU compatibility probe failed or owned children not reaped"
        raise RuntimeError(message)
    selected, summary = compatibility_paths(root)
    save(selected, compatibility.decision_from_file(summary))
    # Read back and bind the exact receipt bytes before any expensive solve.
    selected_cpu_environment(root)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    legacy.save(Path(path), value)


def finish_cpu_compatibility(root, started, expires):
    status_path = root / "bootstrap-status.json"
    status = json.loads(status_path.read_text())
    status.update(phase="CPU_compatibility_pending", CPU_compatibility="pending")
    save(status_path, status)
    try:
        run_compatibility_probe(root, started, expires)
        dispatch = selected_cpu_environment(root)
    except BaseException as error:
        status.update(
            phase="failed_CPU_compatibility", failure_type=type(error).__name__
        )
        save(status_path, status)
        raise
    status.update(
        phase="complete",
        CPU_compatibility="exact_replay_selection_only",
        CPU_compatibility_dispatch=dispatch,
    )
    save(status_path, status)


def config(root):
    value = json.loads((root / "pilot-config.json").read_text())
    if (
        value.get("flags") != REQUIRED_FLAGS
        or value.get("reference_extension_sha256") != REFERENCE_SHA
    ):
        message = "Current nine flags/residual32/reference144e required"
        raise ValueError(message)
    price = value.get("max_hourly_usd")
    if (
        not isinstance(price, int | float)
        or isinstance(price, bool)
        or not math.isfinite(price)
        or not 0 < price <= 1
        or value.get("maximum_lease_seconds") != 3600
    ):
        message = "Pilot policy requires <=1 USD/hour and <=1h external watchdog"
        raise ValueError(message)
    if value.get("controller_pool_GIL_queue") != "OFF":
        message = "Canonical route requires original CPU control and no pool/GIL/queue"
        raise ValueError(message)
    if value.get("saved_output_cpu_compatibility") != "required":
        message = "Saved-output CPU compatibility is mandatory in this pilot"
        raise ValueError(message)
    require_hardware_policy(value)
    if value.get("diagnostic_mode") != "anchor-plus-profile":
        message = "Explicit anchor-plus-profile diagnostic mode required"
        raise ValueError(message)
    prior = root / "provenance/prior-canonical-evidence.json"
    if (
        value.get("prior_canonical_evidence_sha256") != PRIOR_CANONICAL_SHA256
        or digest(prior) != PRIOR_CANONICAL_SHA256
    ):
        message = "Prior canonical evidence receipt missing or mismatched"
        raise ValueError(message)
    audit = json.loads(prior.read_text())
    if not (
        audit.get("actual_equilibrium_evaluations") == 18
        and audit.get("actual_CPU_equilibria") == 6
        and audit.get("actual_GPU_equilibria") == 12
        and audit.get("actual_cases") == 6
        and audit.get("actual_boundaries") == 3
        and audit.get("actual_tracks") == ["P1", "P3"]
        and audit.get("all18_original_global_gates_independently_recomputed_exact")
        is True
        and audit.get("max_relative_metric_drift") == 0
        and audit.get("final_retrieval_verified") is True
        and audit.get("teardown_verified") is True
        and len(audit.get("actual12_raw_comparisons", [])) == 12
        and all(item.get("exact") is True for item in audit["actual12_raw_comparisons"])
    ):
        message = "Completed exact eighteen-evaluation canonical evidence required"
        raise ValueError(message)
    if value.get("raw_compare_dependency_policy") != "pinned-baseline-subprocess":
        message = "Pinned baseline subprocess raw dependency policy required"
        raise ValueError(message)
    return value


def environment(root, variant, *, allow_unselected=False):
    value = config(root)
    env = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    site = root / "work" / variant / "lib/python3.12/site-packages"
    env.update(
        PYTHONPATH=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LD_LIBRARY_PATH=str(root / "runtime") + ":" + str(root / "work/lib"),
        CGA_OPTIMIZATION_WRAPPER_SHA256=digest(Path(__file__)),
        CGA_OPTIMIZATION_CONFIGURATION_SHA256=digest(root / "pilot-config.json"),
    )
    selected, _ = compatibility_paths(root)
    if selected.is_file() or not allow_unselected:
        env.update(selected_cpu_environment(root))
    if variant == "candidate":
        extension = site / "vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
        env.update(
            {
                key: flag.replace("{extension}", str(extension))
                for key, flag in value["flags"].items()
            }
        )
        env["LD_PRELOAD"] = str(root / "work/lib/solver-shim.so")
    return env


def preflight(root):
    manifest = legacy.verify_payload(root)
    value = config(root)
    immutable = (
        "lib",
        "runtime",
        "python",
        "source",
        "fixtures",
        "references",
        "stella-snapshot",
        "tools",
        "profiling",
        "captures",
        "bin",
        "archives",
        "provenance",
    )
    actual = {
        str(path.relative_to(root))
        for directory in immutable
        for path in (root / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual - set(manifest["files"]):
        message = "Uncovered immutable payload files"
        raise ValueError(message)
    if any(
        path.name in ("libc.so.6", "libm.so.6") or path.name.startswith("ld-linux")
        for path in (root / "runtime").iterdir()
    ):
        message = "Private runtime must not replace system glibc/loader"
        raise ValueError(message)
    if manifest.get("matching_extension_sha256") != REFERENCE_SHA:
        message = "Payload extension is not current144e"
        raise ValueError(message)
    for name in value["required_assets"]:
        if name not in manifest["files"]:
            message = "Required current asset absent: " + name
            raise ValueError(message)
    if (
        digest(root / "python/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so")
        != REFERENCE_SHA
    ):
        message = "Current whole-package overlay extension mismatch"
        raise ValueError(message)
    fixtures = json.loads((root / "fixtures/manifest.json").read_text())
    if len(fixtures) != 3:
        message = "Three unchanged frozen fixtures required"
        raise ValueError(message)
    for fixture in fixtures:
        for track in ("P1", "P3"):
            ref = root / "references" / (Path(fixture["path"]).stem + "-" + track)
            report = json.loads((ref / "report.json").read_text())
            if not gates.complete_capture(report, gates.read_capture(ref)):
                message = "Incomplete original reference capture"
                raise ValueError(message)
    return {
        "exact": True,
        "files": len(manifest["files"]),
        "config": value,
        "scope": "CPU payload integrity only; no GPU/lease execution",
    }


def time_budget(started, expires, maximum):
    if not 0 < expires - started <= 3600:
        message = "External watchdog must bound lease to <=1h"
        raise ValueError(message)
    remaining = expires - time.time() - RESERVE
    if remaining <= 0:
        message = "Retrieval/destroy reserve reached; no new child"
        raise TimeoutError(message)
    return min(float(maximum), remaining)


def birth(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def descendants():
    result, pending = {}, [os.getpid()]
    while pending:
        current = pending.pop()
        for task in Path(f"/proc/{current}/task").glob("*/children"):
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                for item in task.read_text().split():
                    pid = int(item)
                    if pid not in result:
                        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                            result[pid] = birth(pid)
                            pending.append(pid)
    return result


def run_owned(root, command, name, budget, env):
    """Track CLI and adopted targets to terminal; persist before finally cleanup."""
    job = {
        "command": command,
        "name": name,
        "phase": "starting",
        "started": time.time(),
        "cleanup_actions": [],
    }
    path = root / "results/commands" / (name + ".json")
    process = None

    def reap():
        assert process is not None
        process.poll()
        if process.returncode is not None:
            job["direct_exit_status"] = process.returncode
        for pid in descendants():
            if pid != process.pid:
                with contextlib.suppress(ChildProcessError, ProcessLookupError):
                    child, code = os.waitpid(pid, os.WNOHANG)
                    if child:
                        job.setdefault("adopted_reaped", []).append(
                            {"pid": child, "wait_status": code}
                        )

    try:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        with (root / "logs" / (name + ".log")).open("x") as log:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            job.update(pid=process.pid, birth=birth(process.pid), phase="running")
            save(path, job)
            deadline = time.monotonic() + budget
            while True:
                reap()
                job["owned_descendants"] = descendants()
                job.setdefault("observed_owned", {}).update(job["owned_descendants"])
                save(path, job)
                if process.returncode is not None and not job["owned_descendants"]:
                    break
                if time.monotonic() >= deadline:
                    message = "Owned CLI/target deadline"
                    raise TimeoutError(message)
                time.sleep(0.1)
    except BaseException as error:
        job.update(error=repr(error), phase="failed_before_cleanup")
        save(path, job)
    finally:
        job["before_cleanup"] = descendants()
        save(path, job)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid, expected in reversed(list(descendants().items())):
                with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                    if (
                        birth(pid) == expected
                        and Path(f"/proc/{pid}").stat().st_uid == os.getuid()
                    ):
                        os.kill(pid, sig)
                        job["cleanup_actions"].append(
                            {"pid": pid, "birth": expected, "signal": int(sig)}
                        )
            save(path, job)
            limit = time.monotonic() + 4
            while descendants() and time.monotonic() < limit:
                if process is not None:
                    reap()
                time.sleep(0.05)
            if not descendants():
                break
        if process is not None:
            reap()
        job.update(
            phase="terminal", ended=time.time(), owned_children_reaped=not descendants()
        )
        job["passed"] = (
            job.get("direct_exit_status") == 0
            and not job.get("error")
            and job["owned_children_reaped"]
            and not job["cleanup_actions"]
        )
        save(path, job)
    return job


def code_sections(path, temporary):
    values = {}
    headers = subprocess.run(
        ["readelf", "-W", "-S", str(path)], capture_output=True, text=True, check=True
    ).stdout
    for name in re.findall(r"\[\s*\d+\]\s+(\S+)\s+PROGBITS", headers):
        if name == ".text" or "nv_fatbin" in name:
            target = temporary / (path.name + name.replace("/", "_"))
            subprocess.run(
                [
                    "objcopy",
                    "--dump-section",
                    name + "=" + str(target),
                    str(path),
                    "/dev/null",
                ],
                capture_output=True,
                check=True,
            )
            values[name] = digest(target)
    return values


def bootstrap(root, started, expires, *, local_cpu_rehearsal=False):
    preflight(root)
    if sys.version_info[:2] != (3, 12):
        message = "Pinned Python3.12 required"
        raise RuntimeError(message)
    neutral = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    (root / "work").mkdir(exist_ok=False)
    legacy.safe_extract(
        root / "archives/oracle-packages.tar.gz", root / "work/dependencies"
    )
    for variant in ("baseline", "candidate"):
        job = run_owned(
            root,
            [sys.executable, "-m", "venv", str(root / "work" / variant)],
            "venv-" + variant,
            time_budget(started, expires, 60),
            neutral,
        )
        if not job["passed"]:
            message = "Venv creation failed"
            raise RuntimeError(message)
        site = root / "work" / variant / "lib/python3.12/site-packages"
        for package in (root / "python").iterdir():
            if package.is_dir():
                import shutil

                shutil.copytree(package, site / package.name)
        (site / "pilot.pth").write_text(
            str(root / "work/dependencies") + "\n" + str(root) + "\n"
        )
        if (
            digest(site / "vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so")
            != REFERENCE_SHA
        ):
            message = "Whole package overlay failed"
            raise RuntimeError(message)
    import shutil

    shutil.copytree(root / "lib", root / "work/lib")
    shutil.copytree(root / "bin", root / "work/bin")
    scratch = root / "work/section-hashes"
    scratch.mkdir()
    relocation = []
    extension = (
        root
        / "work/candidate/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    )
    for path in [
        *sorted((root / "work/lib").glob("*.so")),
        *(p for p in (root / "work/bin").iterdir() if p.name != "patchelf"),
    ]:
        before, sections = digest(path), code_sections(path, scratch)
        dynamic = subprocess.run(
            ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
        ).stdout
        for needed in re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic):
            if needed.startswith("/"):
                replacement = (
                    str(extension) if "_vmecpp" in needed else Path(needed).name
                )
                subprocess.run(
                    [
                        str(root / "bin/patchelf"),
                        "--replace-needed",
                        needed,
                        replacement,
                        str(path),
                    ],
                    check=True,
                )
        subprocess.run(
            [
                str(root / "bin/patchelf"),
                "--set-rpath",
                str(root / "runtime") + ":" + str(root / "work/lib"),
                str(path),
            ],
            check=True,
        )
        if sections != code_sections(path, scratch):
            message = "Relocation changed code/fatbin"
            raise RuntimeError(message)
        relocation.append(
            {
                "path": str(path),
                "before": before,
                "after": digest(path),
                "code_sections": sections,
                "code_unchanged": True,
            }
        )
    save(root / "relocation.json", relocation)
    for name in ("nsys", "ncu"):
        legacy.safe_extract(
            root / "archives" / (name + "-cli.tar.gz"), root / "work" / name
        )
    shutil.copytree(root / "profiling/nsys-host", root / "work/nsys/host-linux-x64")
    loader_environment = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    loader_environment["LD_LIBRARY_PATH"] = (
        str(root / "runtime") + ":" + str(root / "work/lib")
    )
    verify_calibration_loader(root, loader_environment)
    for variant in ("baseline", "candidate"):
        command = [
            str(root / "work" / variant / "bin/python"),
            str(Path(__file__)),
            "import-check",
            "--root",
            str(root),
            "--variant",
            variant,
        ]
        job = run_owned(
            root,
            command,
            "import-" + variant,
            time_budget(started, expires, 45),
            environment(root, variant, allow_unselected=True),
        )
        if not job["passed"]:
            message = "Import/loaded-extension/ABI check failed"
            raise RuntimeError(message)
    save(
        root / "bootstrap-status.json",
        {
            "phase": "CPU_compatibility_pending",
            "relocation_code_unchanged": True,
            "matching_extension": REFERENCE_SHA,
            "local_cpu_rehearsal": local_cpu_rehearsal,
            "scope": "Local CPU extraction/relocation/import rehearsal; no lease/VMEC/GPU workload"
            if local_cpu_rehearsal
            else "Actual existing-lease bootstrap; no equilibrium yet",
            "runtime_files": {
                str(path.relative_to(root)): digest(path)
                for directory in (root / "work/lib", root / "work/bin")
                for path in directory.iterdir()
                if path.is_file()
            },
        },
    )
    save(root / "results/provenance/pilot-config.json", config(root))
    shutil.copy2(
        root / "provenance/prior-canonical-evidence.json",
        root / "results/provenance/prior-canonical-evidence.json",
    )
    save(
        root / "results/provenance/payload-manifest.json",
        json.loads((root / "payload-manifest.json").read_text()),
    )

    finish_cpu_compatibility(root, started, expires)
    clock_witness(root, "bootstrap")
    state = json.loads((root / "bootstrap-status.json").read_text())
    state.update(phase="raw_dependency_closure_pending")
    save(root / "bootstrap-status.json", state)
    try:
        closure = rehearse_raw_contract(root, started, expires)
    except BaseException as error:
        state.update(phase="failed_raw_dependency_closure", error=repr(error))
        save(root / "bootstrap-status.json", state)
        raise
    state.update(phase="complete", raw_dependency_closure=closure)
    save(root / "bootstrap-status.json", state)


def import_check(root, variant):
    cpu_state = floating_point_guard(root)
    import vmecpp

    module = importlib.import_module("vmecpp.cpp._vmecpp")
    assert module.__file__ is not None
    expected = (
        root
        / "work"
        / variant
        / "lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    )
    if (
        Path(module.__file__).resolve() != expected.resolve()
        or digest(expected) != REFERENCE_SHA
    ):
        message = "Imported old/mismatched extension instead of current144e"
        raise RuntimeError(message)
    if not Path(vmecpp.__file__).resolve().is_relative_to(expected.parents[1]):
        message = "Mixed vmecpp Python package/extension"
        raise RuntimeError(message)
    dependencies = subprocess.run(
        ["ldd", str(expected)],
        capture_output=True,
        text=True,
        check=True,
        env=environment(root, "baseline", allow_unselected=True),
    ).stdout
    resolved = {}
    for line in dependencies.splitlines():
        words = line.split()
        if len(words) >= 3 and words[1] == "=>" and Path(words[2]).is_file():
            resolved[words[0]] = {"path": words[2], "sha256": digest(words[2])}
    for name, expected_hash in config(root)["expected_core_sha256"].items():
        if resolved.get(name, {}).get("sha256") != expected_hash:
            message = (
                "Actual loaded core-math dependency differs from verified ABI baseline: "
                + name
            )
            raise RuntimeError(message)
    versions = {
        name: metadata.version(name)
        for name in ("vmecpp", "constellaration", "booz-xform", "numpy")
    }
    if (
        versions["vmecpp"] != "0.4.11"
        or versions["constellaration"] != "0.2.6"
        or versions["booz-xform"] != "0.0.9"
    ):
        message = "Pinned oracle dependency version drift"
        raise RuntimeError(message)
    save(
        root / "results" / ("import-" + variant + ".json"),
        {
            "extension": str(expected),
            "sha256": digest(expected),
            "versions": versions,
            "CPU_floating_point_state": cpu_state,
            "resolved_dependencies": resolved,
        },
    )


def floating_point_guard(root):
    guard = ctypes.CDLL(str(root / "work/bin/pilot-cpu-guard.so"))
    guard.VmecPilotMxcsr.restype = ctypes.c_uint32
    guard.VmecPilotRoundingMode.restype = ctypes.c_int
    guard.VmecPilotFloatingPointStateSupported.restype = ctypes.c_bool
    state = {
        "mxcsr": int(guard.VmecPilotMxcsr()),
        "rounding_mode": int(guard.VmecPilotRoundingMode()),
        "read_only": True,
    }
    if not guard.VmecPilotFloatingPointStateSupported():
        message = "Unsupported CPU rounding/FTZ/DAZ; never modify physics environment"
        raise RuntimeError(message)
    return state


def profiled_call(function, start, stop, push, pop, bounds):
    """Stop on every exit after successful start; never perform record IO here."""
    if start() != 0:
        message = "cudaProfilerStart failed"
        raise RuntimeError(message)
    opened = False
    started = None
    try:
        if push() < 0:
            message = "NVTX whole-call marker unavailable"
            raise RuntimeError(message)
        opened = True
        started = time.perf_counter_ns()
        return function()
    finally:
        ended = time.perf_counter_ns()
        active_error = sys.exc_info()[0]
        try:
            if opened:
                pop()
        finally:
            stopped = stop()
            bounds.append(
                {
                    "before_actual_vmecpp_run_ns": started,
                    "after_return_ns": ended,
                    "stop_status": stopped,
                    "scope": "whole vmecpp.run NVTX; no file IO inside range",
                }
            )
        if stopped != 0 and active_error is None:
            message = "cudaProfilerStop failed"
            raise RuntimeError(message)


def clock_reading():
    """Bracket one MONOTONIC read with RAW reads; no file IO or float clocks."""
    tid = threading.get_native_id()
    raw_before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    monotonic = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    raw_after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return {
        "tid": tid,
        "raw_before_ns": raw_before,
        "monotonic_ns": monotonic,
        "raw_after_ns": raw_after,
    }


def clock_pair(before, after):
    if before["tid"] != after["tid"] or any(
        reading["raw_after_ns"] < reading["raw_before_ns"]
        for reading in (before, after)
    ):
        message = "Clock bookends changed TID or inverted RAW bracket"
        raise RuntimeError(message)
    minimum = after["raw_before_ns"] - before["raw_after_ns"]
    maximum = after["raw_after_ns"] - before["raw_before_ns"]
    mono = after["monotonic_ns"] - before["monotonic_ns"]
    if minimum < 0 or mono < 0:
        message = "Clock bookends inverted"
        raise RuntimeError(message)
    return {
        "before": before,
        "after": after,
        "raw_elapsed_lower_ns": minimum,
        "raw_elapsed_upper_ns": maximum,
        "monotonic_elapsed_ns": mono,
        "monotonic_within_raw_read_bounds": minimum <= mono <= maximum,
        "scope": "integer clock bookends; no wait/CPU decomposition inferred",
    }


def clocked_call(function, records):
    before = clock_reading()
    try:
        return function()
    finally:
        after = clock_reading()
        records.append(clock_pair(before, after))


def clock_witness(root, label="pre-official"):
    samples = []
    for _ in range(2):
        before = clock_reading()
        time.sleep(0.1)
        samples.append(clock_pair(before, clock_reading()))
    result = {
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
        "kernel_release": os.uname().release,
        "python_version": sys.version,
        "clock_constants": {
            "raw": time.CLOCK_MONOTONIC_RAW,
            "monotonic": time.CLOCK_MONOTONIC,
        },
        "clock_resolution_seconds": {
            "raw": time.clock_getres(time.CLOCK_MONOTONIC_RAW),
            "monotonic": time.clock_getres(time.CLOCK_MONOTONIC),
        },
        "affinity": sorted(os.sched_getaffinity(0)),
        "samples": samples,
        "interpretation": "RAW and MONOTONIC preserved independently; coherence does not alter original report timers",
    }
    save(root / ("results/provenance/cpu-clock-witness-" + label + ".json"), result)
    return result


def worker(root, variant, arguments, profile=False):
    output = Path(arguments[arguments.index("--out") + 1])
    records = []
    try:
        return clocked_call(
            lambda: worker_body(root, variant, arguments, profile), records
        )
    finally:
        save(
            output.with_suffix(".worker-clock.json"),
            {
                "pid": os.getpid(),
                "birth": birth(os.getpid()),
                "records": records,
                "scope": "Python worker including imports/benchmark/validation; clock-record IO follows final bookend",
            },
        )


def worker_body(root, variant, arguments, profile=False):
    verify_worker_dispatch(root)
    import numpy as np

    import vmecpp
    from tools.optimization import benchmark

    import_check(root, variant)
    output = Path(arguments[arguments.index("--out") + 1])
    save(
        output.with_suffix(".worker-start.json"),
        {
            "pid": os.getpid(),
            "birth": birth(os.getpid()),
            "variant": variant,
            "profiled": profile,
        },
    )
    original = vmecpp.run
    activity = None
    if variant == "candidate":
        backend = ctypes.CDLL(str(root / "work/lib/solver-backend.so"))
        activity = backend.GpuSolverThreadActivity
        activity.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        activity.restype = None
    counts_before = (ctypes.c_uint64 * 3)()
    counts_after = (ctypes.c_uint64 * 3)()
    if activity is not None:
        activity(counts_before)
    observed = []
    bounds = []
    call_clocks = []
    cudart = ctypes.CDLL(str(root / "runtime/libcudart.so.12")) if profile else None
    markers = (
        ctypes.CDLL(str(root / "work/bin/pilot-cpu-guard.so")) if profile else None
    )

    def capture(*args, **kwargs):
        floating_point_guard(root)
        if profile:
            if threading.current_thread() is not threading.main_thread():
                message = "Profiler lifecycle must remain on the worker main thread"
                raise RuntimeError(message)
            assert cudart is not None
            assert markers is not None
            result = profiled_call(
                lambda: clocked_call(lambda: original(*args, **kwargs), call_clocks),
                cudart.cudaProfilerStart,
                cudart.cudaProfilerStop,
                markers.VmecPilotRangePush,
                markers.VmecPilotRangePop,
                bounds,
            )
        else:
            result = clocked_call(lambda: original(*args, **kwargs), call_clocks)
        observed.append(result)
        return result

    vmecpp.run = capture
    previous = sys.argv
    worker_started = clock_reading()
    try:
        sys.argv = [str(Path(benchmark.__file__)), *arguments]
        code = benchmark.main()
    finally:
        worker_ended = clock_reading()
        vmecpp.run = original
        sys.argv = previous
        save(
            output.with_suffix(".worker-python-finally.json"),
            {
                "pid": os.getpid(),
                "birth": birth(os.getpid()),
                "bounds": bounds,
                "actual_vmecpp_run_clocks": call_clocks,
                "outer_benchmark_clocks": clock_pair(worker_started, worker_ended),
                "scope": "Python benchmark returned/unwound; actual process termination is separately proven by supervisor",
            },
        )
    if activity is not None:
        activity(counts_after)
    delta = [
        int(after) - int(before)
        for before, after in zip(counts_before, counts_after, strict=True)
    ]
    if code != 0 or (variant == "candidate" and not all(value > 0 for value in delta)):
        message = "Evaluation failed or real GPU create/run/graph activation absent"
        raise RuntimeError(message)
    for index, value in enumerate(observed):
        arrays = {
            section + "." + name: np.asarray(array)
            for section in ("jxbout", "mercier", "threed1_volumetrics")
            for name, array in getattr(value, section).model_dump().items()
            if isinstance(array, int | float | bool | np.ndarray | np.number)
        }
        with zipfile.ZipFile(output / f"raw_diagnostics_{index}.npz", "w") as archive:
            for name, array in arrays.items():
                with archive.open(name + ".npy", "w") as entry:
                    np.lib.format.write_array(
                        entry, np.ascontiguousarray(array), allow_pickle=False
                    )
    save(
        output / "activation.json",
        {
            "variant": variant,
            "CPU_baseline_label": BASELINE_LABEL,
            "pid": os.getpid(),
            "birth": birth(os.getpid()),
            "host_TLS_activity_deltas": delta,
            "actual_GPU_activation": variant == "candidate"
            and all(value > 0 for value in delta),
            "profiled": profile,
            "bounds": bounds,
            "actual_vmecpp_run_clocks": call_clocks,
            "outer_benchmark_clocks": clock_pair(worker_started, worker_ended),
        },
    )
    report = json.loads((output / "report.json").read_text())
    report.update(
        CPU_baseline_label=BASELINE_LABEL,
        pilot_variant=variant,
        pilot_scope="current144e correctness/hardware pilot; no stockwheel or main-table latency comparison",
    )
    if profile:
        report.update(
            profiled=True,
            measurement="external full-VMEC profiling diagnostic; no timing qualification",
        )
    save(output / "report.json", report)
    return code


def raw_compare_inline(baseline, candidate):
    import numpy as np

    Dataset = importlib.import_module("netCDF4").Dataset

    files = sorted(baseline.glob("raw_wout_*.nc"))
    if not files or len(files) != len(list(candidate.glob("raw_wout_*.nc"))):
        message = "Missing raw numerical wout files"
        raise ValueError(message)
    checks = []
    for path in files:
        with Dataset(path) as first, Dataset(candidate / path.name) as second:
            if set(first.variables) != set(second.variables):
                message = "Raw NC variable set differs"
                raise ValueError(message)
            for name, variable in first.variables.items():
                if np.dtype(variable.dtype).kind not in "biufc":
                    continue
                a = np.ascontiguousarray(variable[...])
                b = np.ascontiguousarray(second.variables[name][...])
                if (
                    a.shape != b.shape
                    or a.dtype != b.dtype
                    or a.tobytes() != b.tobytes()
                ):
                    message = "Raw NC field differs: " + name
                    raise ValueError(message)
                checks.append(
                    {
                        "file": path.name,
                        "field": name,
                        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                    }
                )
    diagnostics = sorted(baseline.glob("raw_diagnostics_*.npz"))
    if not diagnostics or len(diagnostics) != len(
        list(candidate.glob("raw_diagnostics_*.npz"))
    ):
        message = "Missing raw diagnostics"
        raise ValueError(message)
    for path in diagnostics:
        with np.load(path) as a, np.load(candidate / path.name) as b:
            if set(a.files) != set(b.files):
                message = "Raw diagnostic field set differs"
                raise ValueError(message)
            for name in a.files:
                if (
                    a[name].shape != b[name].shape
                    or a[name].dtype != b[name].dtype
                    or a[name].tobytes() != b[name].tobytes()
                ):
                    message = "Raw diagnostic field differs: " + name
                    raise ValueError(message)
                checks.append(
                    {
                        "file": path.name,
                        "field": name,
                        "sha256": hashlib.sha256(a[name].tobytes()).hexdigest(),
                    }
                )
    return {"exact": bool(checks), "numeric_fields": checks}


def raw_compare_worker(root, baseline, candidate, result):
    verify_worker_dispatch(root)
    if os.environ.get("LD_PRELOAD") or any(
        key.startswith("VMECPP_") for key in os.environ
    ):
        message = "Raw numerical comparison must use clean CPU runtime"
        raise RuntimeError(message)
    if (
        any(
            not path.resolve().is_relative_to((root / "results").resolve())
            for path in (baseline, candidate, result)
        )
        or result.exists()
    ):
        message = "Raw comparison inputs/result must be isolated fresh pilot outputs"
        raise ValueError(message)
    proof = json.loads((root / "profiling/raw-closure/dependencies.json").read_text())
    payload = json.loads((root / "payload-manifest.json").read_text())
    if proof["archive"] != "archives/oracle-packages.tar.gz" or proof[
        "archive_sha256"
    ] != payload["files"].get(proof["archive"]):
        message = "Raw dependency proof does not bind the immutable oracle archive"
        raise RuntimeError(message)
    provenance = {}
    for name in ("numpy", "netCDF4", "cftime"):
        module = importlib.import_module(name)
        filename = getattr(module, "__file__", None)
        expected = proof["modules"][name]
        if expected["archive_member"] != name + "/__init__.py":
            message = "Noncanonical raw dependency archive member"
            raise RuntimeError(message)
        path = root / "work/dependencies" / expected["archive_member"]
        valid = (
            isinstance(filename, str)
            and Path(filename).resolve() == path.resolve()
            and digest(path) == expected["sha256"]
        )
        provenance[name] = {
            "module_path": filename,
            "expected_path": str(path),
            "version": getattr(module, "__version__", None),
            "archive_member_sha256": expected["sha256"],
            "matches_verified_archive": valid,
        }
        save(
            result.with_suffix(".dependency.json"),
            {"archive_sha256": proof["archive_sha256"], "modules": provenance},
        )
        if not valid:
            message = (
                "Raw numerical dependency escaped verified pinned archive module: "
                + name
            )
            raise RuntimeError(message)
    comparison = raw_compare_inline(baseline, candidate)
    if comparison.get("exact") is not True:
        message = "Raw numerical comparison did not validate fields"
        raise RuntimeError(message)
    comparison["dependency_provenance"] = provenance
    comparison["runtime_python"] = sys.executable
    comparison["scope"] = (
        "Pinned clean CPU raw-array comparator only; no VMEC/GPU evaluation"
    )
    save(result, comparison)


def raw_compare(root, baseline, candidate, started, expires):
    label = "raw-" + baseline.name + "--" + candidate.name
    result = root / "results/raw-checks" / (label + ".json")
    command = [
        str(root / "work/baseline/bin/python"),
        "-B",
        str(Path(__file__)),
        "raw-compare",
        "--root",
        str(root),
        "--left",
        str(baseline),
        "--right",
        str(candidate),
        "--result",
        str(result),
    ]
    job = run_owned(
        root,
        command,
        label,
        time_budget(started, expires, 60),
        environment(root, "baseline"),
    )
    if not job["passed"] or not result.is_file():
        message = "Pinned CPU raw comparison failed or produced no result"
        raise RuntimeError(message)
    comparison = json.loads(result.read_text())
    if comparison.get("exact") is not True or not comparison.get("numeric_fields"):
        message = "Pinned CPU raw comparison result is incomplete"
        raise RuntimeError(message)
    return comparison


def rehearse_raw_contract(root, started, expires):
    import shutil

    source = root / "profiling/raw-closure"
    expected = json.loads((source / "manifest.json").read_text())
    for name, value in expected.items():
        if digest(legacy.inside(source, name)) != value:
            message = "Saved raw closure input hash mismatch"
            raise ValueError(message)
    directory = root / "results/raw-dependency-closure"
    paths = [
        directory / name for name in ("baseline", "candidate", "saved-copy-repeat")
    ]
    for path, name in zip(paths, ("baseline", "candidate", "candidate"), strict=True):
        shutil.copytree(source / name, path)
        gate = report_gate(root, path, root / "fixtures/public_p3_seed.json", "P3")
        save(path / "original-global-gate.json", gate)
    raw = raw_compare(root, paths[0], paths[1], started, expires)
    repeated = raw_compare(root, paths[1], paths[2], started, expires)
    gate = case_gate(paths[0], paths[1], paths[2], raw, repeated)
    result = {
        "exact": gate["exact"],
        "gate": gate,
        "scope": "CPU replay of saved actual A100 CPU/GPU outputs plus cloned saved-output repeat; no new VMEC solve or GPU determinism claim",
        "GPU_runs": 0,
        "VMEC_solves": 0,
        "new_GPU_candidate_processes": 0,
    }
    save(directory / "contract.json", result)
    return {
        "exact": result["exact"],
        "scope": result["scope"],
        "path": str(directory / "contract.json"),
    }


def report_gate(root, output, seed, track, profiled=False):
    reference = root / "references" / (seed.stem + "-" + track)
    report = json.loads((output / "report.json").read_text())
    if profiled:
        # Diagnostic-only numeric comparison; stored report remains profiled=True.
        report = dict(report, profiled=False)
    gate = gates.pair_gate(
        json.loads((reference / "report.json").read_text()),
        report,
        gates.read_capture(reference),
        gates.read_capture(output),
        require_harness=False,
    )
    gate.update(profiled=profiled, main_table_eligible=False)
    if not gate["exact"]:
        message = "Original global identity/deck/all numeric output gate failed"
        raise RuntimeError(message)
    return gate


def execute_wrapper(root, variant, arguments):
    python = root / "work" / variant / "bin/python"
    os.execve(str(python), [str(python), "-B", *arguments], environment(root, variant))


def profile_environment(root, variant, inherited):
    from tools.optimization.nsys_injection_guard import compose_profile_environment

    if variant != "candidate":
        message = "Profile-only routing requires canonical candidate"
        raise ValueError(message)
    return compose_profile_environment(root, environment(root, variant), inherited)


def evaluate(
    root,
    seed,
    track,
    variant,
    label,
    started,
    expires,
    profile=False,
    command_prefix=None,
):
    selected_cpu_environment(root)
    output = root / "results/evaluations" / label
    args = [
        "--stella",
        str(root / "stella-snapshot"),
        "--seed",
        str(seed),
        "--track",
        track,
        "--out",
        str(output),
    ]
    action = "profile-worker" if profile else "worker"
    child = [
        str(root / "work" / variant / "bin/python"),
        str(Path(__file__)),
        action,
        "--root",
        str(root),
        "--variant",
        variant,
        "--",
        *args,
    ]
    if command_prefix:
        child = [
            *command_prefix,
            str(root / "work" / variant / "bin/python"),
            str(Path(__file__)),
            "profile-exec",
            "--root",
            str(root),
            "--variant",
            variant,
            "--",
            *child[1:],
        ]
        env = {
            key: os.environ[key]
            for key in ("HOME", "PATH", "LANG", "LC_ALL")
            if key in os.environ
        }
    else:
        env = environment(root, variant)
    job = run_owned(root, child, label, time_budget(started, expires, 240), env)
    if not job["passed"]:
        message = "Evaluation/owned target completion failed"
        raise RuntimeError(message)
    gate = report_gate(root, output, seed, track, profile)
    save(output / "original-global-gate.json", gate)
    return output


def locate_tool(root, name):
    paths = [path for path in (root / "work" / name).rglob(name) if path.is_file()]
    if len(paths) != 1:
        message = "Unique profiler executable required: " + name
        raise RuntimeError(message)
    return paths[0]


def hardware_snapshot(root, label):
    env = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw",
        "--format=csv,noheader",
    ]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=10, check=False
    )
    record = {
        "command": command,
        "exit_status": result.returncode,
        "csv": result.stdout,
        "stderr": result.stderr,
        "time": time.time(),
        "process_affinity": sorted(os.sched_getaffinity(0)),
        "clock_or_power_changes": False,
    }
    save(root / "results/hardware" / (label + ".json"), record)
    if result.returncode != 0:
        message = "Actual GPU hardware metadata unavailable"
        raise RuntimeError(message)
    return record


def calibration_sizes(l2_bytes):
    """Choose TOTAL two-buffer working MiB from actual device metadata."""
    mib = 1 << 20
    if not isinstance(l2_bytes, int) or isinstance(l2_bytes, bool) or l2_bytes <= 0:
        message = "Actual positive integer L2 capacity required"
        raise ValueError(message)
    l2 = l2_bytes // (2 * mib)
    dram = max(256, (4 * l2_bytes + mib - 1) // mib)
    if not 1 <= l2 <= 512 or not 1 <= dram <= 512:
        message = "Cache/DRAM working sets outside measured calibration CLI bounds"
        raise ValueError(message)
    return {
        "actual_l2_bytes": l2_bytes,
        "L2_total_working_MiB": l2,
        "DRAM_total_working_MiB": dram,
        "L2_total_working_bytes": l2 * mib,
        "DRAM_total_working_bytes": dram * mib,
        "interpretation": "capacity-based sizes; actual L2/DRAM traffic counters still required",
    }


def counters(root, started, expires, permission_only=False):
    """Actual counter invocation required; query success alone proves nothing."""
    ncu = locate_tool(root, "ncu")
    neutral = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    neutral["LD_LIBRARY_PATH"] = str(root / "runtime")
    query_label = "ncu-query-permission" if permission_only else "ncu-query-current"
    wanted = [
        "gpu__time_duration.sum",
        "dram__bytes.sum",
        "lts__t_bytes.sum",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
        "sm__cycles_elapsed.avg",
        "sm__cycles_elapsed.avg.per_second",
        "lts__cycles_elapsed.avg",
        "lts__cycles_elapsed.avg.per_second",
        "dram__cycles_elapsed.avg",
        "dram__cycles_elapsed.avg.per_second",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "launch__registers_per_thread",
        "launch__grid_size",
        "launch__block_size",
    ]

    query_bases = sorted(
        {name.split(".", 1)[0] for name in wanted if not name.startswith("launch__")}
        | {"lts__t_sectors"}
    )
    job = run_owned(
        root,
        [
            str(ncu),
            "--query-metrics-mode",
            "suffix",
            "--metrics",
            ",".join(query_bases),
        ],
        query_label,
        time_budget(started, expires, 30),
        neutral,
    )
    text = (root / "logs" / (query_label + ".log")).read_text()
    if "ERR_NVGPUCTRPERM" in text:
        return {
            "counter_permission": False,
            "reason": "ERR_NVGPUCTRPERM",
            "actual_probe_executed": False,
        }
    if not job["passed"]:
        return {
            "counter_permission": False,
            "reason": "metric-query failed",
            "actual_probe_executed": False,
        }
    metrics = [name for name in wanted if name in text or name.startswith("launch__")]
    required = wanted[:6]
    if required[2] not in metrics and "lts__t_sectors.sum" in text:
        required[2] = "lts__t_sectors.sum"
        metrics.append("lts__t_sectors.sum")
    if not set(required).issubset(metrics):
        return {
            "counter_permission": None,
            "reason": "required physical/FP64 metrics unsupported",
            "missing_metrics": list(set(required) - set(metrics)),
            "actual_probe_executed": False,
        }
    cases = [
        (
            "permission",
            [str(root / "work/bin/roofline-matched"), "compute", "16"],
            "StrictMixed",
        )
    ]
    sizing = None
    if not permission_only:
        permission = json.loads((root / "results/counter-permission.json").read_text())
        if permission.get("counter_values_validated") is not True:
            message = "Actual validated permission calibration metadata required"
            raise ValueError(message)
        sizing = calibration_sizes(permission["application_metadata"]["l2_bytes"])
        cases = [
            (
                "compute",
                [str(root / "work/bin/roofline-matched"), "compute", "128"],
                "StrictMixed",
            ),
            (
                "memory-L2-" + str(sizing["L2_total_working_MiB"]),
                [
                    str(root / "work/bin/roofline-matched"),
                    "memory",
                    str(sizing["L2_total_working_MiB"]),
                ],
                "CopyVectors",
            ),
            (
                "memory-DRAM-" + str(sizing["DRAM_total_working_MiB"]),
                [
                    str(root / "work/bin/roofline-matched"),
                    "memory",
                    str(sizing["DRAM_total_working_MiB"]),
                ],
                "CopyVectors",
            ),
        ]
        for kernel in ("ForwardPartials", "ForwardFinish"):
            cases.append(
                (
                    kernel,
                    [
                        str(root / "work/bin/groups-benchmark"),
                        str(root / "captures/real-forward.bin"),
                        str(root / "work/lib/baseline-resident.so"),
                        str(root / "work/lib/resident.so"),
                        str(root / "results" / (kernel + "-parity.json")),
                        "1",
                        "1",
                        "profile",
                    ],
                    kernel,
                )
            )
    results = []
    (root / "results/profiles").mkdir(parents=True, exist_ok=True)
    for label, application, kernel in cases:
        hardware_snapshot(root, "ncu-" + label + "-before")
        target = root / "results/profiles" / label
        command = [
            str(ncu),
            "--profile-from-start",
            "off",
            "--cache-control",
            "none",
            "--clock-control",
            "none",
            "--replay-mode",
            "kernel",
            "--kernel-name",
            "regex:" + kernel,
            "--launch-count",
            "1",
            "--kill",
            "0",
            "--metrics",
            ",".join(metrics),
            "--export",
            str(target),
            *application,
        ]
        job = run_owned(
            root, command, "ncu-" + label, time_budget(started, expires, 60), neutral
        )
        hardware_snapshot(root, "ncu-" + label + "-after")
        log = (root / "logs" / ("ncu-" + label + ".log")).read_text()
        if "ERR_NVGPUCTRPERM" in log:
            return {
                "counter_permission": False,
                "reason": "ERR_NVGPUCTRPERM",
                "actual_probe_executed": True,
                "job": job,
            }
        if not job["passed"] or not target.with_suffix(".ncu-rep").is_file():
            return {
                "counter_permission": None,
                "reason": "actual probe/report failed",
                "actual_probe_executed": True,
                "job": job,
            }
        application_metadata = None
        if not label.startswith("Forward"):
            for line in log.splitlines():
                with contextlib.suppress(ValueError):
                    parsed = json.loads(line)
                    if (
                        isinstance(parsed, dict)
                        and parsed.get("scope") == "counter-matched calibration only"
                    ):
                        application_metadata = parsed
            if (
                not application_metadata
                or application_metadata.get("all_output_words_verified") is not True
            ):
                return {
                    "counter_permission": None,
                    "reason": "actual calibration output-word proof missing",
                    "actual_probe_executed": True,
                }
            if label.startswith("memory-"):
                assert sizing is not None
                expected_bytes = (
                    sizing["L2_total_working_bytes"]
                    if label.startswith("memory-L2-")
                    else sizing["DRAM_total_working_bytes"]
                )
                if (
                    application_metadata.get("working_set_bytes") != expected_bytes
                    or application_metadata.get("l2_bytes") != sizing["actual_l2_bytes"]
                ):
                    return {
                        "counter_permission": None,
                        "reason": "actual cache footprint/device metadata differs",
                        "actual_probe_executed": True,
                    }
        else:
            parity = root / "results" / (label + "-parity.json")
            application_metadata = (
                json.loads(parity.read_text()) if parity.is_file() else None
            )
            if (
                not application_metadata
                or application_metadata.get("all12_partials_and6_outputs_exact")
                is not True
            ):
                return {
                    "counter_permission": None,
                    "reason": "actual Forward partial/output proof missing",
                    "actual_probe_executed": True,
                }
        export = run_owned(
            root,
            [
                str(ncu),
                "--import",
                str(target.with_suffix(".ncu-rep")),
                "--csv",
                "--page",
                "raw",
                "--print-units",
                "base",
            ],
            "ncu-export-" + label,
            time_budget(started, expires, 30),
            neutral,
        )
        csv_text = (root / "logs" / ("ncu-export-" + label + ".log")).read_text()
        validation = legacy.validate_ncu_values(csv_text, metrics)
        valid = export["passed"] and legacy.ncu_roofline_inputs_present(
            validation, set(required)
        )
        result = {
            "case": label,
            "report_created": True,
            "counter_values_validated": valid,
            "validation": validation,
            "application_metadata": application_metadata,
            "scope": "single selected captured kernel or matched calibration, not whole solver",
        }
        results.append(result)
        save(root / "results" / ("counter-" + label + ".json"), result)
        if not valid:
            return {
                "counter_permission": None,
                "reason": "required finite values/units missing",
                "actual_probe_executed": True,
                "results": results,
            }
    return {
        "counter_permission": True,
        "actual_probe_executed": True,
        "metrics": metrics,
        "missing_metrics": [name for name in wanted if name not in metrics],
        "results": results,
        "L2_traffic_metric": required[2],
        "L2_conversion": "not performed; retain actual reported units",
        "calibration_working_sets": sizing,
    }


def require_hardware_policy(value):
    if value.get("counter_policy") != "timing-and-trace":
        msg = "Explicit immutable timing-and-trace counter policy required"
        raise ValueError(msg)


def case_gate(baseline, candidate, repeat, raw, repeated_raw):
    """Derive the controller contract from actual persisted comparison evidence."""
    originals = [
        json.loads((path / "original-global-gate.json").read_text())
        for path in (baseline, candidate, repeat)
    ]
    exact = all(item.get("exact") is True for item in originals) and all(
        item.get("exact") is True for item in (raw, repeated_raw)
    )
    if not exact:
        msg = "Case identity/raw/repeat contract failed"
        raise RuntimeError(msg)
    return {
        "exact": exact,
        "original_global_gates": originals,
        "raw_comparison": raw,
        "repeat_comparison": repeated_raw,
        "main_table_eligible": False,
    }


def calibration_interval(database, kernel, application):
    """Accept one captured kernel only; do not manufacture physical traffic."""
    if (
        application.get("all_output_words_verified") is not True
        or application.get("profiled_graph_launches") != 1
        or application.get("warmup_graph_launches") != 3
        or type(application.get("l2_bytes")) is not int
        or application["l2_bytes"] <= 0
    ):
        msg = "Verified calibration output and actual hardware metadata required"
        raise ValueError(msg)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT k.start,k.end,k.globalPid,s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
            "ON k.demangledName=s.id"
        ).fetchall()
    if len(rows) != 1:
        msg = "Exactly one captured calibration kernel required"
        raise ValueError(msg)
    start, end, pid, name = rows[0]
    if (
        kernel not in name
        or type(start) is not int
        or type(end) is not int
        or end <= start
        or type(pid) is not int
        or pid <= 0
    ):
        msg = "Invalid calibration kernel interval/identity"
        raise ValueError(msg)
    expected_mode = "compute" if kernel == "StrictMixed" else "memory"
    if application.get("mode") != expected_mode:
        msg = "Calibration application/kernel mode mismatch"
        raise ValueError(msg)
    return {
        "scope": "one actual GPU kernel interval; no physical traffic/FLOP/roofline inference",
        "kernel": name,
        "start_ns": start,
        "end_ns": end,
        "duration_ns": end - start,
        "global_pid": pid,
        "application_metadata": application,
        "counter_evidence": False,
        "roofline_claim": False,
        "sqlite_sha256": digest(database),
    }


def verify_calibration_loader(root, env):
    expected_path = str(root / "runtime") + ":" + str(root / "work/lib")
    if env.get("LD_LIBRARY_PATH") != expected_path or "LD_PRELOAD" in env:
        message = "Explicit canonical calibration loader environment required"
        raise ValueError(message)
    expected_core = config(root)["expected_core_sha256"]
    importer = root / "work/nsys/host-linux-x64/QdstrmImporter"
    paths = [root / "work/bin/roofline-matched", locate_tool(root, "nsys"), importer]
    records = []
    for path in paths:
        completed = subprocess.run(
            ["ldd", str(path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = completed.stdout + completed.stderr
        if completed.returncode != 0 or "not found" in text or "/stubs/" in text:
            message = (
                "Calibration/tool loader missing dependency or driver stub: "
                + str(path)
            )
            raise RuntimeError(message)
        resolved = {}
        for line in text.splitlines():
            words = line.split()
            if len(words) >= 3 and words[1] == "=>" and words[2].startswith("/"):
                library = Path(words[2]).resolve()
                if not library.is_file() or "stubs" in library.parts:
                    message = "Unresolved/stub calibration dependency"
                    raise RuntimeError(message)
                resolved[words[0]] = {"path": str(library), "sha256": digest(library)}
                if words[0] in ("libc.so.6", "libm.so.6") and (
                    library.is_relative_to(root)
                    or digest(library) != expected_core.get(words[0])
                ):
                    message = "Calibration/tool private or mismatched core math library"
                    raise RuntimeError(message)
        if "libc.so.6" not in resolved:
            message = "Actual calibration/tool libc resolution required"
            raise RuntimeError(message)
        runtime_linkage = None
        if path.name == "roofline-matched":
            if "libcudart.so.12" in resolved:
                if resolved["libcudart.so.12"]["path"] != str(
                    (root / "runtime/libcudart.so.12").resolve()
                ):
                    message = "Calibration did not resolve the exact copied cudart"
                    raise RuntimeError(message)
                runtime_linkage = {"kind": "dynamic copied cudart"}
            else:
                symbols = subprocess.run(
                    ["nm", "--defined-only", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                ).stdout
                required = (
                    "cudaProfilerStart",
                    "cudaGraphLaunch",
                    "cudaStreamSynchronize",
                )
                definitions = [
                    line
                    for line in symbols.splitlines()
                    if line.split() and line.split()[-1] in required
                ]
                if {line.split()[-1] for line in definitions} != set(required):
                    message = "Calibration static CUDA runtime definition proof absent"
                    raise RuntimeError(message)
                runtime_linkage = {
                    "kind": "statically linked CUDA runtime in frozen binary",
                    "defined_symbols": definitions,
                }
        records.append(
            {
                "binary": str(path),
                "sha256": digest(path),
                "ldd": text,
                "resolved": resolved,
                "CUDA_runtime_linkage": runtime_linkage,
            }
        )
    result = {
        "exact": True,
        "environment": env,
        "records": records,
        "scope": "CPU-only ldd proof; no CUDA or profiler workload",
    }
    save(root / "results/provenance/calibration-loader.json", result)
    return result


def calibration_timing(root, started, expires):
    nsys = locate_tool(root, "nsys")
    binary = root / "work/bin/roofline-matched"
    neutral = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    neutral["LD_LIBRARY_PATH"] = str(root / "runtime") + ":" + str(root / "work/lib")
    verify_calibration_loader(root, neutral)
    results = []
    cases = [("compute", "128", "StrictMixed")]
    for mode, parameter, kernel in cases:
        label = "timing-" + mode + "-" + parameter
        target = root / "results/profiles" / label
        target.parent.mkdir(parents=True, exist_ok=True)
        job = run_owned(
            root,
            [
                str(nsys),
                "profile",
                "--trace=cuda,nvtx",
                "--cuda-graph-trace=node",
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--kill=none",
                "--wait=all",
                "--output=" + str(target),
                str(binary),
                mode,
                parameter,
            ],
            label,
            time_budget(started, expires, 90),
            neutral,
        )
        if not job["passed"] or not target.with_suffix(".nsys-rep").is_file():
            msg = "Calibration profiler/owned target did not complete"
            raise RuntimeError(msg)
        applications = []
        for line in (root / "logs" / (label + ".log")).read_text().splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and "all_output_words_verified" in value:
                applications.append(value)
        if len(applications) != 1:
            msg = "Exactly one calibration application result required"
            raise ValueError(msg)
        export = run_owned(
            root,
            [
                str(nsys),
                "export",
                "--type=sqlite",
                "--output=" + str(target.with_suffix(".sqlite")),
                str(target.with_suffix(".nsys-rep")),
            ],
            label + "-export",
            time_budget(started, expires, 60),
            neutral,
        )
        if not export["passed"]:
            msg = "Calibration SQLite export failed"
            raise RuntimeError(msg)
        application = applications[0]
        if application.get("parameter") != int(parameter):
            message = "Calibration requested parameter/output mismatch"
            raise ValueError(message)
        if mode == "memory" and (
            application.get("l2_bytes")
            != results[0]["application_metadata"]["l2_bytes"]
            or application.get("working_set_bytes") != int(parameter) * (1 << 20)
        ):
            message = "Calibration hardware/total working-set output mismatch"
            raise ValueError(message)
        interval = calibration_interval(
            target.with_suffix(".sqlite"), kernel, applications[0]
        )
        if mode == "compute":
            sizing = calibration_sizes(applications[0]["l2_bytes"])
            cases.extend(
                [
                    ("memory", str(sizing["L2_total_working_MiB"]), "CopyVectors"),
                    ("memory", str(sizing["DRAM_total_working_MiB"]), "CopyVectors"),
                ]
            )
        interval.update(
            binary_sha256=digest(binary),
            command=job["command"],
            owned_target_reaped=job["passed"],
        )
        save(target.with_suffix(".interval.json"), interval)
        results.append(interval)
    return results


def run_pilot(root, started, expires, maximum_pairs):
    if maximum_pairs != 2:
        message = "Profile-only diagnostic requires exactly two GPU evaluations"
        raise ValueError(message)
    bootstrap_status = json.loads((root / "bootstrap-status.json").read_text())
    if bootstrap_status.get("phase") != "complete" or bootstrap_status.get(
        "local_cpu_rehearsal"
    ):
        message = "Actual completed bootstrap required before diagnostic GPU runs"
        raise RuntimeError(message)
    preflight(root)
    value = config(root)
    selected_cpu_environment(root)
    cpus = sorted(os.sched_getaffinity(0))
    if not cpus:
        message = "No CPU available"
        raise RuntimeError(message)
    os.sched_setaffinity(0, {cpus[0]})
    hardware_snapshot(root, "profile-only-start")
    state = {
        "phase": "running",
        "diagnostic_mode": "anchor-plus-profile",
        "diagnostic_complete": False,
        "completed_diagnostic_evaluations": [],
        "main_table_eligible": False,
        "new_host_counter_permission": "UNEXECUTED",
        "Ncu_executed": False,
        "physical_roofline_claim": False,
        "prior_canonical_evidence_sha256": value["prior_canonical_evidence_sha256"],
        "scope": "Two actual GPU diagnostics only; prior six-case eighteen-evaluation evidence remains separate",
    }
    save(root / "pilot-status.json", state)
    try:
        state["cpu_clock_witness"] = clock_witness(root)
        seed, track = root / "fixtures/public_p3_seed.json", "P3"
        anchor = evaluate(
            root, seed, track, "candidate", "profile-only-anchor", started, expires
        )
        state["completed_diagnostic_evaluations"].append(
            {
                "label": "anchor",
                "path": str(anchor),
                "profiled": False,
                "gate": json.loads((anchor / "original-global-gate.json").read_text()),
            }
        )
        save(root / "pilot-status.json", state)
        nsys = locate_tool(root, "nsys")
        target = root / "results/profiles/timeline"
        target.parent.mkdir(parents=True, exist_ok=True)
        prefix = [
            str(nsys),
            "profile",
            "--trace=cuda,nvtx",
            "--cuda-graph-trace=node",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--kill=none",
            "--wait=all",
            "--output=" + str(target),
        ]
        profiled = evaluate(
            root,
            seed,
            track,
            "candidate",
            "profile-only-nsys",
            started,
            expires,
            True,
            prefix,
        )
        state["completed_diagnostic_evaluations"].append(
            {
                "label": "profiled",
                "path": str(profiled),
                "profiled": True,
                "gate": json.loads(
                    (profiled / "original-global-gate.json").read_text()
                ),
            }
        )
        raw = raw_compare(root, anchor, profiled, started, expires)
        save(profiled / "raw-profile-output-gate.json", raw)
        state["anchor_profile_raw_comparison"] = raw
        state["nsys_report"] = {
            "created": target.with_suffix(".nsys-rep").is_file(),
            "target_worker_terminal_reaped": True,
        }
        save(root / "pilot-status.json", state)
        if not state["nsys_report"]["created"]:
            message = "Profile-only report missing; preserve raw stream without claiming completion"
            raise RuntimeError(message)
        neutral = {
            key: os.environ[key]
            for key in ("HOME", "PATH", "LANG", "LC_ALL")
            if key in os.environ
        }
        neutral["LD_LIBRARY_PATH"] = (
            str(root / "runtime") + ":" + str(root / "work/lib")
        )
        export = run_owned(
            root,
            [
                str(nsys),
                "export",
                "--type=sqlite",
                "--output=" + str(target.with_suffix(".sqlite")),
                str(target.with_suffix(".nsys-rep")),
            ],
            "profile-only-sqlite-export",
            time_budget(started, expires, 90),
            neutral,
        )
        if not export["passed"]:
            message = "Profile-only SQLite export failed"
            raise RuntimeError(message)
        from tools.optimization.analyze_full_vmec_profile import analyze

        pid = json.loads((profiled / "activation.json").read_text())["pid"]
        analysis = analyze(target.with_suffix(".sqlite"), pid)
        if not analysis.get("kernels"):
            message = "Actual target worker GPU node evidence missing"
            raise RuntimeError(message)
        save(root / "results/profiles/timeline-analysis.json", analysis)
        state["nsys_report"].update(
            sqlite_export_passed=True, actual_worker_kernel_node_evidence=True
        )
        if (
            len(state["completed_diagnostic_evaluations"]) != 2
            or any(
                item["gate"].get("exact") is not True
                for item in state["completed_diagnostic_evaluations"]
            )
            or raw.get("exact") is not True
        ):
            message = "Profile-only actual identity/raw evidence incomplete"
            raise RuntimeError(message)
        state.update(phase="diagnostic_complete", diagnostic_complete=True)
        return 0
    except BaseException as error:
        state.update(phase="failed", error=repr(error))
        return 1
    finally:
        state["owned_children_reaped"] = not descendants()
        runtime = json.loads((root / "bootstrap-status.json").read_text())[
            "runtime_files"
        ]
        state["runtime_unchanged"] = all(
            digest(root / name) == expected for name, expected in runtime.items()
        )
        state["immutable_payload_unchanged"] = all(
            digest(legacy.inside(root, name)) == expected
            for name, expected in json.loads(
                (root / "payload-manifest.json").read_text()
            )["files"].items()
        )
        if (
            not state["runtime_unchanged"]
            or not state["immutable_payload_unchanged"]
            or not state["owned_children_reaped"]
        ):
            state.update(phase="failed_postflight", diagnostic_complete=False)
        save(root / "pilot-status.json", state)
        if state["phase"] == "failed_postflight":
            message = "Profile-only frozen/runtime/ownership postflight failed"
            raise RuntimeError(message)


def verify_collection(archive, expected, destination):
    if digest(archive) != expected:
        message = "Collection SHA mismatch"
        raise ValueError(message)
    destination.mkdir(exist_ok=False)
    legacy.safe_extract(archive, destination)
    manifest = json.loads((destination / "retrieval-manifest.json").read_text())
    actual = {
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual != set(manifest) | {"retrieval-manifest.json"}:
        message = "Uncovered or missing collection files"
        raise ValueError(message)
    for name, expected_hash in manifest.items():
        if digest(legacy.inside(destination, name)) != expected_hash:
            message = "Collection member hash mismatch"
            raise ValueError(message)
    return {"exact": True, "verified_files": len(manifest)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "bootstrap",
            "rehearse-bootstrap",
            "run",
            "worker",
            "profile-worker",
            "import-check",
            "exec",
            "profile-exec",
            "collect",
            "verify-collection",
            "raw-compare",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--lease-started-at", type=float)
    parser.add_argument("--lease-expires-at", type=float)
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--result", type=Path)
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if rest and args.action not in ("worker", "profile-worker", "exec", "profile-exec"):
        parser.error("Unexpected arguments")
    root = args.root.resolve()
    if (
        args.action
        in ("worker", "profile-worker", "import-check", "exec", "profile-exec")
        and args.variant is None
    ):
        parser.error("Worker/import/exec actions require explicit variant")
    if args.action == "preflight":
        print(json.dumps(preflight(root), indent=2))
    elif args.action in ("bootstrap", "run", "rehearse-bootstrap"):
        if not args.execute:
            print(
                json.dumps(
                    {
                        "plan_only": True,
                        "GPU_runs": 0,
                        "root": str(root),
                        "maximum_case_pairs": args.max_pairs,
                        "policy": "<=1h external watchdog; <=1USD/hour owner quote gate; 600s retrieval reserve",
                    }
                )
            )
            return 0
        if args.action == "rehearse-bootstrap":
            local_started = time.time()
            if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
                message = "Cannot establish local rehearsal subreaper"
                raise RuntimeError(message)
            bootstrap(
                root, local_started, local_started + 3600, local_cpu_rehearsal=True
            )
            return 0
        if args.lease_started_at is None or args.lease_expires_at is None:
            parser.error("Actual external lease timestamps required")
        time_budget(args.lease_started_at, args.lease_expires_at, 1)
        if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
            message = "Cannot establish child subreaper"
            raise RuntimeError(message)

        def interrupted(number, _frame):
            message = f"Outer supervisor signal {number}"
            raise InterruptedError(message)

        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        if args.action == "bootstrap":
            bootstrap(root, args.lease_started_at, args.lease_expires_at)
        else:
            return run_pilot(
                root, args.lease_started_at, args.lease_expires_at, args.max_pairs
            )
    elif args.action == "import-check":
        import_check(root, args.variant)
    elif args.action in ("worker", "profile-worker"):
        return worker(root, args.variant, rest, args.action == "profile-worker")
    elif args.action == "exec":
        execute_wrapper(root, args.variant, rest)
    elif args.action == "profile-exec":
        python = root / "work" / args.variant / "bin/python"
        os.execve(
            str(python),
            [str(python), "-B", *rest],
            profile_environment(root, args.variant, os.environ),
        )
    elif args.action == "collect":
        print(json.dumps(legacy.collect(root)))
    elif args.action == "raw-compare":
        if args.left is None or args.right is None or args.result is None:
            parser.error("Raw comparison requires both inputs and fresh result path")
        raw_compare_worker(root, args.left, args.right, args.result)
    else:
        print(
            json.dumps(
                verify_collection(args.archive, args.expected_sha256, args.destination)
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
