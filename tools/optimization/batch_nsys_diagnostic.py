"""Explicit diagnostic copies and supervised Nsight batch runs; default plan only."""

# CLI reporting and cleanup of every owned descendant are intentional.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

from tools.optimization.campaign_gates import complete_diagnostics, is_digest
from tools.optimization.validate_resident_solver import (
    compare_reports,
    first_difference,
)

ROOT = Path(__file__).resolve().parents[2]
NSYS = Path("/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys")
INJECTION = NSYS.parent / "libToolsInjection64.so"
INJECTION_SHA256 = "d16d1cbf42e9b9ec8e5750ad1113f513601e20b9a561e54a90c5a8c546518ca3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def diagnostic_injection_allowed(name, value):
    if name not in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH"):
        return False
    try:
        return (
            Path(value).resolve(strict=True) == INJECTION.resolve(strict=True)
            and digest(value) == INJECTION_SHA256
        )
    except (OSError, ValueError):
        return False


def diagnostic_pair_gate(baseline, candidate, baseline_capture, candidate_capture, **_):
    """Numerical diagnostic gate; never changes canonical campaign eligibility."""
    comparison = compare_reports(baseline, candidate)
    decks = (
        bool(baseline_capture.get("actual_indata"))
        and first_difference(
            baseline_capture.get("actual_indata"),
            candidate_capture.get("actual_indata"),
        )
        is None
    )
    numeric = (
        all(
            is_digest(candidate.get(name))
            for name in ("metrics_sha256", "request_metrics_sha256")
        )
        and bool(candidate.get("numerical_wout_hashes"))
        and complete_diagnostics(candidate)
    )
    return {
        "exact": bool(
            comparison["exact"]
            and decks
            and numeric
            and candidate.get("profiled") is True
        ),
        "original_report_comparison": comparison,
        "actual_indata_exact": decks,
        "external_profiled": True,
        "main_table_eligible": False,
    }


def diagnostic_source(original):
    replacements = {
        "from tools.optimization.summarize import metric_drift": "from tools.optimization.summarize import metric_drift\nfrom tools.optimization.batch_nsys_diagnostic import diagnostic_injection_allowed, diagnostic_pair_gate\npair_gate = diagnostic_pair_gate",
        'or name in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH")': 'or (name in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH") and not diagnostic_injection_allowed(name, value))',
        '"profiled": False,': '"profiled": True,',
        '"measurement": "measured batch experiment",': '"measurement": "measured external Nsight diagnostic; not performance qualification",',
        '"measurement": "measured batch correctness/throughput experiment",': '"measurement": "measured external Nsight diagnostic; performance unqualified",',
        "    result = execute(args, cases)": "    args.correctness_only = True\n    result = execute(args, cases)",
    }
    for before, after in replacements.items():
        if original.count(before) != 1:
            message = "Canonical driver changed; review the exact diagnostic patch"
            raise RuntimeError(message)
        original = original.replace(before, after)
    return original


def profile_lifecycle(function, start, stop, record):
    """Only main thread calls profiler APIs, outside the entire worker lifetime."""

    def run(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            message = "Profiler lifecycle must be owned by the main thread"
            raise RuntimeError(message)
        status = start()
        record("start_before_pool", status)
        if status != 0:
            message = "cudaProfilerStart failed before worker construction"
            raise RuntimeError(message)
        try:
            return function(*args, **kwargs)
        finally:
            # The canonical run_rounds drains all futures and shuts down its pool
            # even when a request or interrupted wait fails.
            record("stop_after_drained_pool", stop())

    return run


def birth(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def children(parent):
    result = []
    pending = [parent]
    seen = {parent}
    while pending:
        current = pending.pop()
        for task in Path(f"/proc/{current}/task").glob("*/children"):
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                for value in task.read_text().split():
                    pid = int(value)
                    if pid in seen:
                        continue
                    with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                        result.append((pid, birth(pid)))
                        pending.append(pid)
                        seen.add(pid)
    return result


def reap_adopted(excluded_pid):
    for pid, _ in children(os.getpid()):
        if pid != excluded_pid:
            with contextlib.suppress(ChildProcessError, ProcessLookupError):
                os.waitpid(pid, os.WNOHANG)


def cleanup_owned(excluded_pid):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, expected_birth in reversed(children(os.getpid())):
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                if birth(pid) == expected_birth:
                    os.kill(pid, sig)
        limit = time.monotonic() + 3
        while time.monotonic() < limit and children(os.getpid()):
            reap_adopted(excluded_pid)
            time.sleep(0.05)
    return not children(os.getpid())


def worker_alive(worker, nonce):
    if not worker or not nonce:
        return False
    pid = worker["pid"]
    try:
        if birth(pid) != worker["birth"]:
            return False
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        if (
            worker.get("nonce") != nonce
            or ("CGA_BATCH_DIAGNOSTIC_NONCE=" + nonce).encode() not in environment
            or Path(f"/proc/{pid}").stat().st_uid != os.getuid()
        ):
            message = "Live worker ownership/nonce proof failed"
            raise RuntimeError(message)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False


def cleanup_worker(worker, nonce):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not worker_alive(worker, nonce):
            break
        descendants = children(worker["pid"])
        for pid, expected_birth in [
            *reversed(descendants),
            (worker["pid"], worker["birth"]),
        ]:
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                if birth(pid) == expected_birth:
                    os.kill(pid, sig)
        limit = time.monotonic() + 3
        while time.monotonic() < limit and worker_alive(worker, nonce):
            reap_adopted(-1)
            time.sleep(0.05)
    return not worker_alive(worker, nonce)


def supervise(command, environment, out, timeout):
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        message = "Cannot establish owned-child subreaper"
        raise RuntimeError(message)
    status = {
        "phase": "running",
        "supervisor_pid": os.getpid(),
        "started_at": time.time(),
        "deadline_seconds": timeout,
        "command": command,
    }
    nonce = environment.get("CGA_BATCH_DIAGNOSTIC_NONCE")
    process = None
    try:
        with (out / "nsys.log").open("x") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status["cli_pid"] = process.pid
            status["cli_birth"] = birth(process.pid)
            started = time.monotonic()
            while True:
                code = process.poll()
                if code is not None and "cli_exit_status" not in status:
                    status.update(cli_exit_status=code, cli_finished_at=time.time())
                reap_adopted(process.pid)
                worker_path = out / "worker-status.json"
                worker = (
                    json.loads(worker_path.read_text())
                    if worker_path.exists()
                    else None
                )
                remaining = children(os.getpid())
                live_worker = worker_alive(worker, nonce)
                status.update(
                    worker=worker,
                    live_owned_descendants=remaining,
                    matching_birth_worker_live=live_worker,
                )
                save(out / "supervisor-status.json", status)
                if code is not None and not remaining and not live_worker:
                    break
                if time.monotonic() - started > timeout:
                    status["deadline_exceeded"] = True
                    message = (
                        "Independent worker deadline exceeded, including after CLI exit"
                    )
                    raise TimeoutError(message)
                time.sleep(0.2)
    except BaseException as error:
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if process is not None:
            worker_clean = cleanup_worker(status.get("worker"), nonce)
            if process.poll() is not None:
                reap_adopted(process.pid)
            clean = not children(os.getpid()) or cleanup_owned(process.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            reap_adopted(-1)
            clean = not children(os.getpid()) and worker_clean
        else:
            clean = not children(os.getpid())
        status.update(
            phase="terminal" if clean else "cleanup_unconfirmed",
            owned_children_reaped=clean,
            finished_at=time.time(),
        )
        save(out / "supervisor-status.json", status)
    return status


def worker(out):
    configuration = json.loads((out / "configuration.json").read_text())
    state = {
        "pid": os.getpid(),
        "birth": birth(os.getpid()),
        "phase": "running",
        "events": [],
        "external_profiled": True,
        "nonce": os.environ.get("CGA_BATCH_DIAGNOSTIC_NONCE"),
    }
    save(out / "worker-status.json", state)
    code = 1
    try:
        if state["nonce"] != configuration["nonce"]:
            message = "Worker nonce does not match frozen launch configuration"
            raise RuntimeError(message)
        for name, expected in configuration["frozen"].items():
            if digest(name) != expected:
                message = "Frozen diagnostic input changed"
                raise RuntimeError(message)
        injected = {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("CUDA_", "NVTX_")) and "INJECTION" in name
        }
        state["injection_environment"] = injected
        if not injected or not all(
            diagnostic_injection_allowed(name, value)
            for name, value in injected.items()
        ):
            message = "Nsight did not supply only the exact frozen injection library"
            raise RuntimeError(message)
        spec = importlib.util.spec_from_file_location(
            "frozen_batch_diagnostic", out / "frozen/batch_driver.py"
        )
        if spec is None or spec.loader is None:
            message = "Cannot load frozen diagnostic driver"
            raise RuntimeError(message)
        module = cast(Any, importlib.util.module_from_spec(spec))
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cuda = ctypes.CDLL("/usr/local/cuda-12.9/lib64/libcudart.so.12")
        for name in ("cudaProfilerStart", "cudaProfilerStop"):
            function = getattr(cuda, name)
            function.argtypes = []
            function.restype = ctypes.c_int

        def record(event, status):
            state["events"].append(
                {
                    "event": event,
                    "status": status,
                    "tid": threading.get_native_id(),
                    "at": time.time(),
                }
            )
            save(out / "worker-status.json", state)

        module.run_rounds = profile_lifecycle(
            module.run_rounds, cuda.cudaProfilerStart, cuda.cudaProfilerStop, record
        )
        sys.argv = configuration["driver_command"]
        code = module.main()
        if len(state["events"]) != 2 or any(
            event["status"] != 0 for event in state["events"]
        ):
            code = 1
    except BaseException as error:
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        state.update(phase="terminal", exit_status=code, finished_at=time.time())
        save(out / "worker-status.json", state)
    return code


def execute(args):
    out = args.out.resolve()
    if out.parent != ROOT / "build/optimization/validation-agent/batch-diagnostic":
        message = "Diagnostic outputs require a fresh isolated batch-diagnostic child"
        raise ValueError(message)
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "frozen"
    frozen.mkdir()
    original = ROOT / "tools/optimization/batch_throughput.py"
    copied = frozen / "batch_driver.py"
    copied.write_text(diagnostic_source(original.read_text()))
    paths = [original, copied, Path(__file__).resolve(), NSYS, INJECTION]
    for name in ("campaign_gates.py", "summarize.py"):
        source = ROOT / "tools/optimization" / name
        (frozen / name).write_bytes(source.read_bytes())
        paths.extend([source, frozen / name])
    opt = ROOT / "build/optimization"
    gil = opt / "validation-agent/gil-release-prototype/build4/gil-release-shim.so"
    solver = opt / "resident-solver18-lattice-coordinated/solver-shim.so"
    reference = (
        opt
        / "gpu-env/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
    )
    paths.extend([gil, solver, solver.with_name("solver-backend.so"), reference])
    if digest(INJECTION) != INJECTION_SHA256:
        message = "Frozen Nsight injection library mismatch"
        raise RuntimeError(message)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    minimum = 4096 + 2048 * (args.workers - 1)
    free = int(query.stdout.strip().split(",")[3].strip())
    save(
        out / "memory-preflight.json",
        {
            "actual_query": query.stdout,
            "minimum_free_mib_estimated": minimum,
            "shared_load_performance_unqualified": True,
        },
    )
    if free < minimum:
        message = "Insufficient estimated memory headroom; no GPU job launched"
        raise RuntimeError(message)
    driver_command = [
        str(copied),
        "--execute",
        "--correctness-only",
        "--backend",
        "gpu",
        "--workers",
        str(args.workers),
        "--rounds",
        "2",
        "--warmup-rounds",
        "0",
        "--replicates",
        str(2 if args.workers == 4 else 1),
        "--case",
        "public_p3_seed.json:P1",
        "--case",
        "public_p2_seed.json:P3",
        "--manifest",
        str(opt / "fixtures-frozen/manifest.json"),
        "--references",
        str(opt / "validation-agent/cloud-pilot8-bundle/references"),
        "--stella",
        "/mnt/f/Users/might/Sync/stella",
        "--out",
        str(out / "evaluation"),
    ]
    configuration = {
        "nonce": secrets.token_hex(16),
        "external_profiled": True,
        "main_table_eligible": False,
        "canonical_source_sha256": digest(original),
        "diagnostic_source_sha256": digest(copied),
        "driver_command": driver_command,
        "frozen": {str(path): digest(path) for path in paths},
    }
    save(out / "configuration.json", configuration)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in (
            "HOME",
            "PATH",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "WSL_DISTRO_NAME",
            "WSL_INTEROP",
        )
    }
    environment.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    environment["CGA_BATCH_DIAGNOSTIC_NONCE"] = configuration["nonce"]
    flags = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VMECPP_RELEASE_GIL": "1",
        "VMECPP_GPU_SOLVER": "1",
        "VMECPP_GPU_SOLVER_RESIDENT_CONTROL": "1",
        "VMECPP_GPU_RESIDUAL_ROUNDS": "32",
        "VMECPP_GPU_RESIDUAL_GRAPH": "1",
        "VMECPP_GPU_DIAGNOSTICS_ROUNDS": "16",
        "VMECPP_GPU_DIAGNOSTICS_GRAPH": "1",
        "VMECPP_GPU_SOLVER_GRAPH": "1",
        "VMECPP_GPU_SOLVER_PINNED_SCALARS": "1",
        "VMECPP_FORWARD_REFERENCE_SO": str(reference),
        "LD_PRELOAD": str(gil) + ":" + str(solver),
    }
    command = [
        str(NSYS),
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--cuda-event-trace=false",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--duration=" + str(args.window),
        "--kill=none",
        "--wait=all",
        "--force-overwrite=false",
        "--output=" + str(out / "timeline"),
        "/usr/bin/env",
        *[key + "=" + value for key, value in flags.items()],
        str(opt / "gpu-env/bin/python"),
        "-B",
        "-m",
        "tools.optimization.batch_nsys_diagnostic",
        "--worker",
        "--out",
        str(out),
    ]
    status = supervise(command, environment, out, args.timeout)
    actual_worker = json.loads((out / "worker-status.json").read_text())
    summary = json.loads((out / "evaluation/batch-summary.json").read_text())
    exact = (
        status.get("cli_exit_status") == 0
        and status["owned_children_reaped"]
        and actual_worker["phase"] == "terminal"
        and actual_worker["exit_status"] == 0
        and summary["all_original_global_gates_exact"]
        and summary["repeated_outputs_exact"]
        and all(digest(name) == sha for name, sha in configuration["frozen"].items())
    )
    save(
        out / "completion-gate.json",
        {
            "exact": bool(exact),
            "external_profiled": True,
            "main_table_eligible": False,
            "worker": actual_worker,
            "supervisor": status,
            "frozen_unchanged": all(
                digest(name) == sha for name, sha in configuration["frozen"].items()
            ),
            "cli_finished_before_worker_seconds": actual_worker["finished_at"]
            - status.get("cli_finished_at", actual_worker["finished_at"]),
            "all_original_global_gates_exact": summary[
                "all_original_global_gates_exact"
            ],
            "repeated_outputs_exact": summary["repeated_outputs_exact"],
        },
    )
    if exact and (out / "timeline.nsys-rep").is_file():
        with (out / "export.log").open("x") as log:
            export = subprocess.run(
                [
                    str(NSYS),
                    "export",
                    "--type=sqlite",
                    "--output=" + str(out / "timeline.sqlite"),
                    str(out / "timeline.nsys-rep"),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=60,
            )
        save(out / "export-status.json", {"exit_status": export.returncode})
    return 0 if exact else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.worker:
        return worker(args.out.resolve())
    if not 12 <= args.window <= 30 or not 120 <= args.timeout <= 900:
        parser.error("Diagnostic capture/deadline outside bounded limits")
    if not args.execute:
        print(
            json.dumps(
                {
                    "scope": "PLAN ONLY; no GPU/profiler launch or output writes",
                    "workers": args.workers,
                    "rounds": 2,
                    "replicates_per_case": 2 if args.workers == 4 else 1,
                    "node_window_seconds": args.window,
                    "independent_worker_deadline_seconds": args.timeout,
                    "main_table_eligible": False,
                },
                indent=2,
            )
        )
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
