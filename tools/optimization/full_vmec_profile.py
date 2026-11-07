"""Isolated warmed single-main-thread full VMEC Nsight diagnostic."""

# Explicit diagnostic output and standalone staging are intentional.
# ruff: noqa: PLC0415
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

from tools.optimization import batch_nsys_diagnostic as diagnostic
from tools.optimization.native_batch_campaign import raw_gate
from tools.optimization.validate_resident_solver import (
    compare_reports,
    first_difference,
)


def profile_call(function, start, stop, record, push, pop):
    def wrapped(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            msg = "Profiler must run on the OS main thread"
            raise RuntimeError(msg)
        status = start()
        record("start_immediately_before_vmecpp_run", status)
        if status:
            msg = "cudaProfilerStart failed"
            raise RuntimeError(msg)
        push(b"whole vmecpp.run")
        try:
            return function(*args, **kwargs)
        finally:
            pop()
            status = stop()
            record("stop_immediately_after_vmecpp_run", status)
            if status:
                msg = "cudaProfilerStop failed"
                raise RuntimeError(msg)

    return wrapped


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = "Cannot load frozen diagnostic dependency"
        raise RuntimeError(msg)
    module = cast(Any, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


def verify(configuration):
    for name, expected in configuration["frozen"].items():
        if diagnostic.digest(name) != expected:
            raise RuntimeError("Frozen file changed: " + name)


def worker(root):
    configuration = json.loads((root / "configuration.json").read_text())
    state = {
        "pid": os.getpid(),
        "birth": diagnostic.birth(os.getpid()),
        "nonce": os.environ.get("CGA_BATCH_DIAGNOSTIC_NONCE"),
        "phase": "running",
        "events": [],
        "main_table_eligible": False,
    }
    diagnostic.save(root / "worker-status.json", state)
    code = 1
    try:
        verify(configuration)
        if state["nonce"] != configuration["nonce"]:
            msg = "Worker nonce mismatch"
            raise RuntimeError(msg)
        injection = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "NVTX_")) and "INJECTION" in key
        }
        state["injection"] = injection
        if not injection or not all(
            diagnostic.diagnostic_injection_allowed(k, v) for k, v in injection.items()
        ):
            msg = "Only frozen Nsight injection is permitted"
            raise RuntimeError(msg)
        benchmark = load(root / "tools/optimization/benchmark.py", "benchmark")
        import vmecpp

        original = vmecpp.run
        cuda = ctypes.CDLL(configuration["cudart"])
        observer = ctypes.CDLL(str(root / "observer.so"))
        observer.ProfileNvtxPush.argtypes = [ctypes.c_char_p]
        observer.ProfileNvtxPush.restype = None
        observer.ProfileNvtxPop.argtypes = []
        observer.ProfileNvtxPop.restype = None
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
                    "monotonic_ns": time.monotonic_ns(),
                }
            )

        records = []
        for index, label in enumerate(("warmup", "profiled")):
            vmecpp.run = (
                original
                if index == 0
                else profile_call(
                    original,
                    cuda.cudaProfilerStart,
                    cuda.cudaProfilerStop,
                    record,
                    observer.ProfileNvtxPush,
                    observer.ProfileNvtxPop,
                )
            )
            out = root / label
            sys.argv = [
                "benchmark.py",
                "--stella",
                configuration["stella"],
                "--seed",
                configuration["seed"],
                "--track",
                "P3",
                "--out",
                str(out),
            ]
            if benchmark.main() != 0:
                msg = "Actual official evaluation failed"
                raise RuntimeError(msg)
            report = json.loads((out / "report.json").read_text())
            reference = Path(configuration["reference"])
            comparison = compare_reports(
                json.loads((reference / "report.json").read_text()), report
            )
            deck = json.loads((out / "actual_indata_0.json").read_text())
            original_deck = json.loads((reference / "actual_indata_0.json").read_text())
            exact = (
                comparison["exact"]
                and first_difference(deck, original_deck) is None
                and raw_gate(out, report)
            )
            records.append({"label": label, "exact": exact, "comparison": comparison})
            diagnostic.save(root / "parity.json", {"samples": records})
            if not exact:
                msg = "Actual original/raw/deck parity failed"
                raise RuntimeError(msg)
        repeated = compare_reports(
            json.loads((root / "warmup/report.json").read_text()),
            json.loads((root / "profiled/report.json").read_text()),
        )
        if not repeated["exact"] or len(state["events"]) != 2:
            msg = "Repeat/profiler lifecycle gate failed"
            raise RuntimeError(msg)
        verify(configuration)
        diagnostic.save(
            root / "parity.json",
            {"samples": records, "repeat": repeated, "exact": True},
        )
        code = 0
    finally:
        state.update(phase="terminal", exit_status=code, finished_at=time.time())
        diagnostic.save(root / "worker-status.json", state)
    return code


def owned_worker_alive(worker, nonce):
    if not worker:
        return False
    try:
        pid = worker["pid"]
        if diagnostic.birth(pid) != worker["birth"]:
            return False
        if (
            worker.get("nonce") != nonce
            or Path(f"/proc/{pid}").stat().st_uid != os.getuid()
            or (pid, worker["birth"]) not in diagnostic.children(os.getpid())
        ):
            msg = "Worker descendant/birth/UID/nonce ownership failed"
            raise RuntimeError(msg)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        return worker(root)
    configuration = json.loads((root / "configuration.json").read_text())
    verify(configuration)
    diagnostic.ROOT = root
    diagnostic.worker_alive = owned_worker_alive
    status = diagnostic.supervise(
        configuration["command"], configuration["environment"], root, 360
    )
    return int(
        not (
            status.get("cli_exit_status") == 0
            and status.get("owned_children_reaped")
            and status.get("worker", {}).get("exit_status") == 0
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
