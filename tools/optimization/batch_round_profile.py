"""Isolated warmed full-round NVTX scheduling diagnostic; never qualification."""

# Explicit diagnostic CLI and durable terminal receipts are intentional.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

from tools.optimization import batch_nsys_diagnostic as diagnostic
from tools.optimization.native_batch_campaign import pinned_executor, raw_gate

SOURCE = Path("/bulk/vmecpp-optimization-20261103/batch18-qualification")
NSYS = Path("/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys")


def instrument(source):
    source = diagnostic.diagnostic_source(source)
    patches = [
        (
            "    try:\n        for jobs in groups:",
            "    try:\n        for index, jobs in enumerate(groups):",
        ),
        (
            "            started = time.perf_counter()\n            futures = []",
            "            if index == 1:\n                round_profile.start()\n            started = time.perf_counter()\n            futures = []",
        ),
        (
            "            ended = time.perf_counter()\n            round_records",
            "            ended = time.perf_counter()\n            if index == 1:\n                round_profile.stop()\n            round_records",
        ),
        (
            "        executor.shutdown(wait=True, cancel_futures=False)",
            "        round_profile.stop_if_active()\n        executor.shutdown(wait=True, cancel_futures=False)",
        ),
        (
            "        state = RequestState(job, submitted_at, started=time.perf_counter())",
            "        state = RequestState(job, submitted_at, started=time.perf_counter())\n        phase_push('job:' + job.key)",
        ),
        (
            "            state.finished = time.perf_counter()",
            "            state.finished = time.perf_counter()\n            phase_pop()",
        ),
        (
            "            waiting = time.perf_counter()\n            self.gate.acquire()",
            "            phase_push('gatewait:initial')\n            waiting = time.perf_counter()\n            self.gate.acquire()\n            phase_pop()\n            phase_push('setup')",
        ),
        (
            "                self.gate.release()\n            self.local.request = None",
            "                self.gate.release()\n                phase_pop()\n            self.local.request = None",
        ),
        (
            "        self.gate.release()\n        started = time.perf_counter()",
            "        self.gate.release()\n        phase_pop()\n        phase_push('nativecall')\n        started = time.perf_counter()",
        ),
        (
            "            call.ended = time.perf_counter()",
            "            call.ended = time.perf_counter()\n            phase_pop()",
        ),
        (
            "                waiting = time.perf_counter()\n                self.gate.acquire()",
            "                phase_push('gatewait:post-native')\n                waiting = time.perf_counter()\n                self.gate.acquire()\n                phase_pop()\n                phase_push('post-native')",
        ),
    ]
    for before, after in patches:
        if source.count(before) != 1:
            raise ValueError("Reviewed diagnostic source anchor changed: " + before)
        source = source.replace(before, after)
    return source


class RoundProfile:
    def __init__(self, start, stop, record):
        self.start_api, self.stop_api, self.record = start, stop, record
        self.active = False
        self.started = False

    def start(self):
        if threading.current_thread() is not threading.main_thread() or self.started:
            msg = "Exactly one main-thread warmed round is permitted"
            raise RuntimeError(msg)
        status = self.start_api()
        self.record("start_after_warmup_before_submission", status)
        if status:
            msg = "cudaProfilerStart failed"
            raise RuntimeError(msg)
        self.active = self.started = True

    def stop(self):
        if threading.current_thread() is not threading.main_thread() or not self.active:
            msg = "Stop requires the owning active main thread"
            raise RuntimeError(msg)
        self.active = False
        status = self.stop_api()
        self.record("stop_after_futures_before_validation_and_shutdown", status)
        if status:
            msg = "cudaProfilerStop failed"
            raise RuntimeError(msg)

    def stop_if_active(self):
        if self.active:
            self.stop()


def worker(out):
    config = json.loads((out / "configuration.json").read_text())
    status = {
        "pid": os.getpid(),
        "birth": diagnostic.birth(os.getpid()),
        "nonce": os.environ.get("CGA_BATCH_DIAGNOSTIC_NONCE"),
        "phase": "running",
        "events": [],
    }
    diagnostic.save(out / "worker-status.json", status)
    code = 1
    try:
        if status["nonce"] != config["nonce"]:
            msg = "Worker nonce mismatch"
            raise RuntimeError(msg)
        for path, expected in config["frozen"].items():
            if diagnostic.digest(path) != expected:
                msg = "Frozen diagnostic input changed"
                raise RuntimeError(msg)
        injection = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUDA_", "NVTX_")) and "INJECTION" in k
        }
        status["injection_environment"] = injection
        if not injection or not all(
            diagnostic.diagnostic_injection_allowed(k, v) for k, v in injection.items()
        ):
            msg = "Unexpected profiler injection"
            raise RuntimeError(msg)
        spec = importlib.util.spec_from_file_location(
            "round_driver", out / "batch_driver.py"
        )
        if spec is None or spec.loader is None:
            message = "Frozen diagnostic module cannot be loaded"
            raise RuntimeError(message)
        module = cast(Any, importlib.util.module_from_spec(spec))
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cuda = ctypes.CDLL(str(SOURCE / "lib/libcudart.so.12"))
        nvtx = ctypes.CDLL(
            "/usr/local/cuda-12/targets/x86_64-linux/lib/libnvToolsExt.so.1"
        )
        nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        nvtx.nvtxRangePop.argtypes = []

        def record(event, result):
            status["events"].append(
                {
                    "event": event,
                    "status": result,
                    "at": time.time(),
                    "tid": threading.get_native_id(),
                }
            )
            diagnostic.save(out / "worker-status.json", status)

        profile = RoundProfile(cuda.cudaProfilerStart, cuda.cudaProfilerStop, record)
        module.round_profile = profile
        module.phase_push = lambda name: (
            nvtx.nvtxRangePushA(name.encode("ascii")) if profile.active else None
        )
        module.phase_pop = lambda: nvtx.nvtxRangePop() if profile.active else None
        original = module.run_rounds

        def rounds(
            groups, dispatcher, evaluate, finalize, workers, *, verify_artifacts
        ):
            def audited(state):
                result = finalize(state)
                report = json.loads((state.job.out / "report.json").read_text())
                result["raw_netcdf_hashes_exact"] = raw_gate(state.job.out, report)
                if not result["raw_netcdf_hashes_exact"]:
                    msg = "Raw NetCDF mismatch"
                    raise RuntimeError(msg)
                return result

            return original(
                groups,
                dispatcher,
                evaluate,
                audited,
                workers,
                executor_factory=pinned_executor([0, 1, 2, 3]),
                verify_artifacts=verify_artifacts,
            )

        module.run_rounds = rounds
        sys.argv = config["driver_command"]
        code = module.main()
        if len(status["events"]) != 2 or any(e["status"] for e in status["events"]):
            code = 1
    except BaseException as error:
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        status.update(phase="terminal", exit_status=code, finished_at=time.time())
        diagnostic.save(out / "worker-status.json", status)
    return code


def launch(out):
    out.mkdir(parents=True, exist_ok=False)
    canonical = SOURCE / "tools/optimization/batch_throughput.py"
    (out / "batch_driver.py").write_text(instrument(canonical.read_text()))
    for name in ("campaign_gates.py", "summarize.py"):
        (out / name).write_bytes(canonical.with_name(name).read_bytes())
    config = {
        "nonce": secrets.token_hex(16),
        "main_table_eligible": False,
        "scope": "one unprofiled warmup, one full graph-level measured diagnostic round",
    }
    config["driver_command"] = [
        str(out / "batch_driver.py"),
        "--execute",
        "--correctness-only",
        "--backend",
        "gpu",
        "--workers",
        "4",
        "--warmup-rounds",
        "1",
        "--rounds",
        "1",
        "--replicates",
        "1",
        "--manifest",
        str(SOURCE / "fixtures/remote-manifest.json"),
        "--references",
        str(SOURCE / "references"),
        "--stella",
        str(SOURCE / "stella"),
        "--out",
        str(out / "evaluation"),
    ]
    paths = [
        Path(__file__),
        Path(diagnostic.__file__),
        canonical,
        out / "batch_driver.py",
        out / "campaign_gates.py",
        out / "summarize.py",
        NSYS,
        diagnostic.INJECTION,
    ]
    paths += list((SOURCE / "lib").glob("*.so*"))
    original_frozen = json.loads((SOURCE / "run/launch.json").read_text())["files"]
    for path, expected in original_frozen.items():
        if diagnostic.digest(path) != expected:
            msg = "Original qualified campaign input changed"
            raise RuntimeError(msg)
    paths += [Path(path) for path in original_frozen]
    config["frozen"] = {str(p): diagnostic.digest(p) for p in paths}
    diagnostic.save(out / "configuration.json", config)
    environment = {
        k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG", "LC_ALL")
    }
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2])
        + ":"
        + str(SOURCE)
        + ":"
        + str(SOURCE / "python"),
        PYTHONDONTWRITEBYTECODE="1",
        CGA_BATCH_DIAGNOSTIC_NONCE=config["nonce"],
    )
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
        "VMECPP_FORWARD_REFERENCE_SO": str(
            SOURCE / "python/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so"
        ),
        "LD_PRELOAD": str(SOURCE / "lib/gil-release-shim.so")
        + ":"
        + str(SOURCE / "lib/solver-shim.so"),
    }
    command = [
        str(NSYS),
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=graph",
        "--cuda-event-trace=false",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--kill=none",
        "--wait=all",
        "--output=" + str(out / "timeline"),
        "taskset",
        "-c",
        "0,1,2,3",
        "/usr/bin/env",
        *[k + "=" + v for k, v in flags.items()],
        str(SOURCE.parent / "baseline/bin/python"),
        "-B",
        str(Path(__file__)),
        "--worker",
        "--out",
        str(out),
    ]
    diagnostic.save(out / "launch-command.json", command)
    result = diagnostic.supervise(command, environment, out, 420)
    diagnostic.save(out / "supervisor-result.json", result)
    summary = json.loads((out / "evaluation/batch-summary.json").read_text())
    exact = (
        summary["all_original_global_gates_exact"]
        and summary["repeated_outputs_exact"]
        and all(r.get("raw_netcdf_hashes_exact") for r in summary["records"])
    )
    exact = (
        exact
        and result["owned_children_reaped"]
        and all(diagnostic.digest(p) == h for p, h in config["frozen"].items())
    )
    actual_worker = json.loads((out / "worker-status.json").read_text())
    exact = (
        exact
        and result.get("cli_exit_status") == 0
        and actual_worker["phase"] == "terminal"
        and actual_worker["exit_status"] == 0
    )
    diagnostic.save(
        out / "completion.json",
        {"exact": exact, "main_table_eligible": False, "supervisor": result},
    )
    return 0 if exact else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.out))
    if args.execute:
        raise SystemExit(launch(args.out))
    print(
        json.dumps(
            {
                "plan": "isolated warmed 4-worker full-round graph/NVTX diagnostic",
                "main_table_eligible": False,
            }
        )
    )
