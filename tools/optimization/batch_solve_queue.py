"""Opt-in isolated solve-queue experiment; default CPU-only plan, no physics changes."""

# All native futures must drain even after interrupted waits or native failures.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from tools.optimization.batch_throughput import (
    CaptureDispatcher,
    FrozenArtifacts,
    NativeCall,
)
from tools.optimization.native_batch_campaign import raw_gate


def pinned_pool(cpus):
    """Assign each worker one unique core from a larger fixed process mask."""
    assigned = []
    lock = threading.Lock()
    allowed = set(os.sched_getaffinity(0))
    if not set(cpus).issubset(allowed):
        message = "Fixed worker cores exceed the pool creator's allowed affinity"
        raise RuntimeError(message)

    def initialize():
        with lock:
            cpu = cpus[len(assigned)]
            assigned.append(cpu)
            # Lazy native workers may be spawned by a single-core adapter thread.
            # Restore only immutable cores authorized by the main-thread creator.
            os.sched_setaffinity(0, {cpu})
            if os.sched_getaffinity(0) != {cpu}:
                message = "Actual worker affinity does not match its fixed core"
                raise RuntimeError(message)

    def factory(max_workers):
        if max_workers != len(cpus):
            message = "Fixed worker count does not match its CPU core assignment"
            raise ValueError(message)
        return ThreadPoolExecutor(max_workers=max_workers, initializer=initialize)

    return factory


def terminal_result(future):
    """Never abandon native work on an interrupted wait or exit a TLS owner early."""
    interrupted = None
    while True:
        try:
            result = future.result()
        except BaseException as error:
            if not future.done():
                if interrupted is None:
                    interrupted = error
                continue
            raise
        if interrupted is not None:
            raise interrupted
        return result


class QueuedCaptureDispatcher(CaptureDispatcher):
    """Keep adapter lock ownership; native captures belong to fixed solve threads."""

    def __init__(
        self,
        original,
        *,
        native_workers=4,
        native_cpus=None,
        executor_factory=None,
        **options,
    ):
        super().__init__(original, **options)
        self.native_cpus = native_cpus
        factory = executor_factory
        if factory is None:
            factory = (
                pinned_pool(native_cpus)
                if native_cpus is not None
                else ThreadPoolExecutor
            )
        self.executor = factory(max_workers=native_workers)
        self.futures = []
        self.queue_records = []
        self.closed = False
        self.records_lock = threading.Lock()

    def __call__(self, *positional, **keywords):
        state = getattr(self.local, "request", None)
        if (
            state is None
            or state.owner != threading.get_ident()
            or not state.lock_owned
            or state.in_vmec
        ):
            message = "Queued VMEC call requires its adapter-owned request lock"
            raise RuntimeError(message)
        if (
            self.closed
            or keywords.get("max_threads") != 1
            or keywords.get("verbose", True) is not False
        ):
            message = (
                "Queued calls require an open pool, max_threads=1 and verbose=False"
            )
            raise RuntimeError(message)
        self.check_environment()
        input_value = positional[0] if positional else keywords["input"]
        call = NativeCall(copy.deepcopy(input_value), 0)
        submitted = time.perf_counter()
        adapter_tid = threading.get_native_id()
        adapter_affinity = (
            sorted(os.sched_getaffinity(0)) if self.native_cpus is not None else None
        )

        def solve():
            call.started = time.perf_counter()
            cpu_started = time.thread_time()
            call.activity_before = self.activity() if self.activity else None
            native_tid = threading.get_native_id()
            before_affinity = (
                sorted(os.sched_getaffinity(0))
                if self.native_cpus is not None
                else None
            )
            try:
                self.check_environment()
                call.output = self.original(*positional, **keywords)
                return call.output
            finally:
                call.ended = time.perf_counter()
                call.thread_cpu_seconds = time.thread_time() - cpu_started
                call.activity_after = self.activity() if self.activity else None
                after_affinity = (
                    sorted(os.sched_getaffinity(0))
                    if self.native_cpus is not None
                    else None
                )
                with self.records_lock:
                    self.queue_records.append(
                        {
                            "adapter_tid": adapter_tid,
                            "native_tid": native_tid,
                            "adapter_affinity": adapter_affinity,
                            "submitted": submitted,
                            "native_started": call.started,
                            "native_ended": call.ended,
                            "native_queue_seconds": call.started - submitted,
                            "before_affinity": before_affinity,
                            "after_affinity": after_affinity,
                            "activity_before": call.activity_before,
                            "activity_after": call.activity_after,
                        }
                    )
                if self.native_cpus is not None and (
                    before_affinity is None
                    or before_affinity != after_affinity
                    or len(before_affinity) != 1
                    or before_affinity[0] not in self.native_cpus
                ):
                    message = "Actual fixed native-worker affinity changed"
                    raise RuntimeError(message)

        state.in_vmec = True
        state.lock_owned = False
        self.gate.release()
        try:
            future = self.executor.submit(solve)
            with self.records_lock:
                self.futures.append(future)
            return terminal_result(future)
        finally:
            waiting = time.perf_counter()
            self.gate.acquire()
            state.lock_wait_seconds += time.perf_counter() - waiting
            state.lock_owned = True
            state.in_vmec = False
            state.calls.append(call)
            self.check_environment()
            if self.require_gpu and call.output is not None:
                if call.activity_before is None or call.activity_after is None:
                    message = "Native-worker GPU activation counters missing"
                    raise RuntimeError(message)
                delta = [
                    after - before
                    for before, after in zip(
                        call.activity_before, call.activity_after, strict=True
                    )
                ]
                if (
                    len(delta) != 3
                    or min(delta) < 0
                    or delta[0] <= 0
                    or delta[1] <= 0
                    or (self.require_graph and delta[2] <= 0)
                ):
                    message = "Native-worker actual GPU activation proof failed"
                    raise RuntimeError(message)

    def close(self):
        if self.closed:
            return
        self.closed = True
        first_error = None
        for future in self.futures:
            try:
                terminal_result(future)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        # Every submitted native task is terminal before any native worker TLS dies.
        self.executor.shutdown(wait=True, cancel_futures=False)
        if first_error is not None:
            raise first_error


def execute(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    canonical = Path(__file__).with_name("batch_throughput.py")
    copied = out / "batch_driver.py"
    copied.write_bytes(canonical.read_bytes())
    for name in ("campaign_gates.py", "summarize.py"):
        (out / name).write_bytes(canonical.with_name(name).read_bytes())
    spec = importlib.util.spec_from_file_location("queued_batch_driver", copied)
    if spec is None or spec.loader is None:
        message = "Copied batch driver cannot be loaded"
        raise RuntimeError(message)
    module = cast(Any, importlib.util.module_from_spec(spec))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    frozen = FrozenArtifacts(
        [
            Path(__file__),
            canonical,
            copied,
            canonical.with_name("native_batch_campaign.py"),
            out / "campaign_gates.py",
            out / "summarize.py",
        ]
    )
    dispatchers = []

    def factory(original, **options):
        dispatcher = QueuedCaptureDispatcher(
            original, native_workers=4, native_cpus=args.native_cpus, **options
        )
        dispatchers.append(dispatcher)
        return dispatcher

    module.CaptureDispatcher = factory
    original_rounds = module.run_rounds

    def rounds(groups, dispatcher, evaluate, finalize, workers, *, verify_artifacts):
        def verify():
            frozen.check()
            verify_artifacts()

        def audited(state):
            result = finalize(state)
            report = json.loads((state.job.out / "report.json").read_text())
            result["raw_netcdf_hashes_exact"] = raw_gate(state.job.out, report)
            if not result["raw_netcdf_hashes_exact"]:
                message = "Queued candidate raw NetCDF mismatch"
                raise RuntimeError(message)
            return result

        try:
            return original_rounds(
                groups,
                dispatcher,
                evaluate,
                audited,
                workers,
                executor_factory=pinned_pool(args.adapter_cpus),
                verify_artifacts=verify,
            )
        finally:
            # Canonical run_rounds drains every adapter future before returning.
            dispatcher.close()

    module.run_rounds = rounds
    sys.argv = [
        str(copied),
        "--execute",
        "--correctness-only",
        "--backend",
        "gpu",
        "--workers",
        "6",
        "--warmup-rounds",
        "1",
        "--rounds",
        "1",
        "--replicates",
        "1",
        "--manifest",
        str(args.manifest),
        "--references",
        str(args.references),
        "--stella",
        str(args.stella),
        "--out",
        str(out / "evaluation"),
    ]
    code = 1
    try:
        code = module.main()
        return code
    finally:
        (out / "queue-receipt.json").write_text(
            json.dumps(
                {
                    "exit_status": code,
                    "main_table_eligible": False,
                    "scope": "Separate solve-queue correctness prototype, no speed qualification",
                    "native_workers": 4,
                    "adapter_workers": 6,
                    "native_cpus": args.native_cpus,
                    "adapter_cpus": args.adapter_cpus,
                    "all_native_tasks_terminal": all(
                        f.done() for d in dispatchers for f in d.futures
                    ),
                    "all_native_pools_closed": all(d.closed for d in dispatchers),
                    "queue_records": [
                        record for d in dispatchers for record in d.queue_records
                    ],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("manifest", "references", "stella", "out"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--native-cpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--adapter-cpus", type=int, nargs=6, default=[4, 5, 6, 7, 8, 9])
    arguments = parser.parse_args()
    if arguments.execute:
        if any(
            getattr(arguments, name) is None
            for name in ("manifest", "references", "stella", "out")
        ):
            parser.error("Execution requires every isolated path")
        if len(set(arguments.native_cpus + arguments.adapter_cpus)) != 10:
            parser.error("Native and adapter workers require ten distinct CPU cores")
        raise SystemExit(execute(arguments))
    sys.stdout.write(
        json.dumps(
            {
                "plan_only": True,
                "gpu_execution": False,
                "prototype": "six adapter workers, four fixed native workers, unchanged serialization lock",
                "planned_evaluations": 12,
                "main_table_eligible": False,
            }
        )
        + "\n"
    )
