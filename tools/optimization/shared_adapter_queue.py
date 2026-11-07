"""CPU plan and opt-in copied solve-queue driver with one shared adapter core."""

# Interrupted adapters/native tasks must all drain before either pool exits.
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from tools.optimization.batch_throughput import FrozenArtifacts

QUALIFIED_QUEUE_SHA256 = (
    "681da899dbb93cda5e981fde5a5281a4fb035615657c2245c06aa76b09215486"
)


def role_plan(native_cpus, adapter_cpu, native_workers, adapter_workers):
    if native_workers not in (4, 6) or adapter_workers not in (6, 10):
        message = "Prototype permits four/six native and six/ten adapter workers"
        raise ValueError(message)
    if len(native_cpus) != native_workers or len(set(native_cpus)) != native_workers:
        message = "Every native worker requires its own distinct CPU"
        raise ValueError(message)
    if adapter_cpu in native_cpus or min([*native_cpus, adapter_cpu]) < 0:
        message = "The one adapter core must be nonnegative and disjoint"
        raise ValueError(message)
    return {
        "native_workers": native_workers,
        "native_cpus": list(native_cpus),
        "adapter_workers_configured": adapter_workers,
        "adapter_cpu_shared_by_all_workers": adapter_cpu,
        "process_mask_required": sorted([*native_cpus, adapter_cpu]),
        "dedicated_role_cpu_count": native_workers + 1,
        "planned_evaluations": 12,
        "maximum_adapter_threads_needed_per_six_job_round": min(adapter_workers, 6),
        "auxiliary_thread_scope": "Library threads may inherit the full process mask or their creator's single core; snapshots record observed masks",
        "physical_core_scope": "Verify physical topology on native Linux; WSL guest affinity alone does not prove host physical placement",
        "performance_qualification": False,
        "combined_best_runtime_correctness": "Not executed or established by CPU tests",
    }


class AdapterRecords:
    def __init__(self, core):
        self.core = core
        self.lock = threading.Lock()
        self.initializers = []
        self.requests = []
        self.first_thread_snapshot = None

    @staticmethod
    def thread_snapshot():
        rows = []
        for path in Path("/proc/self/task").glob("*/status"):
            try:
                values = dict(
                    line.split(":", 1)
                    for line in path.read_text().splitlines()
                    if ":" in line
                )
                rows.append(
                    {
                        "tid": int(path.parent.name),
                        "name": values.get("Name", "").strip(),
                        "cpu_allowed_list": values.get("Cpus_allowed_list", "").strip(),
                    }
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
                rows.append({"snapshot_error": type(error).__name__})
        return rows


def shared_adapter_pool(core, workers, records):
    """Several adapter threads share a core; the native executor stays separate."""
    if core not in os.sched_getaffinity(0):
        message = "Shared adapter core is outside the creating thread's mask"
        raise RuntimeError(message)

    def initialize():
        os.sched_setaffinity(0, {core})
        actual = sorted(os.sched_getaffinity(0))
        if actual != [core]:
            message = "Adapter thread did not acquire its single fixed core"
            raise RuntimeError(message)
        with records.lock:
            records.initializers.append(
                {
                    "role": "adapter",
                    "tid": threading.get_native_id(),
                    "affinity": actual,
                }
            )

    def factory(max_workers):
        if max_workers != workers:
            message = "Adapter executor count differs from the frozen role plan"
            raise ValueError(message)
        return ThreadPoolExecutor(max_workers=max_workers, initializer=initialize)

    return factory


def observed_adapter_evaluate(evaluate, records):
    def observed(request, output):
        affinity = sorted(os.sched_getaffinity(0))
        started, cpu_started = time.perf_counter(), time.thread_time()
        with records.lock:
            if records.first_thread_snapshot is None:
                records.first_thread_snapshot = records.thread_snapshot()
        error_type = None
        try:
            return evaluate(request, output)
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            after = sorted(os.sched_getaffinity(0))
            with records.lock:
                records.requests.append(
                    {
                        "role": "adapter",
                        "tid": threading.get_native_id(),
                        "output": str(output),
                        "before_affinity": affinity,
                        "after_affinity": after,
                        "adapter_thread_cpu_seconds": time.thread_time() - cpu_started,
                        "adapter_request_wall_seconds": time.perf_counter() - started,
                        "error_type": error_type,
                    }
                )
            if affinity != [records.core] or after != affinity:
                message = "Actual adapter request left its one shared core"
                raise RuntimeError(message)

    return observed


def copied_source(original):
    patches = [
        (
            "original, native_workers=4, native_cpus=args.native_cpus, **options",
            "original, native_workers=args.native_workers, native_cpus=args.native_cpus, **options",
        ),
        (
            '"--workers",\n        "6",',
            '"--workers",\n        str(args.adapter_workers),',
        ),
        (
            "executor_factory=pinned_pool(args.adapter_cpus),",
            "executor_factory=shared_adapter_pool(args.adapter_cpu, args.adapter_workers, args.adapter_records),",
        ),
        (
            "                evaluate,\n                audited,",
            "                observed_adapter_evaluate(evaluate, args.adapter_records),\n                audited,",
        ),
        ('"native_workers": 4,', '"native_workers": args.native_workers,'),
        ('"adapter_workers": 6,', '"adapter_workers": args.adapter_workers,'),
    ]
    result = original
    for before, after in patches:
        if result.count(before) != 1:
            message = "Qualified queue source changed; review exact copy patch"
            raise RuntimeError(message)
        result = result.replace(before, after)
    before = next(
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.ClassDef) and node.name == "QueuedCaptureDispatcher"
    )
    after = next(
        node
        for node in ast.parse(result).body
        if isinstance(node, ast.ClassDef) and node.name == "QueuedCaptureDispatcher"
    )
    if ast.dump(before) != ast.dump(after):
        message = "Native capture/ownership class must remain unchanged"
        raise RuntimeError(message)
    return result


def ordinary_build_receipt(path):
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)

    tokens = list(strings(json.loads(path.read_text())))
    flags = [token for token in tokens if token.startswith("-D")]
    if "-DVMECPP_COORDINATE_CAPTURES" not in flags or any(
        "CONDITIONAL" in flag.upper() or "ADAPTIVE" in flag.upper() for flag in flags
    ):
        message = "Require a reviewed capture-coordinated ordinary build receipt"
        raise ValueError(message)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "compile_flags": flags,
        "scope": "Build receipt plus existing live GIL/backend ownership guard required; not a claim of combined-runtime GPU correctness",
    }


def topology_snapshot(cpus):
    rows = []
    for cpu in cpus:
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        rows.append(
            {
                "logical_cpu": cpu,
                "physical_package_id": int(
                    (directory / "physical_package_id").read_text()
                ),
                "core_id": int((directory / "core_id").read_text()),
                "thread_siblings_list": (directory / "thread_siblings_list")
                .read_text()
                .strip(),
            }
        )
    identities = {(row["physical_package_id"], row["core_id"]) for row in rows}
    if len(identities) != len(cpus):
        message = "Role cores share reported physical-core identities"
        raise ValueError(message)
    return rows


def prepare_copy(directory):
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("batch_solve_queue.py")
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != QUALIFIED_QUEUE_SHA256:
        message = "Expected the preserved qualified queue source; refusing drift"
        raise RuntimeError(message)
    target = directory / "shared_queue_driver.py"
    target.write_text(copied_source(original.decode()))
    for name in (
        "batch_throughput.py",
        "campaign_gates.py",
        "summarize.py",
        "native_batch_campaign.py",
    ):
        (directory / name).write_bytes(source.with_name(name).read_bytes())
    spec = importlib.util.spec_from_file_location("shared_adapter_queue_copy", target)
    if spec is None or spec.loader is None:
        message = "Cannot load isolated queue copy"
        raise RuntimeError(message)
    module = cast(Any, importlib.util.module_from_spec(spec))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.shared_adapter_pool = shared_adapter_pool
    module.observed_adapter_evaluate = observed_adapter_evaluate
    return module


def execute(args, plan):
    if sorted(os.sched_getaffinity(0)) != plan["process_mask_required"]:
        message = "Launch the fresh process with exactly its planned CPU mask"
        raise RuntimeError(message)
    build = ordinary_build_receipt(args.runtime_build_commands)
    plan["reported_cpu_topology"] = topology_snapshot(plan["process_mask_required"])
    initial_threads = AdapterRecords.thread_snapshot()
    staging = args.out.with_name(args.out.name + "-frozen")
    module = prepare_copy(staging)
    args.adapter_records = AdapterRecords(args.adapter_cpu)
    args.adapter_cpus = [args.adapter_cpu] * args.adapter_workers
    frozen = FrozenArtifacts(
        [
            Path(__file__),
            args.runtime_build_commands,
            *[p for p in staging.iterdir() if p.is_file()],
        ]
    )
    code = 1
    try:
        code = module.execute(args)
        return code
    finally:
        frozen.check()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "shared-adapter-receipt.json").write_text(
            json.dumps(
                {
                    "exit_status": code,
                    "main_table_eligible": False,
                    "role_plan": plan,
                    "ordinary_build_receipt": build,
                    "scope": "New scheduling/runtime combination: correctness gate only, no throughput estimate or qualification",
                    "adapter_initializers": args.adapter_records.initializers,
                    "threads_before_worker_construction": initial_threads,
                    "adapter_requests": args.adapter_records.requests,
                    "aux_threads_before_first_native_call": args.adapter_records.first_thread_snapshot,
                    "remaining_threads_after_both_pools_drained": AdapterRecords.thread_snapshot(),
                    "source_freeze_unchanged": True,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--native-workers", type=int, choices=(4, 6), default=4)
    parser.add_argument("--adapter-workers", type=int, choices=(6, 10), default=6)
    parser.add_argument("--native-cpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--adapter-cpu", type=int, default=4)
    for name in ("manifest", "references", "stella", "out", "runtime-build-commands"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    plan = role_plan(
        args.native_cpus, args.adapter_cpu, args.native_workers, args.adapter_workers
    )
    if not args.execute:
        sys.stdout.write(
            json.dumps(
                {"plan_only": True, "GPU_executed": False, "roles": plan}, indent=2
            )
            + "\n"
        )
        return 0
    if any(
        getattr(args, name) is None
        for name in (
            "manifest",
            "references",
            "stella",
            "out",
            "runtime_build_commands",
        )
    ):
        parser.error(
            "Execution requires fresh isolated paths and ordinary build receipt"
        )
    args.out = args.out.resolve()
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
