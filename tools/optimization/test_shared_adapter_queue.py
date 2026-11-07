"""CPU-only one-core adapter ownership and separate native-pool controls."""

# Standard-library tests exercise real OS-thread affinity without GPU work.
# ruff: noqa: PT027
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from tools.optimization.batch_throughput import Job, RequestState
from tools.optimization.shared_adapter_queue import (
    QUALIFIED_QUEUE_SHA256,
    AdapterRecords,
    observed_adapter_evaluate,
    ordinary_build_receipt,
    prepare_copy,
    role_plan,
    shared_adapter_pool,
)


class SharedAdapterControls(unittest.TestCase):
    def test_cpu_roles_and_invalid_overlap(self):
        plan = role_plan([0, 1, 2, 3], 4, 4, 6)
        assert plan["process_mask_required"] == [0, 1, 2, 3, 4]
        future = role_plan([0, 1, 2, 3, 4, 5], 6, 6, 10)
        assert future["maximum_adapter_threads_needed_per_six_job_round"] == 6
        with self.assertRaises(ValueError):
            role_plan([0, 1, 2, 3], 3, 4, 6)
        with self.assertRaises(ValueError):
            role_plan([0, 1, 2, 2], 4, 4, 6)

    def test_build_receipt_requires_ordinary_coordinated_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "commands.json"
            path.write_text(json.dumps([["g++", "-DVMECPP_COORDINATE_CAPTURES"]]))
            assert ordinary_build_receipt(path)["compile_flags"] == [
                "-DVMECPP_COORDINATE_CAPTURES"
            ]
            path.write_text(json.dumps(["-DVMECPP_COORDINATE_CAPTURES", "-DADAPTIVE"]))
            with self.assertRaises(ValueError):
                ordinary_build_receipt(path)

    def test_real_shared_core_progress_and_native_failure_drain(self):
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 5:
            self.skipTest("Five OS CPUs required for four native roles plus adapter")
        native_cores, adapter_core = available[:4], available[4]
        records = AdapterRecords(adapter_core)
        ready = threading.Barrier(6)
        release = threading.Event()
        four_started = threading.Event()
        native_lock = threading.Lock()
        native_entries = []

        def native(value, **_):
            with native_lock:
                native_entries.append(
                    (threading.get_native_id(), sorted(os.sched_getaffinity(0)))
                )
                if len(native_entries) == 4:
                    four_started.set()
            assert release.wait(5)
            if value == 5:
                message = "native failure"
                raise ValueError(message)
            return value

        with tempfile.TemporaryDirectory() as temporary:
            module = prepare_copy(Path(temporary) / "frozen")
            dispatcher = module.QueuedCaptureDispatcher(
                native,
                native_workers=4,
                native_cpus=native_cores,
                fingerprint=lambda: "fixed",
            )

            def evaluate(value, _):
                ready.wait(timeout=5)
                state = RequestState(
                    Job(
                        str(value),
                        "fixture",
                        "P1",
                        0,
                        False,
                        "{}",
                        Path(temporary) / str(value),
                    ),
                    0,
                )
                with dispatcher.request_scope(state):
                    return dispatcher(value, max_threads=1, verbose=False)

            adapters = shared_adapter_pool(adapter_core, 6, records)(6)
            futures = []
            try:
                operation = observed_adapter_evaluate(evaluate, records)
                futures = [
                    adapters.submit(operation, value, str(value)) for value in range(6)
                ]
                assert four_started.wait(5)
                assert len({tid for tid, _ in native_entries}) == 4
                assert {tuple(mask) for _, mask in native_entries} == {
                    (core,) for core in native_cores
                }
                assert len(records.initializers) == 6
                assert all(
                    row["affinity"] == [adapter_core] for row in records.initializers
                )
            finally:
                release.set()
                errors = []
                for future in futures:
                    try:
                        future.result(timeout=5)
                    except ValueError as error:
                        errors.append(str(error))
                assert all(future.done() for future in futures)
                adapters.shutdown(wait=True)
                with self.assertRaisesRegex(ValueError, "native failure"):
                    dispatcher.close()
            assert errors == ["native failure"]
            assert len(records.requests) == 6
            assert (
                sum(row["error_type"] == "ValueError" for row in records.requests) == 1
            )
            assert all(
                row["after_affinity"] == [adapter_core] for row in records.requests
            )
            assert all(future.done() for future in dispatcher.futures)
        original = Path(__file__).with_name("batch_solve_queue.py")
        assert (
            hashlib.sha256(original.read_bytes()).hexdigest() == QUALIFIED_QUEUE_SHA256
        )
