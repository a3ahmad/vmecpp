"""CPU-only ownership/draining tests; fake solves do not qualify physics/timings."""

import contextlib
import hashlib
import io
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import numpy as np

from tools.optimization import batch_throughput as batch


@contextlib.contextmanager
def expect_error(error_type):
    try:
        yield
    except error_type:
        return
    message = "Expected the batch safety gate to reject this operation"
    raise AssertionError(message)


def job(root, number, round_number=0):
    return batch.Job(
        str(number),
        "seed.json",
        "P1",
        round_number,
        False,
        json.dumps({"value": number}),
        root / str(number),
    )


class RecordingExecutor(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.futures = []
        self.terminal_before_shutdown = None

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        self.futures.append(future)
        return future

    def shutdown(self, *args, **kwargs):
        self.terminal_before_shutdown = all(future.done() for future in self.futures)
        return super().shutdown(*args, **kwargs)


class BatchThroughputTest(unittest.TestCase):
    def test_interrupted_pending_future_retries_before_pool_shutdown(self):
        class PendingFuture:
            calls = 0
            terminal = False

            def result(self):
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt
                self.terminal = True

            def done(self):
                return self.terminal

        future = PendingFuture()
        shutdown = []

        class Executor:
            def __init__(self, **_kwargs):
                pass

            def submit(self, *_args):
                return future

            def shutdown(self, **_kwargs):
                assert future.done()
                shutdown.append(True)

        with tempfile.TemporaryDirectory() as temporary:
            dispatcher = batch.CaptureDispatcher(lambda *_args: None)
            with (
                mock.patch.object(batch, "as_completed", side_effect=KeyboardInterrupt),
                expect_error(KeyboardInterrupt),
            ):
                batch.run_rounds(
                    [[job(Path(temporary), 1)]],
                    dispatcher,
                    lambda *_args: None,
                    lambda *_args: None,
                    1,
                    executor_factory=Executor,
                )
        assert future.calls == 2
        assert shutdown == [True]

    def test_gpu_label_requires_actual_creates_runs_and_requested_graph_launches(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, increment in enumerate(((1, 1, 1), (0, 0, 0), (1, 1, 0))):
                counts = [0, 0, 0]

                def original(value, increment=increment, counts=counts, **_kwargs):
                    for i, delta in enumerate(increment):
                        counts[i] += delta
                    return value

                dispatcher = batch.CaptureDispatcher(
                    original,
                    activity=lambda counts=counts: tuple(counts),
                    require_gpu=True,
                    require_graph=True,
                )
                state = batch.RequestState(job(Path(temporary), index), 0)
                with dispatcher.request_scope(state):
                    if index == 0:
                        dispatcher({}, max_threads=1, verbose=False)
                        assert batch.timing_record(state)[
                            "backend_activity_deltas"
                        ] == [[1, 1, 1]]
                    else:
                        with expect_error(RuntimeError):
                            dispatcher({}, max_threads=1, verbose=False)
                        assert state.lock_owned

    def test_activity_read_failure_restores_lock_and_frozen_file_drift_fails(self):
        calls = []

        def activity():
            calls.append(1)
            if len(calls) > 1:
                message = "synthetic counter read failure"
                raise RuntimeError(message)
            return (0, 0, 0)

        dispatcher = batch.CaptureDispatcher(
            lambda value, **_kwargs: value, activity=activity, require_gpu=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = batch.RequestState(job(root, 1), 0)
            with dispatcher.request_scope(state):
                with expect_error(RuntimeError):
                    dispatcher({}, max_threads=1, verbose=False)
                assert state.lock_owned
                assert not state.in_vmec
            artifact = root / "fixture.json"
            artifact.write_text("original")
            frozen = batch.FrozenArtifacts([artifact])
            frozen.check()
            artifact.write_text("changed")
            with expect_error(RuntimeError):
                frozen.check()

    def test_setup_and_postprocess_serialize_while_owned_solves_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counts = {"setup": 0, "native": 0, "post": 0}
            maximum = dict(counts)
            lock = threading.Lock()

            @contextlib.contextmanager
            def phase(name):
                with lock:
                    counts[name] += 1
                    maximum[name] = max(maximum[name], counts[name])
                try:
                    yield
                finally:
                    with lock:
                        counts[name] -= 1

            def original(value, **_kwargs):
                assert not cast(Any, dispatcher.gate)._is_owned()
                with phase("native"):
                    time.sleep(0.06)
                return {"value": value["value"]}

            dispatcher = batch.CaptureDispatcher(original)

            def evaluate(request, _out):
                assert cast(Any, dispatcher.gate)._is_owned()
                with phase("setup"):
                    time.sleep(0.01)
                output = dispatcher(request, max_threads=1, verbose=False)
                assert cast(Any, dispatcher.gate)._is_owned()
                with phase("post"):
                    time.sleep(0.01)
                return output

            def finalize(state):
                assert not state.lock_owned
                assert state.calls[0].input_snapshot["value"] == int(state.job.key)
                assert state.calls[0].output["value"] == int(state.job.key)
                return {"job": state.job.key, "gate": {"exact": state.error is None}}

            rounds, records = batch.run_rounds(
                [[job(root, 1), job(root, 2)]], dispatcher, evaluate, finalize, 2
            )
            assert len(records) == 2
            assert rounds[0]["all_gates_exact"]
            assert maximum == {"setup": 1, "native": 2, "post": 1}

    def test_failure_drains_other_requests_before_tls_pool_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = threading.Event()
            executors = []

            def factory(**kwargs):
                executor = RecordingExecutor(**kwargs)
                executors.append(executor)
                return executor

            def original(value, **_kwargs):
                time.sleep(0.03)
                completed.set()
                return value

            dispatcher = batch.CaptureDispatcher(original)

            def evaluate(request, _out):
                if request["value"] == 1:
                    message = "synthetic request failure"
                    raise RuntimeError(message)
                dispatcher(request, max_threads=1, verbose=False)
                return request

            rounds, records = batch.run_rounds(
                [[job(root, 1), job(root, 2)], [job(root, 3, 1)]],
                dispatcher,
                evaluate,
                lambda state: {
                    "job": state.job.key,
                    "gate": {"exact": state.error is None},
                },
                2,
                executor_factory=factory,
            )
            assert completed.is_set()
            assert executors[0].terminal_before_shutdown
            assert len(records) == 2
            assert len(rounds) == 1
            assert rounds[0]["throughput_evaluations_per_second"] is None

    def test_unowned_reentrant_and_nonquiet_calls_fail_closed(self):
        dispatcher = batch.CaptureDispatcher(lambda value, **_kwargs: value)
        with expect_error(RuntimeError):
            dispatcher({}, max_threads=1, verbose=False)
        with tempfile.TemporaryDirectory() as temporary:
            state = batch.RequestState(job(Path(temporary), 1), 0)
            with dispatcher.request_scope(state):
                with expect_error(RuntimeError):
                    dispatcher({}, max_threads=2, verbose=False)
                with expect_error(RuntimeError):
                    dispatcher({}, max_threads=1, verbose=True)
                with expect_error(RuntimeError), dispatcher.request_scope(state):
                    pass
                assert dispatcher({}, max_threads=1, verbose=False) == {}
            assert getattr(dispatcher.local, "request", None) is None

    def test_recursive_native_call_reacquires_gate_before_error_propagates(self):
        def original(value, **kwargs):
            return dispatcher(value, **kwargs)

        dispatcher = batch.CaptureDispatcher(original)
        with tempfile.TemporaryDirectory() as temporary:
            state = batch.RequestState(job(Path(temporary), 1), 0)
            with dispatcher.request_scope(state):
                with expect_error(RuntimeError):
                    dispatcher({}, max_threads=1, verbose=False)
                assert state.lock_owned
                assert not state.in_vmec
                assert len(state.calls) == 1

    def test_environment_drift_is_failure_without_mutating_process_environment(self):
        fingerprint = ["initial"]
        dispatcher = batch.CaptureDispatcher(
            lambda *_args, **_kwargs: None, fingerprint=lambda: fingerprint[0]
        )
        with tempfile.TemporaryDirectory() as temporary:
            state = batch.RequestState(job(Path(temporary), 1), 0)
            with expect_error(RuntimeError), dispatcher.request_scope(state):
                fingerprint[0] = "changed"
            assert not state.lock_owned
            assert getattr(dispatcher.local, "request", None) is None

    def test_capture_installs_only_once_and_numeric_hashes_match_real_bytes(self):
        module = SimpleNamespace(run=lambda: None)
        dispatcher = batch.CaptureDispatcher(module.run)
        batch.install_capture(module, dispatcher)
        with expect_error(RuntimeError):
            batch.install_capture(module, dispatcher)
        array = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
        model = SimpleNamespace(model_dump=lambda: {"array": array, "text": "skip"})
        fields = batch.numeric_fields(model, np)
        assert list(fields) == ["array"]
        assert fields["array"]["shape"] == [3, 2]
        assert fields["array"]["dtype"] == "float64"
        assert (
            fields["array"]["sha256"]
            == hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        )

    def test_default_cli_is_plan_only_without_vmec_import_or_output_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"path": "seed.json"}]))
            output = root / "output"
            arguments = [
                "batch",
                "--manifest",
                str(manifest),
                "--references",
                str(root),
                "--stella",
                str(root),
                "--out",
                str(output),
            ]
            text = io.StringIO()
            with (
                mock.patch("sys.argv", arguments),
                mock.patch.object(batch, "execute") as execute,
                mock.patch.object(batch.importlib, "import_module") as imports,
                contextlib.redirect_stdout(text),
            ):
                assert batch.main() == 0
            assert not execute.called
            assert not imports.called
            assert not output.exists()
            assert json.loads(text.getvalue())["planned_evaluations"] == 6

    def test_reference_extension_requires_existing_absolute_pinned_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary).resolve() / "reference.so"
            reference.write_bytes(b"frozen extension fixture")
            assert (
                batch.reference_extension(
                    {"VMECPP_FORWARD_REFERENCE_SO": str(reference)}
                )
                == reference
            )
            environment = {"VMECPP_FORWARD_REFERENCE_SO": str(reference)}
            frozen = {str(reference): batch.digest(reference)}
            assert batch.reference_extension(environment, frozen) == reference
            with expect_error(RuntimeError):
                batch.reference_extension(environment, {})
            reference.write_bytes(b"changed extension fixture")
            with expect_error(RuntimeError):
                batch.reference_extension(environment, frozen)
            for environment in (
                {},
                {"VMECPP_FORWARD_REFERENCE_SO": "relative.so"},
                {"VMECPP_FORWARD_REFERENCE_SO": str(reference.parent)},
                {"VMECPP_FORWARD_REFERENCE_SO": str(reference.with_suffix(".missing"))},
            ):
                with expect_error(RuntimeError):
                    batch.reference_extension(environment)

    def test_repeat_coverage_never_calls_singletons_deterministic(self):
        first = {
            "seed": "first.json",
            "track": "P1",
            "request_metrics_sha256": "metric identity",
            "numeric_output_identity": "numeric identity",
        }
        second = {**first, "seed": "second.json"}
        singleton = batch.repeat_coverage([first, second])
        assert not singleton["all_cases_repeated"]
        assert singleton["repeated_outputs_exact"] is None
        mixed = batch.repeat_coverage([first, first, second])
        assert mixed["repeated_outputs_exact"] is None
        repeated = batch.repeat_coverage([first, first, second, second])
        assert repeated["all_cases_repeated"]
        assert repeated["repeated_outputs_exact"] is True
        drifted = {**first, "numeric_output_identity": "changed identity"}
        drift = batch.repeat_coverage([first, drifted])
        assert drift["all_cases_repeated"]
        assert drift["repeated_outputs_exact"] is False


if __name__ == "__main__":
    unittest.main()
