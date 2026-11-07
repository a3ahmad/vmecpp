"""CPU-only ownership, queue progress and drain regressions."""

# Standard-library tests also run in the pinned solver environment.
# ruff: noqa: PT027
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.optimization.batch_solve_queue import (
    QueuedCaptureDispatcher,
    pinned_pool,
    terminal_result,
)
from tools.optimization.batch_throughput import Job, RequestState


def state(name):
    return RequestState(
        Job(name, "fixture", "P1", 0, False, "{}", Path("/tmp") / name), 0
    )


class QueueControls(unittest.TestCase):
    def test_lazy_native_worker_created_from_pinned_adapter(self):
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 2:
            self.skipTest("Two CPU cores required for nested affinity control")
        native = pinned_pool([available[0]])(max_workers=1)
        adapters = pinned_pool([available[1]])(max_workers=1)
        try:

            def adapter():
                assert os.sched_getaffinity(0) == {available[1]}
                return native.submit(lambda: sorted(os.sched_getaffinity(0))).result(
                    timeout=3
                )

            assert adapters.submit(adapter).result(timeout=3) == [available[0]]
        finally:
            adapters.shutdown(wait=True)
            native.shutdown(wait=True)

    def test_native_owner_and_result_lifetime(self):
        owner = threading.get_native_id()
        observed = []
        activity = threading.local()

        def counters():
            return getattr(activity, "counts", (0, 0, 0))

        def native(value, **_):
            observed.append(threading.get_native_id())
            activity.counts = (1, 5, 7)
            return {"value": value}

        dispatcher = QueuedCaptureDispatcher(
            native,
            native_workers=1,
            fingerprint=lambda: "fixed",
            activity=counters,
            require_gpu=True,
            require_graph=True,
        )
        request = state("one")
        value = {"array": [1, 2]}
        with dispatcher.request_scope(request):
            output = dispatcher(value, max_threads=1, verbose=False)
        dispatcher.close()
        value["array"][0] = 9
        assert observed[0] != owner
        assert request.calls[0].activity_before == (0, 0, 0)
        assert request.calls[0].activity_after == (1, 5, 7)
        assert request.calls[0].input_snapshot == {"array": [1, 2]}
        assert request.calls[0].output is output
        assert output == {"value": value}
        assert all(f.done() for f in dispatcher.futures)

    def test_worker_advances_while_adapter_waits_for_post_gate(self):
        native_started = threading.Event()
        release_first = threading.Event()
        second_completed = threading.Event()
        gate_held = threading.Event()
        release_gate = threading.Event()

        def native(value, **_):
            if value == 1:
                native_started.set()
                assert release_first.wait(3)
            else:
                second_completed.set()
            return value

        dispatcher = QueuedCaptureDispatcher(
            native, native_workers=1, fingerprint=lambda: "fixed"
        )

        def adapter(value):
            with dispatcher.request_scope(state(str(value))):
                return dispatcher(value, max_threads=1, verbose=False)

        def hold_postprocessing():
            with dispatcher.gate:
                gate_held.set()
                assert release_gate.wait(3)

        with ThreadPoolExecutor(max_workers=3) as adapters:
            first = adapters.submit(adapter, 1)
            assert native_started.wait(3)
            second = adapters.submit(adapter, 2)
            # The second request must have submitted before the gate holder wins.
            while len(dispatcher.futures) < 2:
                threading.Event().wait(0.001)
            holder = adapters.submit(hold_postprocessing)
            assert gate_held.wait(3)
            release_first.set()
            assert second_completed.wait(3)
            assert not first.done()
            assert not second.done()
            release_gate.set()
            assert first.result(timeout=3) == 1
            assert second.result(timeout=3) == 2
            holder.result(timeout=3)
        dispatcher.close()

    def test_native_exception_restores_adapter_ownership_and_drains(self):
        def native(*_, **__):
            message = "native failure"
            raise ValueError(message)

        dispatcher = QueuedCaptureDispatcher(
            native, native_workers=1, fingerprint=lambda: "fixed"
        )
        request = state("failure")
        with dispatcher.request_scope(request):
            with self.assertRaisesRegex(ValueError, "native failure"):
                dispatcher(1, max_threads=1, verbose=False)
            assert request.lock_owned
            assert not request.in_vmec
        with self.assertRaisesRegex(ValueError, "native failure"):
            dispatcher.close()
        assert all(f.done() for f in dispatcher.futures)

    def test_interrupted_wait_retries_until_terminal(self):
        class InterruptedFuture:
            calls = 0

            def result(self):
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt
                return 4

            def done(self):
                return self.calls >= 2

        future = InterruptedFuture()
        with self.assertRaises(KeyboardInterrupt):
            terminal_result(future)
        assert future.done()
        assert future.calls == 2

    def test_missing_request_never_submits(self):
        dispatcher = QueuedCaptureDispatcher(
            lambda *_: None, native_workers=1, fingerprint=lambda: "fixed"
        )
        with self.assertRaises(RuntimeError):
            dispatcher(1, max_threads=1, verbose=False)
        assert not dispatcher.futures
        dispatcher.close()


if __name__ == "__main__":
    unittest.main()
