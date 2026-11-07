"""CPU fixtures for opt-in request/native phase clocks and exception paths."""

# Only reviewed local source is executed to exercise diagnostic copies.
# ruff: noqa: PT027, S102
import sys
import tempfile
import types
import unittest
from pathlib import Path

from tools.optimization.queue_phase_diagnostic import (
    PhaseRecorder,
    instrument_batch_source,
    instrument_queue_source,
)


class QueuePhaseTests(unittest.TestCase):
    def test_fake_queue_preserves_role_ownership_and_propagates_failure(self):
        directory = Path(__file__).parent
        recorder = PhaseRecorder()
        batch = types.ModuleType("diagnostic_batch_test")
        batch.__file__ = str(directory / "batch_throughput.py")
        sys.modules[batch.__name__] = batch
        batch.__dict__["PHASE_RECORDER"] = recorder
        exec(instrument_batch_source(Path(batch.__file__).read_text()), batch.__dict__)
        queue = types.ModuleType("diagnostic_queue_test")
        queue.__file__ = str(directory / "batch_solve_queue.py")
        queue.__dict__["PHASE_RECORDER"] = recorder
        exec(instrument_queue_source(Path(queue.__file__).read_text()), queue.__dict__)

        def native(value, **_):
            if value == 2:
                message = "native error"
                raise ValueError(message)
            return value

        dispatcher = queue.QueuedCaptureDispatcher(
            native, native_workers=2, fingerprint=lambda: "fixed"
        )
        with tempfile.TemporaryDirectory() as temporary:
            jobs = [
                batch.Job(
                    str(i),
                    "seed",
                    "P1",
                    0,
                    False,
                    '{"value":' + str(i) + "}",
                    Path(temporary) / str(i),
                )
                for i in (1, 2)
            ]
            seen = []

            def finalize(state):
                seen.append(state)
                assert recorder.snapshot(state.job.key)[-1]["phase"] == "request_end"
                return {"gate": {"exact": state.error is None}}

            batch.run_rounds(
                [jobs],
                dispatcher,
                lambda request, _: dispatcher(
                    request["value"], max_threads=1, verbose=False
                ),
                finalize,
                2,
            )
            with self.assertRaisesRegex(ValueError, "native error"):
                dispatcher.close()
            assert all(future.done() for future in dispatcher.futures)
            assert len(seen) == 2
            assert sum(state.error is not None for state in seen) == 1
            for job in jobs:
                events = recorder.snapshot(job.key)
                names = [event["phase"] for event in events]
                assert names == [
                    "request_start",
                    "initial_gate_acquired",
                    "setup_end_before_submit",
                    "native_method_start",
                    "native_method_end",
                    "native_wait_end_before_gate",
                    "post_gate_acquired",
                    "evaluation_end",
                    "request_end",
                ]
                assert len({e["tid"] for e in events if e["role"] == "adapter"}) == 1
                assert len({e["tid"] for e in events if e["role"] == "native"}) == 1
                assert {e["tid"] for e in events if e["role"] == "adapter"}.isdisjoint(
                    {e["tid"] for e in events if e["role"] == "native"}
                )

    def test_changed_anchor_fails_closed(self):
        with self.assertRaises(ValueError):
            instrument_queue_source("not the reviewed source")
