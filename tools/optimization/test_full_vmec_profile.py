"""CPU-only lifecycle checks for the isolated whole-solver diagnostic."""

# ruff: noqa: PT027
import threading
import unittest

from tools.optimization.full_vmec_profile import profile_call


class ProfileLifecycleTests(unittest.TestCase):
    def test_original_call_bracketed_and_value_preserved(self):
        events = []
        call = profile_call(
            lambda: events.append("original") or 42,
            lambda: events.append("start") or 0,
            lambda: events.append("stop") or 0,
            lambda event, status: events.append((event, status)),
            lambda _: events.append("push"),
            lambda: events.append("pop"),
        )
        assert call() == 42
        assert events == [
            "start",
            ("start_immediately_before_vmecpp_run", 0),
            "push",
            "original",
            "pop",
            "stop",
            ("stop_immediately_after_vmecpp_run", 0),
        ]

    def test_original_exception_still_stops(self):
        events = []

        def fail():
            msg = "original"
            raise ValueError(msg)

        call = profile_call(
            fail,
            lambda: 0,
            lambda: events.append("stop") or 0,
            lambda *_: None,
            lambda _: None,
            lambda: None,
        )
        with self.assertRaisesRegex(ValueError, "original"):
            call()
        assert events == ["stop"]

    def test_nonmain_thread_rejected_before_profiler(self):
        events = []
        call = profile_call(
            lambda: None,
            lambda: events.append("start") or 0,
            lambda: 0,
            lambda *_: None,
            lambda _: None,
            lambda: None,
        )

        def run():
            with self.assertRaisesRegex(RuntimeError, "main thread"):
                call()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert not events
