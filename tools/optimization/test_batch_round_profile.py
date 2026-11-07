"""CPU-only guards for the isolated full-round diagnostic."""

# Keep these guards runnable with the standard-library test runner.
# ruff: noqa: PT027

import unittest
from pathlib import Path

from tools.optimization.batch_round_profile import RoundProfile, instrument


class RoundProfileTest(unittest.TestCase):
    def test_exact_source_and_stop_before_validation(self):
        path = Path(__file__).with_name("batch_throughput.py")
        source = instrument(path.read_text())
        compile(source, "diagnostic-copy", "exec")
        assert source.index("round_profile.stop()") < source.index(
            "record = finalize(state)"
        )
        assert "round_profile.stop_if_active()" in source
        assert source.count("phase_push('nativecall')") == 1
        assert "gatewait:post-native" in source
        assert "gatewait:initial" in source

    def test_lifecycle_once_and_cleanup(self):
        events = []
        profile = RoundProfile(
            lambda: 0, lambda: 0, lambda *event: events.append(event)
        )
        assert not profile.active
        profile.stop_if_active()
        profile.start()
        assert profile.active
        profile.stop_if_active()
        assert not profile.active
        assert len(events) == 2
        with self.assertRaises(RuntimeError):
            profile.start()
        with self.assertRaises(RuntimeError):
            profile.stop()

    def test_start_failure_never_activates(self):
        profile = RoundProfile(lambda: 1, lambda: 0, lambda *_: None)
        with self.assertRaises(RuntimeError):
            profile.start()
        assert not profile.active


if __name__ == "__main__":
    unittest.main()
