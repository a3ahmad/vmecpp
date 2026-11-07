"""Analytic interval controls for diagnostic overlap measurements."""

import contextlib
import unittest

from tools.optimization.analyze_batch_timeline import interval_metrics


@contextlib.contextmanager
def expect_value_error(match):
    try:
        yield
    except ValueError as error:
        caught = str(error)
    else:
        message = "Expected invalid interval input to be rejected"
        raise AssertionError(message)
    assert match in caught


class IntervalTests(unittest.TestCase):
    def test_known_intervals(self):
        cases = [
            ([(0, 4, 1), (4, 8, 2)], 8, 0, 1, 2),
            ([(1, 6, 1), (3, 8, 2)], 7, 3, 2, 2),
            ([(1, 6, 1), (3, 8, 1)], 7, 0, 1, 2),
            ([(-2, 2, 1), (8, 12, 2)], 4, 0, 1, 6),
            ([(3, 3, 1)], 0, 0, 0, 10),
            ([], 0, 0, 0, 10),
        ]
        for intervals, busy, overlap, streams, gap in cases:
            with self.subTest(intervals=intervals):
                actual = interval_metrics(intervals, 10)
                assert actual["union_busy_ns"] == busy
                assert actual["distinct_stream_overlap_ns"] == overlap
                assert actual["maximum_simultaneous_streams"] == streams
                assert actual["longest_no_target_activity_ns"] == gap
                assert (
                    sum(actual["duration_ns_by_simultaneous_stream_count"].values())
                    == 10
                )

    def test_invalid_intervals_rejected(self):
        with expect_value_error("reversed"):
            interval_metrics([(4, 2, 1)], 10)
        with expect_value_error("positive"):
            interval_metrics([], 0)
