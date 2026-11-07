"""Analytic controls for named interval clipping in full-solver reports."""

import unittest

from tools.optimization.analyze_full_vmec_profile import grouped


class NamedIntervalTests(unittest.TestCase):
    def test_clips_partial_intervals_and_excludes_touching(self):
        rows = [(0, 5, "a"), (4, 8, "a"), (8, 10, "b"), (10, 20, "c")]
        assert grouped(rows, 5, 10) == [
            {"name": "a", "calls": 1, "clipped_duration_ns": 3},
            {"name": "b", "calls": 1, "clipped_duration_ns": 2},
        ]

    def test_summed_overlap_is_explicitly_not_union(self):
        assert grouped([(0, 10, "a"), (0, 10, "a")], 0, 10) == [
            {"name": "a", "calls": 2, "clipped_duration_ns": 20}
        ]
