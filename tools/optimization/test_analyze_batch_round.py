"""Analytic interval controls for graph/NVTX scheduling analysis."""

import unittest

from tools.optimization.analyze_batch_round import (
    complement,
    duration,
    intersection,
    presence_bins,
    union,
)


class IntervalControl(unittest.TestCase):
    def test_union_clips_and_merges_touching(self):
        assert union([(-2, 2), (2, 4), (3, 5), (9, 11)], 0, 10) == [(0, 5), (9, 10)]

    def test_intersection_touching_is_empty(self):
        assert intersection([(0, 3), (5, 8)], [(3, 6)]) == [(5, 6)]

    def test_complement(self):
        assert complement([(2, 3), (5, 20)], 0, 10) == [(0, 2), (3, 5)]

    def test_exclusive_partition(self):
        bins = presence_bins([(2, 4)], {"a": [(0, 6)], "b": [(5, 10)]}, 0, 10)
        actual = {tuple(row["phases_present"]): row["duration_ns"] for row in bins}
        assert actual == {("a",): 3, ("a", "b"): 1, ("b",): 4}
        assert sum(actual.values()) == duration(complement([(2, 4)], 0, 10))


if __name__ == "__main__":
    unittest.main()
