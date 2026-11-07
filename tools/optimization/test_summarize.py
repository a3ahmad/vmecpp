"""Regression checks for the identity gate used in optimization reports."""

import math
import unittest

from tools.optimization.summarize import metric_drift


class MetricDriftTest(unittest.TestCase):
    def test_one_ulp_is_rejected(self):
        result = metric_drift({"x": 1.0}, {"x": math.nextafter(1.0, 2.0)})
        assert not result["x"]["exact"]
        assert result["x"]["relative"] > 0

    def test_zero_baseline_does_not_mask_drift(self):
        result = metric_drift({"x": 0.0}, {"x": 1e-30})
        assert result["x"]["relative"] is None
        assert result["x"]["absolute"] == 1e-30

    def test_null_and_missing_are_distinct(self):
        assert not metric_drift({"x": None}, {})["x"]["exact"]
        assert metric_drift({"x": None}, {"x": None})["x"]["exact"]


if __name__ == "__main__":
    unittest.main()
