"""Recorded marker evidence takes precedence over callback depth returns."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.optimization import cloud_pilot_profileonly2 as pilot
from tools.optimization.validate_profile_marker import validate_marker


class RecordedMarker(unittest.TestCase):
    def test_negative_callback_return_does_not_skip_actual_call(self):
        bounds = []
        calls = []
        result = pilot.profiled_call(
            lambda: calls.append("actual") or 7,
            lambda: 0,
            lambda: 0,
            lambda: -1,
            lambda: -1,
            bounds,
        )
        assert result == 7
        assert calls == ["actual"]
        assert bounds[0]["nvtx_push_return"] == -1
        assert bounds[0]["nvtx_pop_return"] == -1
        assert bounds[0]["before_actual_vmecpp_run_ns"] is not None

    def test_recorded_range_and_contained_target_kernels(self):
        assert self.check("valid")["exact"]

    def test_absent_empty_memset_table_is_supported(self):
        assert self.check("no-memset-table")["exact"]

    def test_missing_range_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"range|kernels"):  # noqa: PT027
            self.check("missing")

    def test_other_process_range_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"range|kernels"):  # noqa: PT027
            self.check("other")

    def test_kernel_outside_range_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"range|kernels"):  # noqa: PT027
            self.check("outside")

    def test_no_kernel_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"range|kernels"):  # noqa: PT027
            self.check("no-kernel")

    def check(self, case):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript(
                    "CREATE TABLE PROCESSES(pid,globalPid);"
                    "CREATE TABLE StringIds(id,value);"
                    "CREATE TABLE NVTX_EVENTS(start,end,globalTid,text,textId);"
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid);"
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,globalPid);"
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start,end,globalPid);"
                    "INSERT INTO PROCESSES VALUES(12,201326592);"
                )
                if case != "missing":
                    db.execute(
                        "INSERT INTO NVTX_EVENTS VALUES(10,100,?,?,NULL)",
                        (201326592 if case != "other" else 0, "whole vmecpp.run"),
                    )
                if case != "no-kernel":
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,201326592)",
                        (9 if case == "outside" else 20, 30),
                    )
                if case == "no-memset-table":
                    db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMSET")
            return validate_marker(path, 12)


if __name__ == "__main__":
    unittest.main()
