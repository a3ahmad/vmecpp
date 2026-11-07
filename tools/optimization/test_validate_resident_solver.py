"""Regression tests for rejecting incomplete or drifting resident solver results."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.optimization.validate_resident_lifecycle import parse_experiment_overrides
from tools.optimization.validate_resident_solver import (
    compare_input_decks,
    compare_reports,
    compare_traces,
    first_difference,
)


def complete_report():
    return {
        "status": "SUCCESS",
        "metrics": {"x": 1.0, "zero": 0.0, "optional": None},
        "metrics_sha256": "metrics",
        "request_metrics_sha256": "identity",
        "numerical_wout_hashes": [{"value": {"sha256": "wout"}}],
        "numerical_diagnostic_hashes": [
            {
                name: {"value": {"sha256": "diagnostic"}}
                for name in ("jxbout", "mercier", "threed1_volumetrics")
            }
        ],
    }


class ResidentValidationTest(unittest.TestCase):
    def test_lifecycle_flags_allow_explicit_valid_experiments(self):
        flags = {
            "VMECPP_GPU_SOLVER_RESIDENT_CONTROL": "1",
            "VMECPP_GPU_SOLVER_GRAPH": "1",
            "VMECPP_GPU_SOLVER_PINNED_SCALARS": "1",
            "VMECPP_GPU_RESIDUAL_ROUNDS": "12",
            "VMECPP_GPU_RESIDUAL_GRAPH": "1",
            "VMECPP_GPU_DIAGNOSTICS_ROUNDS": "16",
            "VMECPP_GPU_DIAGNOSTICS_GRAPH": "1",
        }
        assert (
            parse_experiment_overrides(
                [f"{key}={value}" for key, value in flags.items()]
            )
            == flags
        )

    def test_lifecycle_flags_reject_credentials_reserved_and_invalid_values(self):
        for assignment in (
            "VAST_KEY=do-not-record",
            "VMECPP_GPU_SECRET=do-not-record",
            "VMECPP_GPU_SOLVER=1",
            "VMECPP_FORWARD_REFERENCE_SO=/wrong/extension.so",
            "VMECPP_GPU_SOLVER_TRACE=/wrong/trace.jsonl",
            "VMECPP_GPU_DIAGNOSTICS_ROUNDS=secret",
            "VMECPP_GPU_DIAGNOSTICS_ROUNDS=33",
            "VMECPP_GPU_DIAGNOSTICS_GRAPH=2",
            "VMECPP_GPU_SOLVER_GRAPH=2",
            "VMECPP_GPU_SOLVER_PINNED_SCALARS=2",
            "VMECPP_GPU_RESIDUAL_ROUNDS=-1",
            "VMECPP_GPU_RESIDUAL_ROUNDS",
        ):
            try:
                parse_experiment_overrides([assignment])
            except ValueError:
                continue
            message = "Unsafe lifecycle override was accepted"
            raise AssertionError(message)

    def test_complete_identity_passes(self):
        report = complete_report()
        assert compare_reports(report, copy.deepcopy(report))["exact"]

    def test_one_ulp_and_signed_zero_are_rejected_even_with_stale_hash(self):
        baseline = complete_report()
        for key, value in (("x", math.nextafter(1.0, 2.0)), ("zero", -0.0)):
            candidate = copy.deepcopy(baseline)
            candidate["metrics"][key] = value
            assert not compare_reports(baseline, candidate)["exact"]

    def test_missing_diagnostics_on_both_sides_cannot_pass(self):
        for value in ([], [{}], [{"jxbout": {"value": "x"}}]):
            report = complete_report()
            report["numerical_diagnostic_hashes"] = value
            assert not compare_reports(report, report)["exact"]

    def test_failed_status_rejected(self):
        baseline = complete_report()
        candidate = copy.deepcopy(baseline)
        candidate["status"] = "FAILED"
        assert not compare_reports(baseline, candidate)["exact"]

    def test_first_hash_difference_names_the_field(self):
        baseline = complete_report()
        candidate = copy.deepcopy(baseline)
        candidate["numerical_wout_hashes"][0]["value"]["sha256"] = "different"
        result = compare_reports(baseline, candidate)
        assert not result["exact"]
        assert (
            result["first_difference_per_identity_field"]["numerical_wout_hashes"][
                "path"
            ]
            == "[0].value.sha256"
        )

    def test_missing_and_null_are_distinct(self):
        assert first_difference({"x": None}, {}) == {
            "path": "x",
            "baseline_present": True,
            "candidate_present": False,
        }

    def test_trajectory_difference_precedes_later_length_difference(self):
        result = first_difference([{"residual": 1.0}, {}], [{"residual": 2.0}])
        assert result["path"] == "[0].residual"

    def test_trace_float_signed_zero_is_not_equal(self):
        assert first_difference([0.0], [-0.0]) is not None

    def test_trace_ignores_only_gpu_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            candidate = Path(directory) / "candidate.jsonl"
            baseline.write_text('{"gpu":0,"ns":25,"residuals":[1.0]}\n')
            candidate.write_text('{"gpu":1,"ns":25,"residuals":[1.0]}\n')
            assert compare_traces(baseline, candidate)["exact"]
            candidate.write_text('{"gpu":1,"ns":25,"residuals":[1.01]}\n')
            difference = compare_traces(baseline, candidate)
            assert not difference["exact"]
            assert difference["first_difference"]["path"] == "[0].residuals[0]"

    def test_changed_physics_deck_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline"
            candidate = Path(directory) / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            (baseline / "actual_indata_0.json").write_text('{"ftol_array":[1e-15]}')
            (candidate / "actual_indata_0.json").write_text('{"ftol_array":[1e-14]}')
            result = compare_input_decks(baseline, candidate)
            assert not result["exact"]
            assert result["first_difference"]["path"] == "[0].ftol_array[0]"

    def test_only_undefined_initial_bad_jacobian_residuals_ignored(self):
        events = [
            {"ns": 25, "iter2": 1, "bad_jacobian": 1, "residuals": [123.0]},
            {"ns": 25, "iter2": 1, "bad_jacobian": 0, "residuals": [1.0]},
            {"ns": 25, "iter2": 2, "bad_jacobian": 1, "residuals": [1.0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            candidate = Path(directory) / "candidate.jsonl"
            baseline.write_text("\n".join(json.dumps(event) for event in events))
            events[0]["residuals"] = [456.0]
            candidate.write_text("\n".join(json.dumps(event) for event in events))
            result = compare_traces(baseline, candidate)
            assert result["exact"]
            assert result["undefined_initial_bad_jacobian_residual_events"] == [
                [0],
                [0],
            ]
            events[2]["residuals"] = [2.0]
            candidate.write_text("\n".join(json.dumps(event) for event in events))
            result = compare_traces(baseline, candidate)
            assert not result["exact"]
            assert result["first_difference"]["path"] == "[2].residuals[0]"


if __name__ == "__main__":
    unittest.main()
