"""Regression checks for complete, isolated, identity-preserving paired runs."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.optimization.campaign_gates import (
    clean_environment,
    complete_capture,
    pair_gate,
)
from tools.optimization.report_pairs import summarize_pairs


def fixture():
    fields = {"value": {"shape": [1], "dtype": "float64", "sha256": "a" * 64}}
    report = {
        "status": "SUCCESS",
        "profiled": False,
        "metrics": {"x": 1.0},
        "metrics_sha256": "b" * 64,
        "request_metrics_sha256": "c" * 64,
        "vmec_wall_seconds": [1.0],
        "adapter_wall_seconds": 2.0,
        "numerical_wout_hashes": [fields],
        "numerical_diagnostic_hashes": [
            {name: fields for name in ("jxbout", "mercier", "threed1_volumetrics")}
        ],
    }
    capture = {
        "actual_indata": [{"mpol": 10, "ftol_array": [1e-17]}],
        "actual_indata_sha256": ["d" * 64],
        "provenance_sha256": "d" * 64,
        "provenance": {
            "harness_sha256": "a" * 64,
            "campaign_wrapper_sha256": "b" * 64,
            "campaign_configuration_sha256": "c" * 64,
            "extensions": {"extension": "e" * 64},
            "packages": {"vmecpp": "0.4.11"},
            "experimental_runtime": {"VMECPP_GPU_SOLVER_TRACE": None},
        },
    }
    return report, capture


def samples():
    report, capture = fixture()
    return [
        {
            "label": label,
            "pair": pair,
            "warmup": pair == 0,
            "seed": "fixture.json",
            "track": "P3",
            "exit_status": 0,
            "report": copy.deepcopy(report),
            "captures": copy.deepcopy(capture),
        }
        for pair in range(6)
        for label in ("baseline", "candidate")
    ]


class PairedCampaignTest(unittest.TestCase):
    def test_complete_pair_passes(self):
        report, capture = fixture()
        assert pair_gate(
            report, copy.deepcopy(report), capture, copy.deepcopy(capture)
        )["exact"]

    def test_missing_diagnostics_on_both_sides_fail(self):
        report, capture = fixture()
        for missing in (None, [], [{}], [{"jxbout": {"x": {}}}]):
            report["numerical_diagnostic_hashes"] = missing
            assert not pair_gate(report, report, capture, capture)["exact"]

    def test_tolerance_ulp_and_signed_zero_deck_changes_fail(self):
        report, capture = fixture()
        for value in (math.nextafter(1e-17, math.inf), -0.0):
            candidate = copy.deepcopy(capture)
            candidate["actual_indata"][0]["ftol_array"] = [value]
            assert not pair_gate(report, report, capture, candidate)["exact"]

    def test_signed_zero_metric_drift_with_stale_hash_fails(self):
        report, capture = fixture()
        report["metrics"]["x"] = 0.0
        candidate = copy.deepcopy(report)
        candidate["metrics"]["x"] = -0.0
        assert not pair_gate(report, candidate, capture, capture)["exact"]

    def test_capture_count_and_profiled_runs_fail(self):
        report, capture = fixture()
        report["vmec_wall_seconds"] *= 2
        assert not complete_capture(report, capture)
        report, capture = fixture()
        report["profiled"] = True
        assert not complete_capture(report, capture)
        report["profiled"] = False
        capture["provenance"]["experimental_runtime"]["VMECPP_GPU_SOLVER_TRACE"] = (
            "/trace"
        )
        assert not complete_capture(report, capture)

    def test_environment_isolation_preserves_unrelated_settings(self):
        original = {
            "VMECPP_UNRECOGNIZED_FUTURE_FLAG": "1",
            "LD_PRELOAD": "/old.so",
            "CGA_PROFILE": "1",
            "CUDA_INJECTION64_PATH": "/profiler.so",
            "OMP_NUM_THREADS": "8",
            "TASK_CONTEXT": "preserved",
        }
        cleaned = clean_environment(original)
        assert cleaned["TASK_CONTEXT"] == "preserved"
        assert cleaned["OMP_NUM_THREADS"] == "1"
        assert not any(key.startswith("VMECPP_") for key in cleaned)
        assert "LD_PRELOAD" not in cleaned
        assert "CGA_PROFILE" not in cleaned
        assert original["OMP_NUM_THREADS"] == "8"

    def test_five_complete_pairs_qualify(self):
        rows = summarize_pairs(samples(), Path("/nonexistent"))
        assert rows[0]["main_table_eligible"]
        assert rows[0]["actual_indata_exact"]

    def test_missing_captures_cannot_qualify_five_identical_pairs(self):
        rows = samples()
        for row in rows:
            del row["captures"]
        assert not summarize_pairs(rows, Path("/nonexistent"))[0]["main_table_eligible"]

    def test_historical_captures_can_be_audited_from_actual_files(self):
        rows = samples()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for row in rows:
                capture = row.pop("captures")
                directory = root / row["label"] / f"fixture-P3-{row['pair']:02d}"
                directory.mkdir(parents=True)
                (directory / "actual_indata_0.json").write_text(
                    json.dumps(capture["actual_indata"][0])
                )
                (directory / "provenance.json").write_text(
                    json.dumps(capture["provenance"])
                )
            assert summarize_pairs(rows, root)[0]["main_table_eligible"]

    def test_extra_incomplete_pair_cannot_be_ignored(self):
        rows = samples()
        extra = copy.deepcopy(rows[0])
        extra["pair"] = 6
        rows.append(extra)
        assert not summarize_pairs(rows, Path("/nonexistent"))[0]["main_table_eligible"]

    def test_changed_harness_cannot_qualify(self):
        report, capture = fixture()
        candidate = copy.deepcopy(capture)
        candidate["provenance"]["harness_sha256"] = "other"
        assert not pair_gate(report, report, capture, candidate)["exact"]

    def test_warmup_drift_or_missing_warmup_cannot_qualify(self):
        rows = samples()
        rows[0]["report"]["metrics"]["x"] = 2.0
        assert not summarize_pairs(rows, Path("/nonexistent"))[0]["main_table_eligible"]
        rows = [row for row in samples() if not row["warmup"]]
        assert not summarize_pairs(rows, Path("/nonexistent"))[0]["main_table_eligible"]

    def test_shared_physics_or_build_change_between_pairs_fails(self):
        for field in ("deck", "extension", "wrapper"):
            rows = samples()
            for row in rows:
                if row["pair"] == 2:
                    if field == "deck":
                        row["captures"]["actual_indata"][0]["mpol"] = 12
                    elif field == "extension":
                        row["captures"]["provenance"]["extensions"]["extension"] = (
                            "f" * 64
                        )
                    else:
                        row["captures"]["provenance"][
                            "campaign_configuration_sha256"
                        ] = "f" * 64
            assert not summarize_pairs(rows, Path("/nonexistent"))[0][
                "main_table_eligible"
            ]

    def test_global_reference_check_is_required_and_cannot_be_ignored(self):
        rows = samples()
        assert not summarize_pairs(
            rows, Path("/nonexistent"), require_original_reference=True
        )[0]["main_table_eligible"]
        for row in rows:
            row["original_reference"] = {
                "report_sha256": "a" * 64,
                "gate": {"exact": True},
            }
        assert summarize_pairs(
            rows, Path("/nonexistent"), require_original_reference=True
        )[0]["main_table_eligible"]
        rows[0]["original_reference"]["gate"]["exact"] = False
        assert not summarize_pairs(rows, Path("/nonexistent"))[0]["main_table_eligible"]


if __name__ == "__main__":
    unittest.main()
