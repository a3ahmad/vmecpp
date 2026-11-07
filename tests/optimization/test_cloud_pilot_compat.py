"""No official process may start without exact repeated CPU replay evidence."""

# Stdlib unittest avoids adding a test dependency to the pinned tools environment.
# ruff: noqa: PT027
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.optimization import cloud_cpu_compatibility as compatibility
from tools.optimization import cloud_pilot_compat as pilot


class CompatiblePilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        value = {
            "flags": dict(pilot.REQUIRED_FLAGS),
            "reference_extension_sha256": pilot.REFERENCE_SHA,
            "controller_pool_GIL_queue": "OFF",
            "max_hourly_usd": 1,
            "maximum_lease_seconds": 3600,
            "saved_output_cpu_compatibility": "required",
        }
        pilot.save(self.root / "pilot-config.json", value)

    def select_haswell(self):
        records = []
        for repeat in (0, 1):
            row = {
                "variant": "blas-haswell",
                "environment_overrides": {"OPENBLAS_CORETYPE": "Haswell"},
                "repeat_index": repeat,
                "job_passed": True,
                "owned_children_reaped": True,
                "metric_exact": True,
                "equilibrium_exact": True,
                "source_exact": True,
                "intermediates_exact": True,
            }
            if repeat:
                row["repeat_against_first"] = {
                    "metric_exact": True,
                    "equilibrium_exact": True,
                    "source_exact": True,
                    "intermediates_exact": True,
                }
            records.append(row)
        selected, summary = pilot.compatibility_paths(self.root)
        pilot.save(
            summary,
            {
                "records": records,
                "requested_variants": ["blas-haswell"],
                "requested_repeats": 2,
            },
        )
        pilot.save(selected, compatibility.decision_from_file(summary))

    def test_official_environment_requires_cpu_evidence(self):
        with self.assertRaises(FileNotFoundError):
            pilot.environment(self.root, "baseline")
        imported = pilot.environment(self.root, "baseline", allow_unselected=True)
        assert "OPENBLAS_CORETYPE" not in imported

    def test_both_variants_get_same_selected_dispatch_and_exact_gpu_flags(self):
        self.select_haswell()
        with patch.dict(
            os.environ,
            {"OPENBLAS_CORETYPE": "untrusted", "VMECPP_GPU_RESIDUAL_ROUNDS": "12"},
        ):
            baseline = pilot.environment(self.root, "baseline")
            candidate = pilot.environment(self.root, "candidate")
        assert (
            baseline["OPENBLAS_CORETYPE"] == candidate["OPENBLAS_CORETYPE"] == "Haswell"
        )
        assert "VMECPP_GPU_RESIDUAL_ROUNDS" not in baseline
        assert candidate["VMECPP_GPU_RESIDUAL_ROUNDS"] == "32"
        assert "LD_PRELOAD" not in baseline
        assert candidate["LD_PRELOAD"] == str(self.root / "work/lib/solver-shim.so")

    def test_worker_declines_wrong_dispatch_before_metric_imports(self):
        self.select_haswell()
        with (
            patch.dict(os.environ, {"OPENBLAS_CORETYPE": "wrong"}),
            self.assertRaisesRegex(RuntimeError, "Actual worker CPU dispatch"),
        ):
            pilot.worker(self.root, "baseline", [])

    def test_profile_evaluation_cannot_bypass_missing_replay(self):
        with patch.object(pilot, "run_owned") as execute:
            with self.assertRaises(FileNotFoundError):
                pilot.evaluate(
                    self.root,
                    self.root / "seed.json",
                    "P3",
                    "candidate",
                    "diagnostic",
                    1,
                    3601,
                    True,
                    ["nsys"],
                )
            execute.assert_not_called()

    def test_failed_probe_does_not_write_a_selection(self):
        with (
            patch.object(pilot, "run_owned", return_value={"passed": False}),
            patch.object(pilot, "time_budget", return_value=1),
            self.assertRaisesRegex(RuntimeError, "CPU compatibility probe failed"),
        ):
            pilot.run_compatibility_probe(self.root, 1, 3601)
        selected, _ = pilot.compatibility_paths(self.root)
        assert not selected.exists()

    def test_bootstrap_records_failure_and_cannot_claim_complete(self):
        pilot.save(self.root / "bootstrap-status.json", {"phase": "complete"})
        with (
            patch.object(pilot, "run_compatibility_probe", side_effect=TimeoutError),
            self.assertRaises(TimeoutError),
        ):
            pilot.finish_cpu_compatibility(self.root, 1, 3601)
        status = json.loads((self.root / "bootstrap-status.json").read_text())
        assert status["phase"] == "failed_CPU_compatibility"

    def test_bootstrap_records_exact_selection_without_qualifying_metrics(self):
        self.select_haswell()
        pilot.save(self.root / "bootstrap-status.json", {"phase": "complete"})
        with patch.object(pilot, "run_compatibility_probe"):
            pilot.finish_cpu_compatibility(self.root, 1, 3601)
        status = json.loads((self.root / "bootstrap-status.json").read_text())
        assert status["phase"] == "complete"
        assert status["CPU_compatibility"] == "exact_replay_selection_only"
        assert status["CPU_compatibility_dispatch"] == {"OPENBLAS_CORETYPE": "Haswell"}

    def test_denied_counters_stop_before_any_fixture_solve(self):
        pilot.save(
            self.root / "bootstrap-status.json",
            {"phase": "complete", "runtime_files": {}},
        )
        pilot.save(self.root / "payload-manifest.json", {"files": {}})
        pilot.save(
            self.root / "fixtures/manifest.json",
            [{"path": "seed-" + str(index) + ".json"} for index in range(3)],
        )
        with (
            patch.object(pilot, "preflight"),
            patch.object(pilot, "selected_cpu_environment", return_value={}),
            patch.object(pilot, "hardware_snapshot"),
            patch.object(pilot.os, "sched_getaffinity", return_value={0}),
            patch.object(pilot.os, "sched_setaffinity"),
            patch.object(pilot, "descendants", return_value={}),
            patch.object(pilot, "counters", return_value={"counter_permission": False}),
            patch.object(pilot, "evaluate") as evaluate,
        ):
            code = pilot.run_pilot(self.root, 1, 3601, 6)
        evaluate.assert_not_called()
        assert code == 78
        state = json.loads((self.root / "pilot-status.json").read_text())
        assert state["phase"] == "stopped_counter_permission_or_coverage"
        assert state["case_pairs"] == []
