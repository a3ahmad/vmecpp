"""CPU-only guards for paid-time orchestration and exact parity evidence."""

# Use stdlib unittest in the pinned oracle; optional raw-array imports stay local.
# ruff: noqa: PT009, PT027, PLC0415
import ctypes
import hashlib
import importlib
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.optimization import cloud_pilot_current as pilot
from tools.optimization import cloud_pilot_current_controller as controller


class CurrentPilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.value = {
            "flags": dict(pilot.REQUIRED_FLAGS),
            "reference_extension_sha256": pilot.REFERENCE_SHA,
            "controller_pool_GIL_queue": "OFF",
            "max_hourly_usd": 1,
            "maximum_lease_seconds": 3600,
        }
        self.write_config()

    def write_config(self):
        (self.root / "pilot-config.json").write_text(json.dumps(self.value))

    def test_actual_current_config_and_flags_required(self):
        self.assertEqual(pilot.config(self.root)["flags"], pilot.REQUIRED_FLAGS)
        for wrong in ("12", "0", "24"):
            self.value["flags"]["VMECPP_GPU_RESIDUAL_ROUNDS"] = wrong
            self.write_config()
            with self.assertRaises(ValueError):
                pilot.config(self.root)

    def test_wrong_reference_and_unqualified_route_rejected(self):
        self.value["reference_extension_sha256"] = "3" * 64
        self.write_config()
        with self.assertRaises(ValueError):
            pilot.config(self.root)

    def test_local_rehearsal_cannot_start_paid_solver_flow(self):
        (self.root / "bootstrap-status.json").write_text(
            json.dumps({"phase": "complete", "local_cpu_rehearsal": True})
        )
        with self.assertRaises(RuntimeError):
            pilot.run_pilot(self.root, time.time(), time.time() + 3500, 1)

    def test_current_controller_rejects_legacy_optional_before_owner_calls(self):
        with (
            patch.object(sys, "argv", ["controller", "--include-optional-tools"]),
            patch.object(controller.cloud_pilot_controller, "main") as owner,
        ):
            with self.assertRaises(ValueError):
                controller.main()
            owner.assert_not_called()
        self.value["reference_extension_sha256"] = pilot.REFERENCE_SHA
        self.value["controller_pool_GIL_queue"] = "fused"
        self.write_config()
        with self.assertRaises(ValueError):
            pilot.config(self.root)

    def test_budget_nonfinite_and_hourly_cap(self):
        for cost in (float("nan"), float("inf"), -1, 0, 1.01, True):
            self.value["max_hourly_usd"] = cost
            self.write_config()
            with self.assertRaises(ValueError):
                pilot.config(self.root)
        now = time.time()
        with self.assertRaises(ValueError):
            pilot.time_budget(now, now + 3601, 15)
        with self.assertRaises(TimeoutError):
            pilot.time_budget(now - 3500, now + 100, 15)
        self.assertLessEqual(pilot.time_budget(now, now + 3600, 15), 15)

    def test_actual_L2_capacity_controls_calibration_footprints(self):
        mib = 1 << 20
        for capacity, expected_L2, expected_DRAM in (
            (6, 3, 256),
            (40, 20, 256),
            (96, 48, 384),
            (128, 64, 512),
        ):
            sizes = pilot.calibration_sizes(capacity * mib)
            self.assertEqual(sizes["L2_total_working_MiB"], expected_L2)
            self.assertEqual(sizes["DRAM_total_working_MiB"], expected_DRAM)
            self.assertLessEqual(sizes["L2_total_working_bytes"], capacity * mib // 2)
            self.assertGreaterEqual(
                sizes["DRAM_total_working_bytes"], 4 * capacity * mib
            )
        for bad in (0, -1, True, None, 3.5, mib, 128 * mib + 1):
            with self.assertRaises(ValueError):
                pilot.calibration_sizes(bad)

    def test_every_case_schedules_two_fresh_candidates_with_CPU_fakes(self):
        (self.root / "bootstrap-status.json").write_text(
            json.dumps({"phase": "complete", "runtime_files": {}})
        )
        (self.root / "payload-manifest.json").write_text(json.dumps({"files": {}}))
        (self.root / "fixtures").mkdir()
        (self.root / "fixtures/manifest.json").write_text(
            json.dumps(
                [
                    {"path": name + ".json"}
                    for name in ("public_p3_seed", "public_p2_seed", "third_frozen")
                ]
            )
        )
        calls = []

        def fake_evaluate(root, seed, track, variant, label, _started, _expires):
            calls.append((seed.stem, track, variant, label))
            target = root / "results/evaluations" / label
            target.mkdir(parents=True)
            return target

        now = time.time()
        with (
            patch.object(pilot, "preflight", return_value={"exact": True}),
            patch.object(pilot, "hardware_snapshot", return_value={}),
            patch.object(pilot, "evaluate", fake_evaluate),
            patch.object(pilot, "raw_compare", return_value={"exact": True}),
            patch.object(pilot, "counters", return_value={"counter_permission": True}),
            patch.object(pilot.os, "sched_getaffinity", return_value={0, 1}),
            patch.object(pilot.os, "sched_setaffinity"),
        ):
            self.assertEqual(pilot.run_pilot(self.root, now, now + 800, 6), 0)
        self.assertEqual(len(calls), 18)
        for seed in ("public_p3_seed", "public_p2_seed", "third_frozen"):
            for track in ("P1", "P3"):
                selected = [call[2] for call in calls if call[:2] == (seed, track)]
                self.assertEqual(selected, ["baseline", "candidate", "candidate"])
        state = json.loads((self.root / "pilot-status.json").read_text())
        self.assertEqual(len(state["candidate_repeat_coverage"]), 6)
        self.assertTrue(state["canonical_parity_complete"])

    def test_environment_isolation_and_one_solver_thread(self):
        inherited = {
            "PATH": "/usr/bin",
            "HOME": "/tmp",
            "VAST_KEY": "synthetic",
            "LD_PRELOAD": "/untrusted.so",
            "VMECPP_GPU_SOLVER_TRACE": "/untrusted.trace",
            "CUDA_INJECTION64_PATH": "/untrusted.so",
        }
        with patch.object(pilot.os, "environ", inherited):
            baseline = pilot.environment(self.root, "baseline")
            candidate = pilot.environment(self.root, "candidate")
        self.assertNotIn("VAST_KEY", baseline)
        self.assertNotIn("VAST_KEY", candidate)
        self.assertNotIn("LD_PRELOAD", baseline)
        self.assertNotIn("CUDA_INJECTION64_PATH", candidate)
        self.assertNotIn("VMECPP_GPU_SOLVER_TRACE", candidate)
        self.assertEqual(candidate["OMP_NUM_THREADS"], "1")
        self.assertEqual(
            len([key for key in candidate if key.startswith("VMECPP_")]), 9
        )

    def test_exact_profiler_injection_allowlist(self):
        injection = self.root / "work/nsys/target-linux-x64/libToolsInjection64.so"
        injection.parent.mkdir(parents=True)
        injection.write_bytes(b"synthetic frozen injection")
        self.value["nsys_injection_sha256"] = pilot.digest(injection)
        self.write_config()
        env = pilot.profile_environment(
            self.root,
            "candidate",
            {"CUDA_INJECTION64_PATH": str(injection), "VAST_KEY": "synthetic"},
        )
        self.assertEqual(env["CUDA_INJECTION64_PATH"], str(injection))
        self.assertNotIn("VAST_KEY", env)
        for inherited in (
            {},
            {"CUDA_INJECTION64_PATH": "/tmp/untrusted"},
            {"LD_PRELOAD": "/tmp/untrusted"},
        ):
            with self.assertRaises(ValueError):
                pilot.profile_environment(self.root, "candidate", inherited)

    def test_query_success_does_not_prove_permission(self):
        fake = self.root / "work/ncu/ncu"
        fake.parent.mkdir(parents=True)
        fake.write_text("synthetic")
        (self.root / "logs").mkdir()
        calls = []

        def fake_run(_root, _command, name, _budget, _env):
            calls.append(name)
            text = (
                (
                    "gpu__time_duration.sum dram__bytes.sum lts__t_bytes.sum "
                    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum "
                    "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum "
                    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"
                )
                if name.startswith("ncu-query-")
                else "ERR_NVGPUCTRPERM"
            )
            (self.root / "logs" / (name + ".log")).write_text(text)
            return {"passed": name.startswith("ncu-query-")}

        with (
            patch.object(pilot, "run_owned", fake_run),
            patch.object(pilot, "hardware_snapshot", return_value={}),
        ):
            now = time.time()
            result = pilot.counters(self.root, now, now + 3600, True)
        self.assertEqual(calls, ["ncu-query-permission", "ncu-permission"])
        self.assertFalse(result["counter_permission"])
        self.assertTrue(result["actual_probe_executed"])
        (self.root / "results/counter-permission.json").write_text(
            json.dumps(
                {
                    "counter_values_validated": True,
                    "application_metadata": {"l2_bytes": 6 << 20},
                }
            )
        )
        with (
            patch.object(pilot, "run_owned", fake_run),
            patch.object(pilot, "hardware_snapshot", return_value={}),
        ):
            pilot.counters(self.root, now, now + 3600, False)
        self.assertEqual(
            calls,
            [
                "ncu-query-permission",
                "ncu-permission",
                "ncu-query-current",
                "ncu-compute",
            ],
        )

    def test_counter_units_and_finite_values_required(self):
        good = '"Metric Name","Metric Unit","Metric Value"\n"gpu__time_duration.sum","nsecond","100"\n'
        value = pilot.legacy.validate_ncu_values(good, ["gpu__time_duration.sum"])
        self.assertTrue(
            pilot.legacy.ncu_roofline_inputs_present(value, {"gpu__time_duration.sum"})
        )
        for text in (good.replace("nsecond", ""), good.replace('"100"', '"nan"')):
            value = pilot.legacy.validate_ncu_values(text, ["gpu__time_duration.sum"])
            self.assertFalse(
                pilot.legacy.ncu_roofline_inputs_present(
                    value, {"gpu__time_duration.sum"}
                )
            )

    def test_profiler_main_call_lifecycle_on_success_and_exception(self):
        events = []
        bounds = []

        def start():
            events.append("start")
            return 0

        def stop():
            events.append("stop")
            return 0

        def push():
            events.append("push")
            return 0

        def pop():
            events.append("pop")

        def original():
            events.append("run")
            return 17

        self.assertEqual(
            pilot.profiled_call(original, start, stop, push, pop, bounds), 17
        )
        self.assertEqual(events, ["start", "push", "run", "pop", "stop"])
        events.clear()

        def failing():
            events.append("run")
            message = "original physics exception"
            raise ArithmeticError(message)

        with self.assertRaises(ArithmeticError):
            pilot.profiled_call(failing, start, stop, push, pop, bounds)
        self.assertEqual(events, ["start", "push", "run", "pop", "stop"])
        self.assertEqual(len(bounds), 2)

    def test_profiler_marker_failure_still_stops(self):
        events = []

        def start():
            events.append("start")
            return 0

        def stop():
            events.append("stop")
            return 0

        with self.assertRaises(RuntimeError):
            pilot.profiled_call(lambda: None, start, stop, lambda: -2, lambda: None, [])
        self.assertEqual(events, ["start", "stop"])

    @unittest.skipUnless(
        sys.version_info >= (3, 12), "Actual safe extraction requires Python3.12"
    )
    def test_archive_root_and_uncovered_collection_rejected(self):
        archive = self.root / "collection.tar.gz"
        data = b"approved"
        manifest = {"result.txt": hashlib.sha256(data).hexdigest()}
        with tarfile.open(archive, "w:gz") as target:
            info = tarfile.TarInfo(".")
            info.type = tarfile.DIRTYPE
            target.addfile(info)
            for name, content in (
                ("./result.txt", data),
                ("retrieval-manifest.json", json.dumps(manifest).encode()),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                target.addfile(info, io.BytesIO(content))
        self.assertTrue(
            pilot.verify_collection(
                archive, pilot.digest(archive), self.root / "verified"
            )["exact"]
        )
        with tarfile.open(archive, "w:gz") as target:
            for name, content in (
                ("result.txt", data),
                ("uncovered.txt", data),
                ("retrieval-manifest.json", json.dumps(manifest).encode()),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                target.addfile(info, io.BytesIO(content))
        with self.assertRaises(ValueError):
            pilot.verify_collection(
                archive, pilot.digest(archive), self.root / "bad-verified"
            )

    @unittest.skipUnless(
        sys.platform == "linux", "Owned process identity uses Linux procfs"
    )
    def test_adopted_worker_outlives_cli_and_timeout_cleanup(self):
        self.assertEqual(ctypes.CDLL(None).prctl(36, 1, 0, 0, 0), 0)
        env = {"PATH": os.environ["PATH"]}
        success = pilot.run_owned(
            self.root,
            [
                sys.executable,
                "-c",
                "import os,time; p=os.fork(); time.sleep(.2) if p==0 else None",
            ],
            "adopt",
            5,
            env,
        )
        self.assertTrue(success["passed"])
        self.assertTrue(success["owned_children_reaped"])
        self.assertTrue(success.get("adopted_reaped"))
        failure = pilot.run_owned(
            self.root,
            [sys.executable, "-c", "import os,time; os.fork(); time.sleep(20)"],
            "timeout",
            0.3,
            env,
        )
        self.assertFalse(failure["passed"])
        self.assertTrue(failure["cleanup_actions"])
        self.assertTrue(failure["owned_children_reaped"])

    def test_raw_numeric_and_diagnostic_drift_rejected(self):
        try:
            import numpy as np

            Dataset = importlib.import_module("netCDF4").Dataset
        except ImportError:
            self.skipTest("Run this control in pinned Python3.12 oracle environment")
        for name in ("a", "b"):
            directory = self.root / name
            directory.mkdir()
            with Dataset(directory / "raw_wout_0.nc", "w") as target:
                target.createDimension("n", 2)
                target.createVariable("metric", "f8", ("n",))[:] = [0.0, 1.0]
            np.savez(directory / "raw_diagnostics_0.npz", force=np.array([1.0, 2.0]))
        self.assertTrue(pilot.raw_compare(self.root / "a", self.root / "b")["exact"])
        np.savez(
            self.root / "b/raw_diagnostics_0.npz",
            force=np.array([1.0, np.nextafter(2.0, 3.0)]),
        )
        with self.assertRaises(ValueError):
            pilot.raw_compare(self.root / "a", self.root / "b")


if __name__ == "__main__":
    unittest.main()
