# ruff: noqa: PT027
"""CPU-only contracts for explicit counter-free timing diagnostics."""

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.optimization import cloud_pilot_hardware as pilot


class HardwareContracts(unittest.TestCase):
    def test_outer_worker_persists_only_after_body_and_final_bookend(self):
        events = []

        def body(*_args):
            events.append("body")
            return 0

        def save(_path, value):
            events.append("save")
            assert len(value["records"]) == 1
            assert value["records"][0]["raw_elapsed_lower_ns"] >= 0

        with (
            patch.object(pilot, "worker_body", side_effect=body),
            patch.object(pilot, "save", side_effect=save),
        ):
            assert (
                pilot.worker(Path("/unused"), "baseline", ["--out", "/unused/result"])
                == 0
            )
        assert events == ["body", "save"]

    def test_clocked_call_has_ordered_integer_bookends_without_io(self):
        before = {
            "tid": 7,
            "raw_before_ns": 100,
            "raw_after_ns": 102,
            "monotonic_ns": 90,
        }
        after = {
            "tid": 7,
            "raw_before_ns": 300,
            "raw_after_ns": 302,
            "monotonic_ns": 280,
        }
        events = []
        records = []

        def reading():
            events.append("read")
            return before if len(events) == 1 else after

        def function():
            events.append("call")
            assert records == []
            return 42

        with (
            patch.object(pilot, "clock_reading", side_effect=reading),
            patch.object(pilot, "save") as save,
        ):
            assert pilot.clocked_call(function, records) == 42
        assert events == ["read", "call", "read"]
        save.assert_not_called()
        assert records[0]["raw_elapsed_lower_ns"] == 198
        assert records[0]["monotonic_elapsed_ns"] == 190
        assert records[0]["monotonic_within_raw_read_bounds"] is False
        with self.assertRaisesRegex(RuntimeError, "changed TID"):
            pilot.clock_pair(before, dict(after, tid=8))
        with self.assertRaisesRegex(RuntimeError, "inverted"):
            pilot.clock_pair(after, before)

    def test_clocked_call_preserves_exception_and_records_end(self):
        records = []
        with self.assertRaisesRegex(ArithmeticError, "sentinel"):
            pilot.clocked_call(
                lambda: (_ for _ in ()).throw(ArithmeticError("sentinel")), records
            )
        assert len(records) == 1

    def test_loader_fails_closed_on_environment_or_missing_library(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "loader environment"):
                pilot.verify_calibration_loader(root, {})
            env = {
                "LD_LIBRARY_PATH": str(root / "runtime") + ":" + str(root / "work/lib")
            }
            with (
                patch.object(
                    pilot, "config", return_value={"expected_core_sha256": {}}
                ),
                patch.object(pilot, "locate_tool", return_value=Path("nsys")),
                patch.object(pilot.subprocess, "run") as run,
            ):
                run.return_value.returncode = 0
                run.return_value.stdout = "libcudart.so.12 => not found"
                run.return_value.stderr = ""
                with self.assertRaisesRegex(RuntimeError, "missing dependency"):
                    pilot.verify_calibration_loader(root, env)
                assert run.call_args.kwargs["env"] == env

    def test_policy_is_explicit(self):
        for value in ({}, {"counter_policy": "required"}, {"counter_policy": True}):
            with self.assertRaisesRegex(ValueError, "policy required"):
                pilot.require_hardware_policy(value)
        pilot.require_hardware_policy({"counter_policy": "timing-and-trace"})

    def test_case_marker_requires_all_three_actual_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / str(index) for index in range(3)]
            for path in paths:
                path.mkdir()
                pilot.save(path / "original-global-gate.json", {"exact": True})
            assert pilot.case_gate(
                paths[0], paths[1], paths[2], {"exact": True}, {"exact": True}
            )["exact"]
            for missing in (False, True):
                pilot.save(
                    paths[2] / "original-global-gate.json",
                    {} if missing else {"exact": False},
                )
                with self.assertRaises(RuntimeError):
                    pilot.case_gate(
                        paths[0], paths[1], paths[2], {"exact": True}, {"exact": True}
                    )
            (paths[2] / "original-global-gate.json").unlink()
            with self.assertRaises(FileNotFoundError):
                pilot.case_gate(
                    paths[0], paths[1], paths[2], {"exact": True}, {"exact": True}
                )

    def test_kernel_interval_requires_one_verified_row(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "trace.sqlite"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    "CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,demangledName INTEGER); INSERT INTO StringIds VALUES(1,'void StrictMixed(double*)'); INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,250,123,1);"
                )
            application = {
                "all_output_words_verified": True,
                "profiled_graph_launches": 1,
                "warmup_graph_launches": 3,
                "l2_bytes": 40 * 1024**2,
                "mode": "compute",
            }
            result = pilot.calibration_interval(database, "StrictMixed", application)
            assert result["duration_ns"] == 150
            assert result["roofline_claim"] is False
            assert "FLOP_per_second" not in result
            assert "physical_bytes" not in result
            with self.assertRaisesRegex(ValueError, "metadata required"):
                pilot.calibration_interval(
                    database,
                    "StrictMixed",
                    dict(application, all_output_words_verified=False),
                )
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(251,350,123,1)"
                )
            with self.assertRaisesRegex(ValueError, "one captured"):
                pilot.calibration_interval(database, "StrictMixed", application)

    def test_denied_counters_continue_only_through_all_identity_gates(self):
        self.run_mock_pilot(False)

    def test_calibration_schedule_uses_hardware_sizes_and_three_verified_traces(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "work/bin/roofline-matched"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"frozen binary stand-in")
            commands = []

            def run_owned(_root, command, label, _budget, _env):
                commands.append(command)
                output = Path(
                    next(
                        item.split("=", 1)[1]
                        for item in command
                        if item.startswith("--output=")
                    )
                )
                if "profile" in command:
                    output.with_suffix(".nsys-rep").touch()
                    mode, parameter = command[-2:]
                    application = {
                        "mode": mode,
                        "parameter": int(parameter),
                        "all_output_words_verified": True,
                        "l2_bytes": 40 * 1024**2,
                        "working_set_bytes": int(parameter) * 1024**2,
                        "profiled_graph_launches": 1,
                        "warmup_graph_launches": 3,
                    }
                    pilot.save(root / "logs" / (label + ".log"), application)
                else:
                    kernel = "StrictMixed" if "compute" in label else "CopyVectors"
                    with sqlite3.connect(output) as connection:
                        connection.executescript(
                            "CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,demangledName INTEGER);"
                        )
                        connection.execute(
                            "INSERT INTO StringIds VALUES(1,?)", (kernel,)
                        )
                        connection.execute(
                            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,30,123,1)"
                        )
                return {"passed": True, "command": command}

            with (
                patch.object(pilot, "run_owned", side_effect=run_owned),
                patch.object(pilot, "locate_tool", return_value=Path("nsys")),
                patch.object(pilot, "verify_calibration_loader"),
            ):
                # Production application emits one compact JSON line.
                original_save = pilot.save

                def compact_save(path, value):
                    if path.suffix == ".log":
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(value) + "\n")
                    else:
                        original_save(path, value)

                with patch.object(pilot, "save", side_effect=compact_save):
                    results = pilot.calibration_timing(
                        root, time.time(), time.time() + 2000
                    )
            assert len(results) == 3
            assert [item["application_metadata"]["parameter"] for item in results] == [
                128,
                20,
                256,
            ]
            assert all(item["counter_evidence"] is False for item in results)
            assert all(command[0] == "nsys" for command in commands)

    def test_denied_counters_do_not_bypass_metric_failure(self):
        self.run_mock_pilot(True)

    def run_mock_pilot(self, fail_metric):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pilot.save(
                root / "bootstrap-status.json",
                {"phase": "complete", "runtime_files": {}},
            )
            pilot.save(root / "payload-manifest.json", {"files": {}})
            pilot.save(
                root / "fixtures/manifest.json",
                [
                    {"path": name}
                    for name in ("public_p3_seed.json", "b.json", "c.json")
                ],
            )
            calls = []

            def evaluate(_root, _seed, _track, variant, label, *_args):
                calls.append(variant)
                if fail_metric and variant == "candidate":
                    msg = "Actual metric drift"
                    raise RuntimeError(msg)
                output = root / "results" / label
                pilot.save(output / "original-global-gate.json", {"exact": True})
                pilot.save(output / "activation.json", {"pid": 1})
                if label == "nsys-current-full-vmec":
                    target = root / "results/profiles/timeline.nsys-rep"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("profile")
                return output

            with (
                patch.object(pilot, "preflight"),
                patch.object(pilot, "selected_cpu_environment", return_value={}),
                patch.object(
                    pilot.os, "sched_getaffinity", return_value={0}, create=True
                ),
                patch.object(pilot.os, "sched_setaffinity", create=True),
                patch.object(pilot, "hardware_snapshot"),
                patch.object(
                    pilot,
                    "counters",
                    return_value={
                        "counter_permission": False,
                        "reason": "ERR_NVGPUCTRPERM",
                    },
                ) as counters,
                patch.object(pilot, "evaluate", side_effect=evaluate),
                patch.object(pilot, "raw_compare", return_value={"exact": True}),
                patch.object(
                    pilot,
                    "calibration_timing",
                    return_value=[{"counter_evidence": False}],
                ),
                patch.object(pilot, "locate_tool", return_value=Path("nsys")),
                patch.object(pilot, "run_owned", return_value={"passed": True}),
                patch.object(pilot, "descendants", return_value=[]),
                patch(
                    "tools.optimization.analyze_full_vmec_profile.analyze",
                    return_value={"kernels": ["actual"]},
                ),
            ):
                code = pilot.run_pilot(root, time.time(), time.time() + 2000, 6)
            state = json.loads((root / "pilot-status.json").read_text())
            counters.assert_called_once()
            assert (
                state["actual_counter_permission_gate"]["counter_permission"] is False
            )
            if fail_metric:
                assert code == 1
                assert state["phase"] == "failed"
                assert not state["canonical_parity_complete"]
            else:
                assert code == 0
                assert state["phase"] == "complete"
                assert len(state["case_pairs"]) == 6
                assert calls.count("baseline") == 6
                assert calls.count("candidate") == 13
                assert all(item["gate"]["exact"] for item in state["case_pairs"])
                assert not state["counters"]["executed"]
                assert not state["main_table_eligible"]


if __name__ == "__main__":
    unittest.main()
