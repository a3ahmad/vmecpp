"""CPU-only regression checks for the isolated, deadline-bounded pilot tools."""

import contextlib
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.optimization import cloud_pilot
from tools.optimization.test_paired_campaign import fixture


def expect_error(operation, error_type):
    try:
        operation()
    except error_type:
        return
    message = "Expected pilot preflight to reject invalid input"
    raise AssertionError(message)


class CloudPilotTest(unittest.TestCase):
    def test_ncu_real_wide_csv_excerpts_keep_scaled_units(self):
        # Columns selected from residual-correction-input-fusion/profile/
        # baseline-values.csv and fused-values.csv, without changing units/values.
        header = '"ID","gpu__time_duration.sum","dram__bytes.sum","lts__t_bytes.sum","smsp__warps_eligible.avg.per_cycle_active"\n'
        units = '"","us","Kbyte","Mbyte","warp"\n'
        requested = [
            "gpu__time_duration.sum",
            "dram__bytes.sum",
            "lts__t_bytes.sum",
            "smsp__warps_eligible.avg.per_cycle_active",
        ]
        for data, expected in (
            (
                '"0","7.392000","6.144000","2.050176","0.079658"\n',
                [7.392, 6.144, 2.050176, 0.079658],
            ),
            (
                '"0","9.376000","2.048000","2.744160","0.102863"\n',
                [9.376, 2.048, 2.74416, 0.102863],
            ),
        ):
            result = cloud_pilot.validate_ncu_values(header + units + data, requested)
            assert result["exact"]
            assert result["units_complete"]
            assert [result["values"][name][0] for name in requested] == expected
            assert [result["units"][name][0] for name in requested] == [
                "us",
                "Kbyte",
                "Mbyte",
                "warp",
            ]
            assert cloud_pilot.ncu_roofline_inputs_present(result, set(requested))
            without_units = cloud_pilot.validate_ncu_values(header + data, requested)
            assert without_units["exact"]
            assert not without_units["units_complete"]
            assert not cloud_pilot.ncu_roofline_inputs_present(
                without_units, set(requested)
            )
            missing = cloud_pilot.validate_ncu_values(
                header + units + data, [*requested, "missing"]
            )
            assert not missing["exact"]
            assert missing["missing_metrics"] == ["missing"]

    def test_ncu_wide_invalid_later_kernel_cannot_hide_behind_valid_values(self):
        text = '"ID","gpu__time_duration.sum"\n"","us"\n"0","7.392000"\n"1","nan"\n'
        result = cloud_pilot.validate_ncu_values(text, ["gpu__time_duration.sum"])
        assert not result["exact"]
        assert result["values"] == {"gpu__time_duration.sum": [7.392]}
        assert result["invalid_metrics"] == ["gpu__time_duration.sum"]
        assert not cloud_pilot.ncu_roofline_inputs_present(
            result, {"gpu__time_duration.sum"}
        )
        truncated = text.rsplit('"1","nan"', 1)[0] + '"1"\n'
        assert not cloud_pilot.validate_ncu_values(
            truncated, ["gpu__time_duration.sum"]
        )["exact"]

    def test_ncu_long_csv_preserves_units_and_thousands(self):
        text = '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n"kernel","time","ns","1,024"\n"kernel","bytes","byte","4,096"\n'
        result = cloud_pilot.validate_ncu_values(text, ["time", "bytes"])
        assert result["exact"]
        assert result["values"] == {"time": [1024], "bytes": [4096]}
        assert result["units"] == {"time": ["ns"], "bytes": ["byte"]}
        assert cloud_pilot.ncu_roofline_inputs_present(result, {"time", "bytes"})

    def test_ncu_permission_error_precedes_missing_metric_classification(self):
        text = "gpu__time_duration.sum\n==ERROR== ERR_NVGPUCTRPERM - Permission denied"
        for code in (0, 1):
            reason = cloud_pilot.ncu_query_failure(code, text)
            assert reason is not None
            assert "ERR_NVGPUCTRPERM" in reason
        assert cloud_pilot.ncu_query_failure(0, "gpu__time_duration.sum") is None
        failure = cloud_pilot.ncu_query_failure(1, "other error")
        assert failure is not None
        assert "metric query failed" in failure
        values = cloud_pilot.validate_ncu_values(text, ["gpu__time_duration.sum"])
        assert not values["exact"]
        assert values["error"] == "ERR_NVGPUCTRPERM"

    def test_ncu_report_requires_actual_finite_requested_values(self):
        text = 'preamble\n"Kernel Name","Metric Name","Metric Value"\n"kernel","time","1,024"\n"kernel","bytes","nan"\n'
        result = cloud_pilot.validate_ncu_values(text, ["time", "bytes"])
        assert result["exact"] is False
        assert result["values"] == {"time": [1024.0]}
        assert result["missing_metrics"] == ["bytes"]
        assert cloud_pilot.validate_ncu_values(text, ["time"])["exact"]
        assert not cloud_pilot.validate_ncu_values("report created only", ["time"])[
            "exact"
        ]

    def test_diagnostic_trace_gate_preserves_output_checks_and_excludes_timings(self):
        report, capture = fixture()
        traced = json.loads(json.dumps(capture))
        traced["provenance"]["experimental_runtime"] = {
            "VMECPP_GPU_SOLVER_TRACE": "/pilot/results/candidate.gpu-trace.jsonl"
        }
        assert cloud_pilot.pair_gate(report, report, capture, traced)["exact"] is False
        gate = cloud_pilot.pilot_pair_gate(report, report, capture, traced)
        assert gate["exact"]
        assert gate["diagnostic_trace_present"]
        assert gate["main_table_eligible"] is False
        assert traced["provenance"]["experimental_runtime"]["VMECPP_GPU_SOLVER_TRACE"]
        changed = {**report, "metrics_sha256": "f" * 64}
        assert not cloud_pilot.pilot_pair_gate(report, changed, capture, traced)[
            "exact"
        ]

    def test_paths_cannot_escape_or_follow_external_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pilot"
            root.mkdir()
            expect_error(lambda: cloud_pilot.inside(root, "."), ValueError)
            for name in ("../outside", "/etc/passwd", "a/../../outside"):
                expect_error(
                    lambda name=name: cloud_pilot.inside(root, name), ValueError
                )
            (root / "external").symlink_to(Path(temporary), target_is_directory=True)
            expect_error(lambda: cloud_pilot.inside(root, "external/file"), ValueError)

    def test_payload_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input").write_text("frozen")
            expected = hashlib.sha256(b"frozen").hexdigest()
            (root / "payload-manifest.json").write_text(
                json.dumps({"files": {"input": expected}})
            )
            assert cloud_pilot.verify_payload(root)["files"]["input"] == expected
            (root / "input").write_text("changed")
            expect_error(lambda: cloud_pilot.verify_payload(root), ValueError)

    def test_dependency_archive_root_directory_structure_is_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "dependency.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                directory = tarfile.TarInfo(".")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
                data = b"pinned dependency"
                member = tarfile.TarInfo("./package/file")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if sys.version_info < (3, 12):
                expect_error(
                    lambda: cloud_pilot.safe_extract(archive_path, root / "out"),
                    RuntimeError,
                )
                return
            cloud_pilot.safe_extract(archive_path, root / "out")
            assert (root / "out/package/file").read_bytes() == data
            for kind in (tarfile.REGTYPE, tarfile.SYMTYPE, tarfile.LNKTYPE):
                bad = root / ("bad-" + kind.decode() + ".tar")
                with tarfile.open(bad, "w") as archive:
                    member = tarfile.TarInfo(".")
                    member.type = kind
                    member.linkname = "package/file"
                    archive.addfile(member)
                expect_error(
                    lambda bad=bad: cloud_pilot.safe_extract(bad, root / "bad-out"),
                    ValueError,
                )

    def test_deadline_reserves_retrieval_and_rejects_longer_leases(self):
        with mock.patch.object(cloud_pilot.time, "time", return_value=1000):
            cloud_pilot.check_lease_window(900, 4500)
            assert cloud_pilot.remaining_timeout(1700, 180) == 100
            expect_error(lambda: cloud_pilot.remaining_timeout(1599, 180), TimeoutError)
            expect_error(lambda: cloud_pilot.check_lease_window(900, 4501), ValueError)

    def test_environment_never_forwards_credentials_or_optional_libraries(self):
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "VAST_KEY": "dummy-test-only",
                "VAST_API_KEY": "dummy-test-only",
                "LD_PRELOAD": "/old.so",
                "VMECPP_OLD_FLAG": "1",
            },
            clear=True,
        ):
            baseline = cloud_pilot.environment(
                Path("/opt/vmecpp-pilot-test"), "baseline"
            )
            candidate = cloud_pilot.environment(
                Path("/opt/vmecpp-pilot-test"), "candidate"
            )
        for result in (baseline, candidate):
            assert "VAST_KEY" not in result
            assert "VAST_API_KEY" not in result
            assert "VMECPP_OLD_FLAG" not in result
            assert "optional" not in result["LD_LIBRARY_PATH"]
        assert "LD_PRELOAD" not in baseline
        assert candidate["VMECPP_GPU_SOLVER_GRAPH"] == "1"

    def test_collect_hashes_point_in_time_files_and_excludes_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            (root / "results").mkdir()
            (root / "logs/finished.log").write_text("done")
            (root / "results/report.json").write_text('{"exact":true}')
            (root / ".env").write_text("dummy-test-only")
            result = cloud_pilot.collect(root)
            assert result["files"] == 2
            with tarfile.open(result["archive"]) as archive:
                manifest_file = archive.extractfile("retrieval-manifest.json")
                assert manifest_file is not None
                manifest = json.load(manifest_file)
                assert ".env" not in manifest
                for name, expected in manifest.items():
                    captured = archive.extractfile(name)
                    assert captured is not None
                    assert hashlib.sha256(captured.read()).hexdigest() == expected

    def test_bootstrap_is_dry_without_execute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(
                    sys, "argv", ["cloud_pilot", "bootstrap", "--root", str(root)]
                ),
                mock.patch.object(cloud_pilot, "bootstrap") as execute,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                assert cloud_pilot.main() == 0
            assert not execute.called
            assert list(root.iterdir()) == []

    def test_profile_requires_real_final_grid_success_and_matching_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "marker.jsonl"
            assert not cloud_pilot.final_grid_marker(marker, 123)["exact"]
            good = {
                "event": "profiler_start",
                "pid": 123,
                "actual_ns": 99,
                "target_ns": 99,
                "status": 0,
            }
            marker.write_text(json.dumps(good) + "\n")
            assert cloud_pilot.final_grid_marker(marker, 123)["exact"]
            for key, value in (("actual_ns", 51), ("status", 999), ("pid", 456)):
                bad = {**good, key: value}
                marker.write_text(json.dumps(bad) + "\n")
                assert not cloud_pilot.final_grid_marker(marker, 123)["exact"]
            marker.write_text("not-json\n")
            assert not cloud_pilot.final_grid_marker(marker, 123)["exact"]

    def test_global_parity_failure_prevents_candidate_and_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud_pilot.save(root / "bootstrap-status.json", {"phase": "complete"})
            cloud_pilot.save(
                root / "fixtures/manifest.json",
                [
                    {"path": name + ".json"}
                    for name in ("public_p3_seed", "public_p2_seed", "third")
                ],
            )
            cloud_pilot.save(
                root / "pilot-plan.json",
                {
                    "prior_baseline_case_medians_seconds": [
                        {"seed": "public_p3_seed.json", "track": "P3", "seconds": 140}
                    ]
                },
            )
            report, capture = fixture()
            reference = root / "references/public_p3_seed-P3"
            cloud_pilot.save(reference / "report.json", report)
            cloud_pilot.save(reference / "provenance.json", capture["provenance"])
            cloud_pilot.save(
                reference / "actual_indata_0.json", capture["actual_indata"][0]
            )
            launched = []

            def launch(_root, command, name, *_args, **_kwargs):
                launched.append(name)
                if name == "gpu-inventory":
                    (root / "logs").mkdir(exist_ok=True)
                    (root / "logs/gpu-inventory.log").write_text("A100,driver,memory\n")
                elif name == "strict-calibration":
                    cloud_pilot.save(root / "logs/strict-calibration.log", {})
                elif name == "pilot-0-baseline":
                    output = Path(command[command.index("--out") + 1])
                    cloud_pilot.save(
                        output / "report.json", {**report, "metrics_sha256": "f" * 64}
                    )
                    cloud_pilot.save(output / "provenance.json", capture["provenance"])
                    cloud_pilot.save(
                        output / "actual_indata_0.json", capture["actual_indata"][0]
                    )
                return 0

            with (
                mock.patch.object(cloud_pilot, "verify_payload"),
                mock.patch.object(
                    cloud_pilot,
                    "make_wrappers",
                    return_value={
                        "baseline": Path("baseline"),
                        "candidate": Path("candidate"),
                    },
                ),
                mock.patch.object(cloud_pilot, "launch", side_effect=launch),
                mock.patch.object(cloud_pilot, "profiles") as profiles,
            ):
                expect_error(
                    lambda: cloud_pilot.run_pilot(
                        root, cloud_pilot.time.time() + 3500, 1
                    ),
                    RuntimeError,
                )
            assert "pilot-0-candidate" not in launched
            assert not profiles.called
            assert (
                json.loads((root / "pilot-status.json").read_text())["phase"]
                == "failed"
            )


if __name__ == "__main__":
    unittest.main()
