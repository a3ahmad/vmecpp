"""CPU-only injection, lifecycle, numerical-gate and detached-worker regressions."""

import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tools.optimization import batch_nsys_diagnostic as diagnostic
from tools.optimization.test_batch_throughput import expect_error


class DiagnosticTests(unittest.TestCase):
    def test_injection_allowlist_requires_exact_path_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "frozen.so"
            other = root / "other.so"
            library.write_bytes(b"frozen library fixture")
            other.write_bytes(library.read_bytes())
            expected = diagnostic.digest(library)
            with (
                mock.patch.object(diagnostic, "INJECTION", library),
                mock.patch.object(diagnostic, "INJECTION_SHA256", expected),
            ):
                assert diagnostic.diagnostic_injection_allowed(
                    "CUDA_INJECTION64_PATH", str(library)
                )
                assert diagnostic.diagnostic_injection_allowed(
                    "NVTX_INJECTION64_PATH", str(library)
                )
                assert not diagnostic.diagnostic_injection_allowed(
                    "UNRELATED_INJECTION", str(library)
                )
                assert not diagnostic.diagnostic_injection_allowed(
                    "CUDA_INJECTION64_PATH", str(other)
                )
                library.write_bytes(b"changed")
                assert not diagnostic.diagnostic_injection_allowed(
                    "CUDA_INJECTION64_PATH", str(library)
                )

    def test_copy_is_explicit_and_canonical_guard_is_unchanged(self):
        path = diagnostic.ROOT / "tools/optimization/batch_throughput.py"
        before = path.read_text()
        copied = diagnostic.diagnostic_source(before)
        compile(copied, "diagnostic-copy", "exec")
        assert path.read_text() == before
        assert '"profiled": True' in copied
        assert "args.correctness_only = True" in copied
        assert "not diagnostic_injection_allowed(name, value)" in copied
        assert "diagnostic_injection_allowed" not in before
        with expect_error(RuntimeError):
            diagnostic.diagnostic_source(
                before.replace('"profiled": False,', "changed")
            )

    def test_profile_lifecycle_brackets_drained_pool_and_failure(self):
        events = []

        def pool():
            events.append("pool_create")
            thread = threading.Thread(target=lambda: events.append("future_terminal"))
            thread.start()
            thread.join()
            events.append("pool_shutdown")
            return "result"

        wrapped = diagnostic.profile_lifecycle(
            pool,
            lambda: 0,
            lambda: 0,
            lambda event, status: events.append((event, status)),
        )
        assert wrapped() == "result"
        assert events == [
            ("start_before_pool", 0),
            "pool_create",
            "future_terminal",
            "pool_shutdown",
            ("stop_after_drained_pool", 0),
        ]

        def failed_pool():
            events.append("failed_pool_drained")
            message = "original error"
            raise ValueError(message)

        events.clear()
        wrapped = diagnostic.profile_lifecycle(
            failed_pool,
            lambda: 0,
            lambda: 0,
            lambda event, status: events.append((event, status)),
        )
        with expect_error(ValueError):
            wrapped()
        assert events == [
            ("start_before_pool", 0),
            "failed_pool_drained",
            ("stop_after_drained_pool", 0),
        ]

    def test_profile_rejects_failed_start_and_non_main_thread(self):
        events = []
        wrapped = diagnostic.profile_lifecycle(
            lambda: events.append("pool"),
            lambda: 5,
            lambda: events.append("stop"),
            lambda event, status: events.append((event, status)),
        )
        with expect_error(RuntimeError):
            wrapped()
        assert events == [("start_before_pool", 5)]
        caught = []

        def other_thread():
            with expect_error(RuntimeError):
                wrapped()
            caught.append(True)

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert caught == [True]

    def test_profiled_numerical_gate_requires_every_identity_and_actual_deck(self):
        field = {"shape": [1], "dtype": "float64", "sha256": "1" * 64}
        baseline = {
            "status": "SUCCESS",
            "metrics": {"metric": 1.0},
            "metrics_sha256": "2" * 64,
            "request_metrics_sha256": "3" * 64,
            "numerical_wout_hashes": [{"field": field}],
            "numerical_diagnostic_hashes": [
                {
                    name: {"field": field}
                    for name in ("jxbout", "mercier", "threed1_volumetrics")
                }
            ],
            "profiled": False,
        }
        candidate = {**baseline, "profiled": True}
        capture = {"actual_indata": [{"mpol": 10}]}
        assert diagnostic.diagnostic_pair_gate(baseline, candidate, capture, capture)[
            "exact"
        ]
        assert not diagnostic.diagnostic_pair_gate(
            baseline, baseline, capture, capture
        )["exact"]
        assert not diagnostic.diagnostic_pair_gate(
            baseline, candidate, capture, {"actual_indata": [{"mpol": 12}]}
        )["exact"]
        changed = copy.deepcopy(candidate)
        changed["numerical_diagnostic_hashes"][0]["mercier"]["field"]["sha256"] = (
            "4" * 64
        )
        assert not diagnostic.diagnostic_pair_gate(baseline, changed, capture, capture)[
            "exact"
        ]

    def supervisor_probe(self, directory, worker_sleep, timeout):
        worker = (
            "import json,os,time;from pathlib import Path;p=Path("
            + repr(str(directory / "worker-status.json"))
            + ");p.write_text(json.dumps({'phase':'running','pid':os.getpid()}));time.sleep("
            + str(worker_sleep)
            + ");p.write_text(json.dumps({'phase':'terminal','exit_status':0,'finished_at':time.time()}))"
        )
        cli = (
            "import subprocess,sys;subprocess.Popen([sys.executable,'-c',"
            + repr(worker)
            + "],start_new_session=True)"
        )
        program = (
            "from pathlib import Path;import os,sys;from tools.optimization.batch_nsys_diagnostic import supervise;supervise([sys.executable,'-c',"
            + repr(cli)
            + "],dict(os.environ),Path("
            + repr(str(directory))
            + "),"
            + str(timeout)
            + ")"
        )
        return subprocess.run(
            [sys.executable, "-B", "-c", program],
            cwd=diagnostic.ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    @unittest.skipUnless(Path("/proc").is_dir(), "Linux process-ownership regression")
    def test_cli_early_exit_does_not_end_worker_supervision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.supervisor_probe(root, 0.7, 5)
            assert result.returncode == 0, result.stderr
            status = json.loads((root / "supervisor-status.json").read_text())
            worker = json.loads((root / "worker-status.json").read_text())
            assert status["owned_children_reaped"]
            assert worker["phase"] == "terminal"
            assert status["cli_finished_at"] < worker["finished_at"]

    @unittest.skipUnless(Path("/proc").is_dir(), "Linux process-ownership regression")
    def test_deadline_kills_owned_detached_worker_after_cli_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.supervisor_probe(root, 10, 0.3)
            assert result.returncode != 0
            status = json.loads((root / "supervisor-status.json").read_text())
            worker = json.loads((root / "worker-status.json").read_text())
            assert status["deadline_exceeded"]
            assert status["owned_children_reaped"]
            assert status["cli_exit_status"] == 0
            assert not Path(f"/proc/{worker['pid']}").exists()


if __name__ == "__main__":
    unittest.main()
