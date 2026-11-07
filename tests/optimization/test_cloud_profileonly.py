"""Strict profiler forwarding and diagnostic-only completion contracts."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.optimization import cloud_pilot_profileonly_controller as controller
from tools.optimization import nsys_injection_guard as guard

# The stdlib runner is required in isolated CPU-only preflight environments.
# ruff: noqa: PT027


class ProfileOnlyContracts(unittest.TestCase):
    def guard_fixture(self, root):
        directory = root / "work/nsys/target-linux-x64"
        directory.mkdir(parents=True)
        hashes = dict((guard.ROUTING_LIBRARY, *guard.ORDERED_PRELOAD_LIBRARIES))
        for name in hashes:
            (directory / name).write_text("unit-only frozen hash stand-in")
        shim = root / "work/lib/solver-shim.so"
        shim.parent.mkdir(parents=True)
        shim.touch()
        base = {"LD_PRELOAD": str(shim), "OMP_NUM_THREADS": "1"}
        inherited = {
            key: str(directory / guard.ROUTING_LIBRARY[0])
            for key in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH")
        }
        inherited["LD_PRELOAD"] = ":".join(
            str(directory / name) for name, _ in guard.ORDERED_PRELOAD_LIBRARIES
        )
        inherited["NSYS_SESSION_TEST"] = "session routing"
        inherited["UNAPPROVED_ENV"] = "must not propagate"
        return base, inherited, hashes

    def test_exact_ordered_proxies_precede_only_canonical_shim(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, inherited, hashes = self.guard_fixture(root)
            with patch.object(
                guard, "digest", side_effect=lambda path: hashes[path.name]
            ):
                result = guard.compose_profile_environment(root, base, inherited)
            assert (
                result["LD_PRELOAD"]
                == inherited["LD_PRELOAD"] + ":" + base["LD_PRELOAD"]
            )
            assert result["NSYS_SESSION_TEST"] == "session routing"
            assert "UNAPPROVED_ENV" not in result
            assert result["CUDA_INJECTION64_PATH"] == inherited["CUDA_INJECTION64_PATH"]

    def test_extra_reordered_missing_relative_preloads_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, inherited, hashes = self.guard_fixture(root)
            variants = [
                inherited["LD_PRELOAD"] + ":/unapproved.so",
                ":".join(reversed(inherited["LD_PRELOAD"].split(":"))),
                "",
                "libToolsInjectionProxy64.so:libLinuxKeyboardInterceptorProxy.so",
            ]
            for preload in variants:
                with (
                    self.subTest(preload=preload),
                    patch.object(
                        guard, "digest", side_effect=lambda path: hashes[path.name]
                    ),
                    self.assertRaisesRegex(ValueError, "preload set or order"),
                ):
                    guard.compose_profile_environment(
                        root, base, dict(inherited, LD_PRELOAD=preload)
                    )

    def test_mismatched_library_route_or_application_shim_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, inherited, hashes = self.guard_fixture(root)
            with (
                patch.object(guard, "digest", return_value="0" * 64),
                self.assertRaisesRegex(ValueError, "missing or mismatched"),
            ):
                guard.compose_profile_environment(root, base, inherited)
            with patch.object(
                guard, "digest", side_effect=lambda path: hashes[path.name]
            ):
                with self.assertRaisesRegex(ValueError, "routing path"):
                    guard.compose_profile_environment(
                        root,
                        base,
                        dict(inherited, CUDA_INJECTION64_PATH="/unapproved.so"),
                    )
                with self.assertRaisesRegex(ValueError, "canonical application shim"):
                    guard.compose_profile_environment(
                        root,
                        dict(base, LD_PRELOAD=base["LD_PRELOAD"] + ":/extra.so"),
                        inherited,
                    )

    def valid_diagnostic(self):
        return {
            "phase": "diagnostic_complete",
            "diagnostic_complete": True,
            "diagnostic_mode": "anchor-plus-profile",
            "prior_canonical_evidence_sha256": "a" * 64,
            "completed_diagnostic_evaluations": [
                {"label": "anchor", "profiled": False, "gate": {"exact": True}},
                {"label": "profiled", "profiled": True, "gate": {"exact": True}},
            ],
            "anchor_profile_raw_comparison": {
                "exact": True,
                "numeric_fields": [{"field": "real closure exercised separately"}],
            },
            "nsys_report": dict.fromkeys(
                (
                    "created",
                    "target_worker_terminal_reaped",
                    "sqlite_export_passed",
                    "actual_worker_kernel_node_evidence",
                ),
                True,
            ),
            "owned_children_reaped": True,
            "runtime_unchanged": True,
            "immutable_payload_unchanged": True,
        }

    def test_two_diagnostic_contract_has_positive_control(self):
        assert controller.diagnostic_success(self.valid_diagnostic(), "a" * 64)

    def test_missing_false_or_canonical_fiction_cannot_complete(self):
        positive = self.valid_diagnostic()
        for key in positive:
            value = copy.deepcopy(positive)
            del value[key]
            assert not controller.diagnostic_success(value, "a" * 64)
        for item in (0, 1):
            value = copy.deepcopy(positive)
            value["completed_diagnostic_evaluations"][item]["gate"]["exact"] = False
            assert not controller.diagnostic_success(value, "a" * 64)
        for key in ("case_pairs", "canonical_parity_complete"):
            value = dict(positive, **{key: []})
            assert not controller.diagnostic_success(value, "a" * 64)
        assert not controller.diagnostic_success(positive, "b" * 64)


if __name__ == "__main__":
    unittest.main()
