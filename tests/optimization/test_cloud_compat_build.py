"""Keep compatibility preparation separate from frozen payloads and legacy probes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.optimization import build_cloud_compat as builder
from tools.optimization import cloud_pilot_compat_controller as wrapper

# Stdlib controls run without pytest in the pinned environment.
# ruff: noqa: PT027


class CompatibilityPreparationTests(unittest.TestCase):
    def test_output_cannot_overlap_frozen_base(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "frozen"
            with mock.patch.object(builder.current, "preflight") as preflight:
                with self.assertRaisesRegex(ValueError, "separate from immutable"):
                    builder.build(base / "new", base, Path(directory) / "probe")
                preflight.assert_not_called()

    def test_probe_manifest_traversal_rejected_before_payload_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, probe = root / "base", root / "probe"
            base.mkdir()
            probe.mkdir()
            (base / "payload-manifest.json").write_text(json.dumps({"files": {}}))
            (probe / "candidate-manifest.json").write_text(
                json.dumps({"../escape": "a"})
            )
            (probe / "probe-ready.json").write_text(
                json.dumps(
                    {
                        "CPU_only_private_probe_ready": True,
                        "prospective_cloud_executed": False,
                        "manifest_sha256": builder.pilot.digest(
                            probe / "candidate-manifest.json"
                        ),
                    }
                )
            )
            with (
                mock.patch.object(builder.current, "preflight"),
                mock.patch.object(builder.shutil, "copytree") as copy,
            ):
                with self.assertRaisesRegex(ValueError, "Probe path/hash mismatch"):
                    builder.build(root / "out", base, probe)
                copy.assert_not_called()

    def test_compat_controller_rejects_legacy_optional_work(self):
        with (
            mock.patch.object(
                wrapper.sys, "argv", ["controller", "--include-optional-tools"]
            ),
            mock.patch.object(wrapper.cloud_pilot_controller, "main") as main,
        ):
            with self.assertRaisesRegex(ValueError, "excludes legacy"):
                wrapper.main()
            main.assert_not_called()

    def test_default_verifier_is_new_compatibility_tool(self):
        argv = ["controller", "--plan"]
        with (
            mock.patch.object(wrapper.sys, "argv", argv),
            mock.patch.object(wrapper.cloud_pilot_controller, "main", return_value=0),
        ):
            assert wrapper.main() == 0
        assert (
            Path(argv[argv.index("--cloud-pilot-tool") + 1]).name
            == "cloud_pilot_compat.py"
        )
