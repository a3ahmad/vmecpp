"""CPU-only image-pinning and preflight checks; no provider or GPU access."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.optimization import lease


def expect_error(operation, error_type):
    try:
        operation()
    except error_type:
        return
    message = "Expected lease preflight rejection"
    raise AssertionError(message)


class LeaseImageTest(unittest.TestCase):
    def test_pinned_image_is_used_by_create_and_recorded(self):
        digest = "sha256:" + "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / ".ssh/id_ed25519.pub"
            public.parent.mkdir()
            public.write_text("ssh-ed25519 TEST-PUBLIC-KEY\n")
            state = root / "lease.json"
            api = mock.Mock()
            api.show_instances.return_value = []
            api.search_offers.return_value = [
                {"id": 123, "dph_total": 0.5, "gpu_name": "A100 PCIE"}
            ]
            api.create_instance.return_value = {"new_contract": 456}
            arguments = [
                "lease.py",
                "create",
                "--state",
                str(state),
                "--hours",
                "1",
                "--image-digest",
                digest,
            ]
            with (
                mock.patch("sys.argv", arguments),
                mock.patch.object(lease, "client", return_value=api),
                mock.patch.object(lease.subprocess, "Popen") as launch,
                mock.patch.object(lease.Path, "home", return_value=root),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch.return_value.pid = 999
                assert lease.main() == 0
            expected = "vastai/base-image@" + digest
            assert api.create_instance.call_args.kwargs["image"] == expected
            assert json.loads(state.read_text())["image"] == expected
            api.attach_ssh.assert_called_once_with(456, "ssh-ed25519 TEST-PUBLIC-KEY")

    def test_invalid_digest_never_contacts_provider(self):
        invalid = ["latest", "sha256:" + "a" * 63, "sha256:" + "G" * 64]
        for digest in invalid:
            with (
                self.subTest(digest=digest),
                mock.patch(
                    "sys.argv",
                    [
                        "lease.py",
                        "create",
                        "--state",
                        "unused",
                        "--image-digest",
                        digest,
                    ],
                ),
                mock.patch.object(lease, "client") as client,
            ):
                expect_error(lease.main, lease.LeasePreflightError)
                client.assert_not_called()

    def test_image_option_is_not_accepted_for_destroy(self):
        with (
            mock.patch(
                "sys.argv",
                [
                    "lease.py",
                    "destroy",
                    "--state",
                    "unused",
                    "--image-digest",
                    "sha256:" + "a" * 64,
                ],
            ),
            mock.patch.object(lease, "client") as client,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            expect_error(lease.main, SystemExit)
            client.assert_not_called()

    def test_legacy_image_default_is_preserved(self):
        assert (
            lease.pilot_image(None)
            == "vastai/base-image:cuda-12.9-mini-py312-2026-10-23"
        )


if __name__ == "__main__":
    unittest.main()
