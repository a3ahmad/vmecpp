"""Direct-child profiling validates inherited state without rewriting it."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.optimization import cloud_pilot_profileonly3 as pilot
from tools.optimization import nsys_direct_guard as guard


class DirectGuard(unittest.TestCase):
    def fixture(self, root):
        directory = root / "work/nsys/target-linux-x64"
        directory.mkdir(parents=True)
        hashes = {}
        for name, sha in (guard.ROUTING_LIBRARY, *guard.ORDERED_PRELOAD_LIBRARIES):
            (directory / name).write_bytes(b"fixture")
            hashes[name] = sha
        shim = root / "work/lib/solver-shim.so"
        shim.parent.mkdir(parents=True)
        shim.write_bytes(b"fixture")
        hashes[shim.name] = "a" * 64
        provider = root / "provider.so"
        provider.write_bytes(b"fixture")
        hashes[provider.name] = guard.REFERENCE_SHA
        expected = {
            key: value.replace("{extension}", str(provider))
            for key, value in pilot.REQUIRED_FLAGS.items()
        }
        expected.update(
            PYTHONPATH=str(root),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CGA_OPTIMIZATION_WRAPPER_SHA256="b" * 64,
            CGA_OPTIMIZATION_CONFIGURATION_SHA256="c" * 64,
        )
        paths = [
            shim,
            *(directory / name for name, _ in guard.ORDERED_PRELOAD_LIBRARIES),
        ]
        actual = dict(
            expected,
            LD_PRELOAD=":".join(map(str, paths)),
            CUDA_INJECTION64_PATH=str(directory / guard.ROUTING_LIBRARY[0]),
            NVTX_INJECTION64_PATH=str(directory / guard.ROUTING_LIBRARY[0]),
            NSYS_PROFILING_SESSION_ID="fixture",
            OTHER_PRIVATE_KEY="preserved",
        )
        maps = "\n".join("0-1 r-xp 0 00:00 1 " + str(path) for path in paths)
        return expected, actual, hashes, maps

    def run_guard(self, root, expected, actual, hashes, maps):
        with (
            patch.object(guard, "digest", side_effect=lambda p: hashes[p.name]),
            patch.object(Path, "read_text", return_value=maps),
        ):
            return guard.validate_direct_environment(
                root, expected, actual, {"work/lib/solver-shim.so": "a" * 64}
            )

    def test_exact_appfirst_positive_and_no_environment_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected, actual, hashes, maps = self.fixture(root)
            original = copy.deepcopy(actual)
            assert self.run_guard(root, expected, actual, hashes, maps)["exact"]
            assert actual == original

    def test_proxy_first_missing_flag_and_wrong_route_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected, actual, hashes, maps = self.fixture(root)
            assert self.run_guard(root, expected, actual, hashes, maps)["exact"]
            values = actual["LD_PRELOAD"].split(":")
            cases = [
                dict(actual, LD_PRELOAD=":".join([*values[1:], values[0]])),
                dict(actual, CUDA_INJECTION64_PATH="/outside.so"),
                {k: v for k, v in actual.items() if k != "VMECPP_GPU_SOLVER"},
            ]
            for value in cases:
                with self.assertRaisesRegex(ValueError, "APPFIRST|routing|nine"):  # noqa: PT027
                    self.run_guard(root, expected, value, hashes, maps)

    def test_provider_or_library_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected, actual, hashes, maps = self.fixture(root)
            for name in ("provider.so", guard.ROUTING_LIBRARY[0]):
                bad = dict(hashes, **{name: "0" * 64})
                with self.assertRaisesRegex(ValueError, "provider|hash"):  # noqa: PT027
                    self.run_guard(root, expected, actual, bad, maps)

    def test_direct_command_never_preloads_cli_or_adds_exec_wrapper(self):
        root = Path("/fixture")
        env = {
            "LD_PRELOAD": "/fixture/work/lib/solver-shim.so",
            "VMECPP_GPU_SOLVER": "1",
            "NPY_DISABLE_CPU_FEATURES": "AVX2,FMA3",
        }
        child = ["python", "pilot.py", "profile-worker"]
        with patch.object(pilot, "environment", return_value=env):
            command, parent = pilot.direct_profile_command(
                root, "candidate", child, ["nsys", "profile"]
            )
        assert "LD_PRELOAD" not in parent
        assert parent == {k: v for k, v in env.items() if k != "LD_PRELOAD"}
        assert command == [
            "nsys",
            "profile",
            "--env-var=LD_PRELOAD=/fixture/work/lib/solver-shim.so",
            *child,
        ]
        assert "profile-exec" not in command


if __name__ == "__main__":
    unittest.main()
