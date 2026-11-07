"""CPU-only paired-barrier, ELF relocation and checkpoint qualification tests."""

import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.optimization import native_batch_campaign as campaign
from tools.optimization.prepare_native_batch import text_digest
from tools.optimization.test_batch_throughput import expect_error


class NativeBatchTests(unittest.TestCase):
    def test_null_numeric_maximum_preserves_drift_and_rejects_invalid_values(self):
        assert campaign.metric_maximum(0.0, None) == 0.0
        assert campaign.metric_maximum(0.0, 0.0) == 0.0
        assert campaign.metric_maximum(0.0, 1e-12) == 1e-12
        for value in (float("nan"), float("inf"), "invalid"):
            with expect_error(ValueError):
                campaign.metric_maximum(0.0, value)

    def test_alternating_paired_order_includes_excluded_warmup(self):
        assert [campaign.paired_order(pair) for pair in range(6)] == [
            (1, 4),
            (4, 1),
            (1, 4),
            (4, 1),
            (1, 4),
            (4, 1),
        ]

    def test_full_coverage_rejects_missing_case_or_wrong_warmup(self):
        records = [
            {"seed": str(seed), "track": track, "round": number, "warmup": number == 0}
            for seed in range(3)
            for track in ("P1", "P3")
            for number in range(6)
        ]
        assert campaign.workload_coverage({"records": records})
        assert not campaign.workload_coverage({"records": records[:-1]})
        records[0]["warmup"] = False
        assert not campaign.workload_coverage({"records": records})

    def test_topology_selection_avoids_smt_siblings_and_other_numa_nodes(self):
        rows = [
            {"cpu": cpu, "core": cpu // 2, "package": 0, "node": "node0"}
            for cpu in range(10)
        ]
        rows.append({"cpu": 10, "core": 9, "package": 0, "node": "node1"})
        selected = campaign.physical_affinity(rows)
        assert selected["service_cpus"] == [0, 2, 4, 6]
        assert selected["coordinator_cpu"] == 8

    def test_preload_applies_to_python_after_non_python_taskset_launcher(self):
        command, environment = campaign.service_launch(
            {"LD_PRELOAD": "/frozen/python-only.so", "PATH": "/usr/bin"},
            [0, 1, 2, 3],
            "/frozen/python",
            ["-B", "-c", "pass"],
        )
        assert "LD_PRELOAD" not in environment
        assert command[:4] == ["/usr/bin/taskset", "-c", "0,1,2,3", "/usr/bin/env"]
        assert command[4:6] == ["LD_PRELOAD=/frozen/python-only.so", "/frozen/python"]

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux thread affinity")
    def test_four_workers_are_pinned_to_unique_cpus_from_same_process_mask(self):
        original = os.sched_getaffinity(0)
        cpus = sorted(original)[:4]
        if len(cpus) < 4:
            self.skipTest("Requires four available CPUs")
        rendezvous = threading.Barrier(4)
        try:
            os.sched_setaffinity(0, set(cpus))
            with campaign.pinned_executor(cpus)(max_workers=4) as executor:

                def read():
                    rendezvous.wait(timeout=5)
                    return sorted(os.sched_getaffinity(0))

                futures = [executor.submit(read) for _ in range(4)]
                actual = sorted(future.result(timeout=10) for future in futures)
            assert actual == [[cpu] for cpu in cpus]
        finally:
            os.sched_setaffinity(0, original)

    def barrier_probe(self, root, *, raw_valid=True, wrong_grant=False):
        (root / "run").mkdir()
        groups = []
        for number in range(2):
            states = []
            for index in range(6):
                output = root / f"sample-{number}-{index}"
                output.mkdir()
                (output / "report.json").write_text("{}")
                states.append(SimpleNamespace(job=SimpleNamespace(out=output)))
            groups.append(states)
            campaign.save(
                root / "run" / f"grant-4-{number:02d}.json",
                {
                    "nonce": "wrong" if wrong_grant else "nonce",
                    "workers": 4,
                    "pair": number,
                },
            )
        completed = []

        def original(
            groups, _dispatcher, _evaluate, finalize, workers, *, verify_artifacts
        ):
            for number, states in enumerate(groups):
                verify_artifacts()
                assert (root / "run" / f"ready-{workers}-{number:02d}.json").is_file()
                for state in states:
                    state.number = number
                    finalize(state)
                assert not (
                    root / "run" / f"completed-{workers}-{number:02d}.json"
                ).exists()
                verify_artifacts()
            return "all drained"

        def finalize(state):
            completed.append(state)
            return {"round": state.number, "gate": {"exact": True}}

        configuration = {
            "nonce": "nonce",
            "expires_monotonic": time.monotonic() + 10,
            "files": {},
        }
        result = campaign.controlled_rounds(
            root,
            configuration,
            4,
            original,
            groups,
            None,
            None,
            finalize,
            verify_artifacts=lambda: None,
            raw_verifier=lambda *_: raw_valid,
        )
        return result, completed

    def test_persistent_pool_barriers_publish_only_after_six_validated_futures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, completed = self.barrier_probe(root)
            assert result == "all drained"
            assert len(completed) == 12
            for number in range(2):
                value = json.loads(
                    (root / "run" / f"completed-4-{number:02d}.json").read_text()
                )
                assert value["exact"]
                assert len(value["records"]) == 6
                assert value["warmup"] == (number == 0)

    def test_wrong_grant_and_raw_drift_cannot_advance_pair(self):
        for wrong_grant, raw_valid in ((True, True), (False, False)):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with expect_error(RuntimeError):
                    self.barrier_probe(
                        root, wrong_grant=wrong_grant, raw_valid=raw_valid
                    )
                assert not (root / "run/completed-4-01.json").exists()
                if not raw_valid:
                    assert not json.loads(
                        (root / "run/completed-4-00.json").read_text()
                    )["exact"]

    @unittest.skipUnless(Path("/proc").exists(), "Linux ELF fixture")
    def test_elf_text_digest_distinguishes_code_from_header_relocation(self):
        value = bytearray(Path(sys.executable).resolve().read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "copied-elf"
            path.write_bytes(value)
            expected = text_digest(path)
            value[24] ^= 1
            path.write_bytes(value)
            assert text_digest(path) == expected
            section_offset = struct.unpack_from("<Q", value, 40)[0]
            stride, count, names_index = struct.unpack_from("<HHH", value, 58)
            sections = [
                struct.unpack_from(
                    "<IIQQQQIIQQ", value, section_offset + index * stride
                )
                for index in range(count)
            ]
            names = sections[names_index]
            strings = value[names[4] : names[4] + names[5]]
            text = next(
                section
                for section in sections
                if strings[section[0] :].split(b"\0", 1)[0] == b".text"
            )
            value[text[4]] ^= 1
            path.write_bytes(value)
            assert text_digest(path) != expected

    def checkpoint_fixture(self, root, *, extra=False, corrupt=False, unsafe=False):
        content = b"executed public report fixture"
        expected = hashlib.sha256(content).hexdigest()
        manifest = json.dumps({"files": {"evals/report.json": expected}}).encode()
        archive = root / "checkpoint.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            entries = [
                ("evals/report.json", b"corrupt" if corrupt else content),
                ("final-manifest.json", manifest),
            ]
            if extra:
                entries.append(("uncovered.txt", b"uncovered"))
            if unsafe:
                entries.append(("../escape", b"unsafe"))
            for name, value in entries:
                info = tarfile.TarInfo(name)
                info.size = len(value)
                bundle.addfile(info, io.BytesIO(value))
        return archive

    def test_checkpoint_positive_control_verifies_all_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.checkpoint_fixture(root)
            campaign.verify_checkpoint(
                archive, campaign.digest(archive), root / "verified"
            )
            assert json.loads((root / "verified/verification.json").read_text())[
                "all_files_exact"
            ]

    def test_checkpoint_rejects_uncovered_corrupt_and_traversal_members(self):
        for options in ({"extra": True}, {"corrupt": True}, {"unsafe": True}):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = self.checkpoint_fixture(root, **options)
                with expect_error(ValueError):
                    campaign.verify_checkpoint(
                        archive, campaign.digest(archive), root / "verified"
                    )
                assert not (root.parent / "escape").exists()


if __name__ == "__main__":
    unittest.main()
