"""A profile-only plan cannot rerun or relabel eighteen canonical evaluations."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.optimization import cloud_pilot_profileonly3 as pilot


class ProfileOnlyFlow(unittest.TestCase):
    def test_two_candidate_calls_and_required_raw_trace_evidence(self):
        self.run_case(False)

    def test_missing_trace_cannot_claim_diagnostic_complete(self):
        self.run_case(True)

    def test_missing_graph_nodes_cannot_complete(self):
        self.run_case(False, True)

    def run_case(self, missing_report, missing_graph=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pilot.save(
                root / "bootstrap-status.json",
                {"phase": "complete", "runtime_files": {}},
            )
            pilot.save(root / "payload-manifest.json", {"files": {}})
            calls = []

            def evaluate(_root, _seed, _track, variant, label, *_args):
                calls.append((variant, label))
                output = root / "results/evaluations" / label
                pilot.save(output / "original-global-gate.json", {"exact": True})
                pilot.save(output / "activation.json", {"pid": 1})
                if label == "profile-only-nsys" and not missing_report:
                    target = root / "results/profiles/timeline.nsys-rep"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.touch()
                return output

            with (
                patch.object(pilot, "preflight"),
                patch.object(
                    pilot,
                    "config",
                    return_value={"prior_canonical_evidence_sha256": "a" * 64},
                ),
                patch.object(pilot, "selected_cpu_environment"),
                patch.object(pilot, "hardware_snapshot"),
                patch.object(pilot, "clock_witness", return_value={}),
                patch.object(
                    pilot.os, "sched_getaffinity", return_value={0}, create=True
                ),
                patch.object(pilot.os, "sched_setaffinity", create=True),
                patch.object(pilot, "evaluate", side_effect=evaluate),
                patch.object(
                    pilot,
                    "raw_compare",
                    return_value={
                        "exact": True,
                        "numeric_fields": [{"field": "fixture"}],
                    },
                ),
                patch.object(pilot, "locate_tool", return_value=Path("nsys")),
                patch.object(pilot, "run_owned", return_value={"passed": True}),
                patch.object(pilot, "descendants", return_value=[]),
                patch(
                    "tools.optimization.analyze_full_vmec_profile.analyze",
                    return_value={"kernels": ["actual nodes required in execution"]},
                ),
                patch(
                    "tools.optimization.validate_profile_marker.validate_marker",
                    return_value={
                        "exact": True,
                        "all_target_cuda_activities_contained": True,
                    },
                ),
                patch(
                    "tools.optimization.validate_graph_nodes.validate_graph_nodes",
                    return_value={
                        "exact": True,
                        "every_successful_host_launch_has_node_activity": True,
                    },
                    side_effect=ValueError("graph nodes missing")
                    if missing_graph
                    else None,
                ),
            ):
                code = pilot.run_pilot(root, time.time(), time.time() + 2000, 2)
            state = json.loads((root / "pilot-status.json").read_text())
            assert calls == [
                ("candidate", "profile-only-anchor"),
                ("candidate", "profile-only-nsys"),
            ]
            assert "case_pairs" not in state
            assert "canonical_parity_complete" not in state
            assert state["main_table_eligible"] is False
            if missing_report or missing_graph:
                assert code == 1
                assert state["diagnostic_complete"] is False
            else:
                assert code == 0
                assert state["diagnostic_complete"] is True
                assert state["phase"] == "diagnostic_complete"
                assert state["Ncu_executed"] is False


if __name__ == "__main__":
    unittest.main()
