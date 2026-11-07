"""CPU-only resource-role and persistent service delegation controls."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.optimization import native_queue_campaign as campaign


class RoleControls(unittest.TestCase):
    def test_roles_are_four_solves_but_different_adapter_resources(self):
        assert (
            campaign.ROLES[1]["native_workers"]
            == campaign.ROLES[4]["native_workers"]
            == 4
        )
        assert campaign.ROLES[1]["adapter_workers"] == 4
        assert campaign.ROLES[4]["adapter_workers"] == 6
        assert set(campaign.ROLES[4]["adapter_cpus"]).isdisjoint(
            campaign.ROLES[4]["native_cpus"]
        )

    def test_service_internal_labels_do_not_change_actual_workers(self):
        for label in (1, 4):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "run").mkdir()
                (root / "run/launch.json").write_text(
                    json.dumps({"nonce": "cpu-control"})
                )
                original = Mock(return_value=([], []))
                dispatcher = SimpleNamespace(
                    close=Mock(), futures=[], queue_records=[], closed=True
                )

                def controlled(
                    _root,
                    _config,
                    role_label,
                    run,
                    groups,
                    dispatch,
                    evaluate,
                    finalize,
                    *,
                    verify_artifacts,
                    label=label,
                ):
                    assert role_label == label
                    return run(
                        groups,
                        dispatch,
                        evaluate,
                        finalize,
                        role_label,
                        verify_artifacts=verify_artifacts,
                    )

                def main(label=label, dispatcher=dispatcher):
                    expected = campaign.ROLES[label]["adapter_workers"]
                    assert sys.argv[sys.argv.index("--workers") + 1] == str(expected)
                    campaign.batch.run_rounds(
                        [[]],
                        dispatcher,
                        None,
                        None,
                        expected,
                        verify_artifacts=lambda: None,
                    )
                    return 0

                with (
                    patch.object(
                        campaign.os, "sched_getaffinity", return_value=set(range(10))
                    ),
                    patch.object(campaign.os, "getpriority", return_value=0),
                    patch.object(campaign.batch, "run_rounds", original),
                    patch.object(campaign.batch, "CaptureDispatcher"),
                    patch.object(campaign.batch, "main", side_effect=main),
                    patch.object(
                        campaign.base, "controlled_rounds", side_effect=controlled
                    ),
                    patch.object(sys, "argv", []),
                ):
                    assert campaign.service(root, label) == 0
                assert (
                    original.call_args.args[4]
                    == campaign.ROLES[label]["adapter_workers"]
                )
                receipt = json.loads(
                    (root / "run" / f"service-{label}.json").read_text()
                )
                assert receipt["exit_status"] == 0
                assert receipt["phase"] == "terminal"
                if label == 4:
                    dispatcher.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
