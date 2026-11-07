"""Keep CPU dispatch experiments separate from official identity qualification."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tools.optimization import cloud_cpu_compatibility as compatibility

# Keep these critical orchestration guards runnable with the stdlib test runner.
# ruff: noqa: PT027


def record(variant="default", environment=None, **changes) -> dict[str, Any]:
    value = {
        "variant": variant,
        "environment_overrides": environment or {},
        "job_passed": True,
        "owned_children_reaped": True,
        "metric_exact": True,
        "equilibrium_exact": True,
        "source_exact": True,
        "intermediates_exact": True,
        "repeat_index": 0,
    }
    value.update(changes)
    return value


def replay_summary(*variants) -> dict[str, Any]:
    records = []
    for first in variants:
        second: dict[str, Any] = dict(first, repeat_index=1)
        second["repeat_against_first"] = {
            "metric_exact": True,
            "source_exact": True,
            "equilibrium_exact": True,
            "intermediates_exact": first["intermediates_exact"],
        }
        records.extend((first, second))
    return {
        "records": records,
        "requested_repeats": 2,
        "requested_variants": [first["variant"] for first in variants],
    }


class CpuCompatibilityTests(unittest.TestCase):
    def test_metric_drift_cannot_be_selected_despite_exact_equilibrium(self):
        with self.assertRaisesRegex(ValueError, "No exact original-metric"):
            compatibility.select_environment(replay_summary(record(metric_exact=False)))

    def test_incomplete_or_untrusted_replay_cannot_be_selected(self):
        for field in ("source_exact", "job_passed", "owned_children_reaped"):
            value = record()
            value[field] = False
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "No exact original-metric"),
            ):
                compatibility.select_environment(replay_summary(value))

    def test_unknown_environment_cannot_inject_solver_controls(self):
        unsafe = record(environment={"VMECPP_GPU_RESIDUAL_ROUNDS": "12"})
        with self.assertRaisesRegex(ValueError, "only declared CPU dispatch"):
            compatibility.select_environment(replay_summary(unsafe))

    def test_exact_metric_with_different_intermediates_remains_diagnostic(self):
        selection = compatibility.select_environment(
            replay_summary(record(intermediates_exact=False))
        )
        assert selection["full_official_all_metric_identity_required"] is True
        assert selection["main_table_eligible"] is False
        assert selection["intermediates_exact"] is False

    def test_strong_replay_is_preferred_without_changing_arithmetic(self):
        selected = compatibility.select_environment(
            replay_summary(
                record(intermediates_exact=False),
                record("blas-haswell", {"OPENBLAS_CORETYPE": "Haswell"}),
            )
        )
        assert selected["environment"] == {"OPENBLAS_CORETYPE": "Haswell"}
        assert selected["main_table_eligible"] is False

    def test_dispatch_scope_does_not_expand(self):
        for features in ("AVX2", "AVX512F,AVX512F", "AVX512F,PATH=bad"):
            with (
                self.subTest(features=features),
                self.assertRaisesRegex(ValueError, "Only explicit AVX512"),
            ):
                compatibility.validate_environment(
                    "numpy-no-avx512", {"NPY_DISABLE_CPU_FEATURES": features}
                )

    def test_selection_is_bound_to_actual_receipt_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            selected = Path(directory) / "selection.json"
            summary.write_text(json.dumps(replay_summary(record())))
            selected.write_text(json.dumps(compatibility.decision_from_file(summary)))
            assert compatibility.read_selection(selected, summary) == {}
            summary.write_text(json.dumps(replay_summary(record(metric_exact=False))))
            with self.assertRaisesRegex(ValueError, "No exact original-metric"):
                compatibility.read_selection(selected, summary)

    def test_missing_process_is_not_a_passing_partial_probe(self):
        value = replay_summary(record())
        value["records"].pop()
        with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
            compatibility.select_environment(value)

    def test_repeat_drift_cannot_be_selected(self):
        value = replay_summary(record())
        value["records"][1]["repeat_against_first"]["metric_exact"] = False
        with self.assertRaisesRegex(ValueError, "No exact original-metric"):
            compatibility.select_environment(value)
