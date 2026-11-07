"""Independently audit a terminal 72-run checkpoint before reporting timings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REMOTE_ROOT = "/bulk/vmecpp-optimization-20261103/solver14-qualification72/"
ORIGINAL_EXTENSION = "33587eacd2c96c5d50e0f27a95fb4385b03bd526c8a7302461ae3032b4fc9088"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def audit_terminal(root: Path) -> dict:
    verification = load(root / "verification.json")
    manifest = load(root / "checkpoint-manifest.json")
    require(verification.get("exact") is True, "Checkpoint verification missing")
    require(manifest.get("phase") == "complete", "Checkpoint is not terminal complete")
    require(manifest.get("completed_evaluations") == 72, "Incomplete checkpoint")
    for name, expected in manifest["files"].items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "Unsafe manifest path")
        require(digest(path) == expected, "Checkpoint member hash changed: " + name)
    status = load(root / "full-campaign-status.json")
    require(status.get("phase") == "complete", "Campaign not complete")
    require(status.get("exit_status") == 0, "Coordinator failed")
    for key in (
        "main_table_qualified",
        "execution_receipts_exact",
        "frozen_hashes_unchanged",
    ):
        require(status.get(key) is True, "Campaign gate failed: " + key)
    require(status.get("completed_evaluations") == 72, "Expected72 evaluations")
    require(status.get("completed_measured_evaluations") == 60, "Expected60 measured")
    require(status.get("completed_case_pairs") == 36, "Expected36 complete pairs")
    require(status.get("failed_evaluations") == 0, "Failed evaluation")
    require(
        status.get("failed_global_reference_checks") == 0, "Global reference failed"
    )
    guard = load(root / "full-guard-status.json")
    require(guard.get("phase") == "complete", "Independent guard failed")
    require(guard.get("supervisor_reaped") is True, "Supervisor not reaped")
    require(guard.get("coordinator_group_empty") is True, "Coordinator group alive")
    require(not guard.get("actions"), "Guard performed cleanup; not qualified")
    require(load(root / "postflight.json").get("exact") is True, "Postflight drift")
    receipts = load(root / "execution-receipts.json")
    require(receipts.get("exact") is True, "Execution receipts failed")
    require(len(receipts.get("rows", [])) == 72, "Missing execution receipts")
    require(all(row.get("exact") is True for row in receipts["rows"]), "Bad receipt")
    return {
        "checkpoint_archive_sha256": verification["archive_sha256"],
        "campaign_status": status,
        "guard": guard,
    }


def import_gates(tools: Path):
    spec = importlib.util.spec_from_file_location(
        "frozen_gates", tools / "campaign_gates.py"
    )
    if spec is None or spec.loader is None:
        message = "Cannot load frozen gates"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_samples(
    root: Path, tools: Path, references: Path, expected_plan: Path
) -> dict:
    baseline = load(root / "baseline-config.json")
    candidate = load(root / "candidate-config.json")
    plan = load(root / "campaign-plan.json")
    require(plan == load(expected_plan / "campaign-plan.json"), "Frozen plan changed")
    for label, config in (("baseline", baseline), ("candidate", candidate)):
        require(
            digest(root / f"{label}-config.json")
            == digest(expected_plan / f"{label}-config.json"),
            "Frozen config changed: " + label,
        )
        for name in (
            "report_pairs.py",
            "campaign_gates.py",
            "summarize.py",
            "benchmark.py",
        ):
            remote = REMOTE_ROOT + "tools/optimization/" + name
            require(
                digest(tools / name) == config["files"][remote],
                "Frozen tool mismatch: " + name,
            )
    require(baseline["flags"] == {} and baseline["preload"] == [], "Baseline modified")
    postflight = load(root / "postflight.json")["checks"]
    expected_files = baseline["files"] | candidate["files"]
    require(set(postflight) == set(expected_files), "Postflight dependency set changed")
    require(
        all(
            postflight[name]["expected"] == expected
            and postflight[name]["actual"] == expected
            for name, expected in expected_files.items()
        ),
        "Postflight dependency hash drift",
    )
    require(
        plan["baseline_original_extension"]["sha256"] == ORIGINAL_EXTENSION,
        "Wrong original CPU extension",
    )
    require(plan["candidate_flags"] == candidate["flags"], "Wrong candidate flags")
    require(
        plan["warmup_pairs"] == 1 and plan["measured_pairs"] == 5, "Wrong pair schedule"
    )
    require(plan["cpu"] == 0 and plan["VMEC_OMP_BLAS_threads"] == 1, "Wrong affinity")
    reference_hashes = {}
    for remote, expected in baseline["files"].items():
        prefix = REMOTE_ROOT + "references/"
        if remote.startswith(prefix):
            relative = remote.removeprefix(prefix)
            require(
                digest(references / relative) == expected,
                "Original reference drift: " + relative,
            )
            reference_hashes[relative] = expected
    require(len(reference_hashes) == 18, "Expected18 frozen original reference files")
    gates = import_gates(tools)
    samples = load(root / "paired/samples.json")
    require(len(samples) == 72, "Wrong sample count")
    keys = [(row["seed"], row["track"], row["pair"], row["label"]) for row in samples]
    require(len(set(keys)) == 72, "Duplicate actual evaluations")
    cases = {(row["seed"], row["track"]) for row in samples}
    require(
        len(cases) == 6 and {row[1] for row in cases} == {"P1", "P3"}, "Wrong cases"
    )
    require(len({row[0] for row in cases}) == 3, "Expected three boundaries")
    for seed, track in cases:
        expected = {
            (seed, track, pair, label)
            for pair in range(6)
            for label in ("baseline", "candidate")
        }
        require(expected.issubset(keys), "Missing case/pair/label")
    for row in samples:
        label = row["label"]
        config = baseline if label == "baseline" else candidate
        require(
            row["exit_status"] == 0 and row["captures_complete"], "Unsuccessful sample"
        )
        require(row["warmup"] == (row["pair"] == 0), "Warmup flag mismatch")
        case = f"{Path(row['seed']).stem}-{row['track']}"
        directory = root / "paired" / label / f"{case}-{row['pair']:02d}"
        require(load(directory / "report.json") == row["report"], "Raw report mismatch")
        capture = gates.read_capture(directory)
        require(capture == row["captures"], "Raw deck/provenance mismatch")
        provenance = capture["provenance"]
        runtime = provenance["experimental_runtime"]
        require(provenance["cpu_affinity"] == [0], "CPU affinity changed")
        require(
            all(
                provenance["thread_environment"].get(key) == "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            ),
            "Thread count changed",
        )
        require(
            provenance["campaign_configuration_sha256"]
            == digest(root / f"{label}-config.json"),
            "Actual config identity mismatch",
        )
        require(
            provenance["campaign_wrapper_sha256"] == config["files"][config["wrapper"]],
            "Actual wrapper identity mismatch",
        )
        require(
            provenance["harness_sha256"] == digest(tools / "benchmark.py"),
            "Actual harness mismatch",
        )
        require(
            provenance["preloaded_libraries"]
            == {name: config["files"][name] for name in config["preload"]},
            "Actual preload mismatch",
        )
        require(
            all(runtime.get(key) == value for key, value in config["flags"].items()),
            "Actual flags mismatch",
        )
        require(
            all(
                value is None or key in config["flags"]
                for key, value in runtime.items()
                if key.startswith("VMECPP_")
            ),
            "Extra experimental flags",
        )
        if label == "baseline":
            require(
                set(provenance["extensions"].values()) == {ORIGINAL_EXTENSION},
                "Baseline extension not original",
            )
        else:
            extension = config["flags"]["VMECPP_FORWARD_REFERENCE_SO"]
            require(
                provenance["extensions"].get(extension) == config["files"][extension],
                "Candidate extension mismatch",
            )
        reference = references / case
        require(
            row["original_reference"]["report_sha256"]
            == digest(reference / "report.json"),
            "Original reference receipt mismatch",
        )
        gate = gates.pair_gate(
            load(reference / "report.json"),
            row["report"],
            gates.read_capture(reference),
            capture,
            require_harness=False,
        )
        require(
            gate["exact"] is True, "Original global output/deck/diagnostic mismatch"
        )
    return {
        "samples": 72,
        "warmups": sum(row["warmup"] for row in samples),
        "case_counts": {
            f"{seed}:{track}": count
            for (seed, track), count in Counter(
                (row["seed"], row["track"]) for row in samples
            ).items()
        },
        "original_reference_files": reference_hashes,
        "all_actual_runtime_and_raw_artifacts_exact": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-tools", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--expected-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Fresh output directory required")
    terminal = audit_terminal(args.checkpoint)
    actual = audit_samples(
        args.checkpoint, args.frozen_tools, args.references, args.expected_plan
    )
    args.out.mkdir(parents=True)
    report = args.out / "qualified-pairs.json"
    command = [
        sys.executable,
        "-B",
        str(args.frozen_tools / "report_pairs.py"),
        str(args.checkpoint / "paired/samples.json"),
        "--out",
        str(report),
        "--require-original-reference",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    rows = load(report)
    require(
        rows == load(args.checkpoint / "full-campaign-summary.json"),
        "Independent reporter differs",
    )
    require(
        len(rows) == 6
        and all(
            row["main_table_eligible"] and row["measured_pairs"] == 5 for row in rows
        ),
        "Reporter qualification failed",
    )
    audit = {
        "measurement": "72 actual evaluations; 12 warmups excluded; 60 measured; all six cases qualified",
        "terminal": terminal,
        "actual_audit": actual,
        "report_command": command,
        "report_sha256": digest(report),
        "audit_source_sha256": digest(Path(__file__)),
    }
    (args.out / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "Measured native RTX3090 results; adapter wall medians, five alternating pairs per case.",
        "One warmup pair per case excluded. All reported values derive from executed runs.",
        "",
        "| Boundary | Track | Original CPU s | Solver14 s | Speedup | Paired bootstrap95% | Max metric relative drift |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        drift = row["maximum_drift_per_metric"]
        numeric = [
            value["relative"]
            for value in drift.values()
            if value.get("relative") is not None
        ]
        interval = row["paired_bootstrap_95_interval"]
        lines.append(
            f"| {Path(row['seed']).stem} | {row['track']} | {row['baseline_seconds']:.6f} | "
            f"{row['candidate_seconds']:.6f} | {row['speedup']:.4f}x | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | {max(numeric, default=0):.3g} |"
        )
    lines += [
        "",
        "Per-metric absolute/relative drift, raw pairs and all numerical/deck/repeat gates are retained in qualified-pairs.json.",
        "Null metrics remain null; both-zero metrics use the frozen reporter's zero-relative-drift convention.",
        "Solver14 uses factor caching, cooperative residual max32 and ordinary energy max32; refresh diagnostics16.",
        "Physics knobs and single-thread VMEC are unchanged. Later experimental builds are excluded.",
    ]
    (args.out / "table.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    with (args.out / "per-metric-drift.csv").open(
        "w", newline="", encoding="ascii"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "boundary",
                "track",
                "metric",
                "maximum_absolute_drift",
                "maximum_relative_drift",
            )
        )
        for row in rows:
            for name, value in sorted(row["maximum_drift_per_metric"].items()):
                writer.writerow(
                    (
                        row["seed"],
                        row["track"],
                        name,
                        value.get("absolute"),
                        value.get("relative"),
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
