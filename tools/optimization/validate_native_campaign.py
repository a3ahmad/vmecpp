"""Audit locally retrieved native campaign artifacts without touching the host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.optimization.campaign_gates import audit_case, pair_gate, read_capture
from tools.optimization.report_pairs import summarize_pairs


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def audit(checkpoint: Path, references: Path) -> dict:
    proof = json.loads((checkpoint / "verification.json").read_text())
    if proof.get("exact") is not True:
        message = "Checkpoint retrieval verification must finish before audit"
        raise ValueError(message)
    checkpoints = sorted(
        path
        for path in checkpoint.parent.glob("c*")
        if path.is_dir() and path.name <= checkpoint.name
    )
    verified = {}
    conflicts = []
    failures = []
    for directory in checkpoints:
        manifest = directory / "checkpoint-manifest.json"
        if not manifest.is_file():
            continue
        for name, expected in json.loads(manifest.read_text())["files"].items():
            path = directory / name
            if not path.is_file() or digest(path) != expected:
                failures.append({"kind": "checkpoint_hash", "path": str(path)})
                continue
            if name.startswith("paired/") and name != "paired/samples.json":
                if name in verified and digest(verified[name]) != expected:
                    conflicts.append(name)
                verified[name] = path
    records = json.loads((checkpoint / "paired/samples.json").read_text())
    status = json.loads((checkpoint / "full-campaign-status.json").read_text())
    if proof.get("completed_evaluations") != len(records):
        message = "Verified checkpoint count does not match its actual records"
        raise ValueError(message)
    configurations = {label: set() for label in ("baseline", "candidate")}
    per_metric = {}
    actual_artifacts = []
    for record in records:
        case = f"{Path(record['seed']).stem}-{record['track']}"
        name = f"paired/{record['label']}/{case}-{record['pair']:02d}"
        report_path = verified.get(name + "/report.json")
        provenance_path = verified.get(name + "/provenance.json")
        if report_path is None or provenance_path is None:
            failures.append({"kind": "missing_actual_artifact", "name": name})
            continue
        captured = record["captures"]
        report = json.loads(report_path.read_text())
        provenance = json.loads(provenance_path.read_text())
        actual_exact = (
            canonical(report) == canonical(record["report"])
            and canonical(provenance) == canonical(captured["provenance"])
            and digest(provenance_path) == captured["provenance_sha256"]
        )
        for number, expected in enumerate(captured["actual_indata_sha256"]):
            deck_path = verified.get(name + f"/actual_indata_{number}.json")
            actual_exact = actual_exact and bool(
                deck_path
                and digest(deck_path) == expected
                and canonical(json.loads(deck_path.read_text()))
                == canonical(captured["actual_indata"][number])
            )
        reference = references / case
        reference_report_path = reference / "report.json"
        reference_report = json.loads(reference_report_path.read_text())
        global_gate = pair_gate(
            reference_report,
            report,
            read_capture(reference),
            captured,
            require_harness=False,
        )
        global_exact = (
            global_gate["exact"]
            and digest(reference_report_path)
            == record["original_reference"]["report_sha256"]
        )
        config_path = checkpoint / (record["label"] + "-config.json")
        config = json.loads(config_path.read_text())
        runtime = provenance["experimental_runtime"]
        flags_exact = {
            key: value for key, value in runtime.items() if value is not None
        } == config["flags"]
        libraries_exact = all(
            config["files"].get(path) == expected
            for section in ("extensions", "preloaded_libraries")
            for path, expected in provenance[section].items()
        ) and set(provenance["preloaded_libraries"]) == set(config["preload"])
        single_threaded = provenance["cpu_affinity"] == [0] and all(
            value == "1" for value in provenance["thread_environment"].values()
        )
        config_exact = provenance["campaign_configuration_sha256"] == digest(
            config_path
        )
        configurations[record["label"]].add(
            canonical(
                {
                    key: provenance.get(key)
                    for key in (
                        "harness_sha256",
                        "campaign_wrapper_sha256",
                        "campaign_configuration_sha256",
                        "extensions",
                        "packages",
                        "cpu_affinity",
                        "thread_environment",
                        "experimental_runtime",
                        "preloaded_libraries",
                    )
                }
            )
        )
        for metric, value in report["metrics"].items():
            baseline = reference_report["metrics"][metric]
            maximum = per_metric.setdefault(
                metric,
                {"absolute": 0, "relative": 0, "numeric_samples": 0, "null_samples": 0},
            )
            if value is None and baseline is None:
                maximum["null_samples"] += 1
                continue
            absolute = abs(value - baseline)
            relative = absolute / abs(baseline) if baseline else float(bool(absolute))
            maximum["numeric_samples"] += 1
            maximum["absolute"] = max(maximum["absolute"], absolute)
            maximum["relative"] = max(maximum["relative"], relative)
        exact = (
            actual_exact
            and global_exact
            and flags_exact
            and config_exact
            and libraries_exact
            and single_threaded
        )
        if not exact:
            failures.append({"kind": "record", "name": name})
        actual_artifacts.append(
            {
                "name": name,
                "actual_report_provenance_decks_exact": actual_exact,
                "original_global_gate_exact": global_exact,
                "frozen_flags_exact": flags_exact,
                "frozen_configuration_hash_exact": config_exact,
                "actual_libraries_match_frozen_recipe": libraries_exact,
                "single_thread_cpu0": single_threaded,
            }
        )
    cases = []
    for seed, track in sorted(
        {(record["seed"], record["track"]) for record in records}
    ):
        selected = [r for r in records if r["seed"] == seed and r["track"] == track]
        cases.append(
            {
                "seed": seed,
                "track": track,
                "records": len(selected),
                "audit": audit_case(
                    selected, checkpoint / "paired", require_original_reference=True
                ),
            }
        )
    rows = summarize_pairs(
        records, checkpoint / "paired", require_original_reference=True
    )
    terminal = status.get("phase") == "complete" and status.get("exit_status") == 0
    complete = (
        terminal
        and len(records) == 72
        and len(rows) == 6
        and sum(record["warmup"] for record in records) == 12
        and all(row["measured_pairs"] == 5 for row in rows)
        and all(row["main_table_eligible"] for row in rows)
    )
    return {
        "scope": "read-only independent audit of executed, locally retrieved native campaign records",
        "checkpoint": str(checkpoint),
        "checkpoint_status": status,
        "records": len(records),
        "verified_checkpoint_files": len(verified),
        "conflicting_immutable_artifacts": sorted(set(conflicts)),
        "failures": failures,
        "actual_record_audits": actual_artifacts,
        "per_metric_maximum_drift_against_original": per_metric,
        "variant_configuration_counts_across_all_cases": {
            key: len(value) for key, value in configurations.items()
        },
        "cases": cases,
        "reporter_rows": rows,
        "all_available_actual_records_exact": not failures and not conflicts,
        "campaign_terminal": terminal,
        "full_qualification": complete
        and not failures
        and not conflicts
        and all(len(v) == 1 for v in configurations.values()),
        "incomplete_campaign_numbers_are_provisional": not complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.references)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["all_available_actual_records_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
