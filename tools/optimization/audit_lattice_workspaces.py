"""Pair lattice observer contexts and audit snapshots without summing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def counters(module: dict[str, Any], size: int) -> list[int] | None:
    status = module.get("status")
    require(status in ("available", "uninitialized"), "Unsafe/error module status")
    values = module.get("counters")
    if status == "uninitialized":
        require(values is None, "Uninitialized counters must be null")
        return None
    require(isinstance(values, list) and len(values) == size, "Wrong counter layout")
    values = cast(list[int], values)
    require(
        all(type(value) is int and 0 <= value < 1 << 64 for value in values),
        "Counters must be unsigned 64-bit integers",
    )
    return values


def audit(records: list[dict[str, Any]], expected_processes: int = 2) -> dict:
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        event = record.get("event")
        require(event in ("create", "before_destroy"), "Unexpected observer event")
        event = cast(str, event)
        require(type(record.get("pid")) is int, "Missing integer PID")
        require(type(record.get("context_sequence")) is int, "Missing sequence")
        require(record.get("context_known") is True, "Unknown context")
        key = record["pid"], record["context_sequence"]
        pair = pairs.setdefault(key, {})
        require(event not in pair, "Duplicate context event")
        pair[event] = record
    processes: dict[int, list[dict[str, Any]]] = {}
    status_counts: Counter = Counter()
    binding_counts: Counter = Counter()
    context_rows = []
    for (pid, sequence), pair in sorted(pairs.items()):
        require(set(pair) == {"create", "before_destroy"}, "Unpaired context")
        create, destroy = pair["create"], pair["before_destroy"]
        for name in (
            "context_address",
            "created_thread_id",
            "dimensions_ns_mpol_ntor_nsMaxF_nsIncl",
            "basis_dimensions",
            "ns",
        ):
            require(create.get(name) == destroy.get(name), f"Metadata changed: {name}")
        require(create["context_sequence"] > 0, "Invalid context sequence")
        require(create["stats_sampled"] is False, "Create unexpectedly sampled CUDA")
        require(create["observation_status"] == "metadata_only", "Bad create status")
        require(destroy["observation_status"] == "ok", "Bad destroy observation")
        require(destroy["context_metadata_status"] == "available", "Missing metadata")
        require(
            create["monotonic_timestamp_ns"] < destroy["monotonic_timestamp_ns"],
            "Context destroyed before creation",
        )
        require(
            create["observed_thread_id"]
            == create["created_thread_id"]
            == destroy["observed_thread_id"],
            "Context owner changed",
        )
        require(
            "no context delta inferred" in destroy["counter_scope"],
            "Counter attribution scope missing",
        )
        lattice = counters(destroy["lattice"], 4)
        energy = counters(destroy["cooperative_energy"], 36)
        require(
            destroy["lattice"]["layout"]
            == "calls,accepted_components,fallback_components,normal_segments",
            "Lattice layout changed",
        )
        require(
            destroy["cooperative_energy"]["layout"]
            == "calls,fallback_thermal,fallback_magnetic,stopping_histogram_0_to_32",
            "Energy layout changed",
        )
        if lattice is not None:
            require(destroy["lattice_host_binding_observed"] is True, "Lattice unbound")
            require(
                lattice[1] + lattice[2] == 3 * lattice[0], "Lattice invariant failed"
            )
        if energy is not None:
            require(destroy["energy_host_binding_observed"] is True, "Energy unbound")
            require(sum(energy[3:]) == energy[0], "Energy histogram count mismatch")
            require(energy[1] <= energy[0], "Thermal fallback count exceeds calls")
            require(energy[2] <= energy[0], "Magnetic fallback count exceeds calls")
        for name in ("lattice", "cooperative_energy"):
            status_counts[name + ":" + destroy[name]["status"]] += 1
        for name in ("lattice_host_binding_observed", "energy_host_binding_observed"):
            binding_counts[name + ":" + str(destroy[name])] += 1
        row = {
            "pid": pid,
            "sequence": sequence,
            "ns": destroy["ns"],
            "owner_thread_id": destroy["created_thread_id"],
            "observation_status": destroy["observation_status"],
            "lattice_binding_observed": destroy["lattice_host_binding_observed"],
            "energy_binding_observed": destroy["energy_host_binding_observed"],
            "lattice": destroy["lattice"],
            "cooperative_energy": destroy["cooperative_energy"],
        }
        processes.setdefault(pid, []).append(row)
        context_rows.append(row)
    require(len(processes) == expected_processes, "Unexpected process count")
    final_fine = []
    for pid, rows in sorted(processes.items()):
        require(
            [row["sequence"] for row in rows] == list(range(1, len(rows) + 1)),
            "Context sequences are incomplete",
        )
        require(len({row["owner_thread_id"] for row in rows}) == 1, "Multiple owners")
        fine = [row for row in rows if row["ns"] == 99]
        require(bool(fine), "No final fine-grid context")
        last = max(fine, key=lambda row: row["sequence"])
        require(last["lattice"]["status"] == "available", "Final lattice unavailable")
        require(
            last["cooperative_energy"]["status"] == "available",
            "Final energy unavailable",
        )
        final_fine.append(
            {
                "pid": pid,
                "sequence": last["sequence"],
                "lattice_snapshot": last["lattice"]["counters"],
                "energy_snapshot": last["cooperative_energy"]["counters"],
            }
        )
    first = final_fine[0]
    fine_equal = all(
        row["lattice_snapshot"] == first["lattice_snapshot"]
        and row["energy_snapshot"] == first["energy_snapshot"]
        for row in final_fine
    )
    return {
        "audit_passed": fine_equal,
        "records": len(records),
        "paired_contexts": len(pairs),
        "processes": len(processes),
        "contexts_per_process": {
            str(pid): len(rows) for pid, rows in processes.items()
        },
        "contexts_by_ns": dict(Counter(str(row["ns"]) for row in context_rows)),
        "module_status_counts": dict(status_counts),
        "binding_counts": dict(binding_counts),
        "final_fine_snapshots_exact_across_repeats": fine_equal,
        "final_fine_snapshots": final_fine,
        "lattice_nonzero_fallback_snapshot_contexts": [
            {"pid": row["pid"], "sequence": row["sequence"], "ns": row["ns"]}
            for row in context_rows
            if row["lattice"]["status"] == "available"
            and row["lattice"]["counters"][2] != 0
        ],
        "energy_nonzero_fallback_snapshot_contexts": [
            {"pid": row["pid"], "sequence": row["sequence"], "ns": row["ns"]}
            for row in context_rows
            if row["cooperative_energy"]["status"] == "available"
            and any(row["cooperative_energy"]["counters"][1:3])
        ],
        "all_contexts": context_rows,
        "counter_scope": "TLS workspace snapshots, cumulative/reset generation unobserved; "
        "NO summation or context-delta inference",
        "main_table_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-status", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--expected-processes", type=int, default=2)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    result = audit(records, args.expected_processes)
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.terminal_status:
        status = json.loads(args.terminal_status.read_text())
        require(status["phase"] in ("terminal", "complete"), "Observer not terminal")
        require(status["exit_status"] == 0, "Observer process failed")
        require(status["frozen_unchanged"] is True, "Frozen process inputs changed")
        for path, expected in status["frozen"].items():
            require(
                hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
                "Frozen hash mismatch",
            )
        result["terminal_status_sha256"] = hashlib.sha256(
            args.terminal_status.read_bytes()
        ).hexdigest()
        result["frozen_process_inputs_exact"] = True
    if args.validation:
        validation = json.loads(args.validation.read_text())
        require(validation["exact"] is True, "Observed solver validation failed")
        require(
            len(validation["samples"]) == args.expected_processes, "Wrong solve count"
        )
        for sample in validation["samples"]:
            require(sample["exit_status"] == 0, "Observed solve failed")
            for field in (
                "baseline_comparison",
                "input_deck_comparison",
                "repeat_comparison",
                "trace_comparison",
            ):
                require(sample[field]["exact"] is True, "Observed strict gate failed")
            require(sample["executed_gpu_updates"] > 0, "No GPU activation")
        result["validation_sha256"] = hashlib.sha256(
            args.validation.read_bytes()
        ).hexdigest()
        result["observed_global_deck_numerical_cpu_trace_repeat_gates_exact"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    sys.stdout.write(
        json.dumps(
            {key: value for key, value in result.items() if key != "all_contexts"}
        )
        + "\n"
    )
    return 0 if result["final_fine_snapshots_exact_across_repeats"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
