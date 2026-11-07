"""Read-only, explicitly scoped analysis of Nsight Systems CUDA-node SQLite."""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

GPU_TABLES = {
    "kernel": "CUPTI_ACTIVITY_KIND_KERNEL",
    "memcpy": "CUPTI_ACTIVITY_KIND_MEMCPY",
    "memset": "CUPTI_ACTIVITY_KIND_MEMSET",
}
GAP_LIMITS_NS = (1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000, 10_000_000)


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Read an existing, internally selected table's column names."""
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def distribution(values: list[int], unit: str) -> dict[str, int | float | str | None]:
    """Summarize exact durations without treating sums as elapsed wall time."""
    return {
        "count": len(values),
        "unit": unit,
        "sum": sum(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


class BusyUnion:
    """Merge start-ordered intervals and retain exact gap statistics."""

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.cursor = start
        self.busy_ns = 0
        self.gap_count = 0
        self.gap_ns = 0
        self.gap_max_ns = 0
        self.gap_histogram = [0] * (len(GAP_LIMITS_NS) + 1)
        self.largest_gaps: list[tuple[int, int, int]] = []

    def add_gap(self, start: int, end: int) -> None:
        duration = end - start
        if duration <= 0:
            return
        self.gap_count += 1
        self.gap_ns += duration
        self.gap_max_ns = max(self.gap_max_ns, duration)
        bucket = next(
            (i for i, bound in enumerate(GAP_LIMITS_NS) if duration < bound),
            len(GAP_LIMITS_NS),
        )
        self.gap_histogram[bucket] += 1
        heapq.heappush(self.largest_gaps, (duration, start, end))
        if len(self.largest_gaps) > 20:
            heapq.heappop(self.largest_gaps)

    def add(self, start: int, end: int) -> None:
        start, end = max(start, self.start), min(end, self.end)
        if end <= start:
            return
        self.add_gap(self.cursor, start)
        self.busy_ns += max(0, end - max(start, self.cursor))
        self.cursor = max(self.cursor, end)

    def finish(self) -> dict[str, Any]:
        self.add_gap(self.cursor, self.end)
        self.cursor = self.end
        return {
            "window_ns": self.end - self.start,
            "recorded_activity_busy_union_ns": self.busy_ns,
            "recorded_activity_busy_fraction_of_window": self.busy_ns
            / (self.end - self.start),
            "uncovered_window_ns": self.gap_ns,
            "uncovered_interval_count": self.gap_count,
            "maximum_uncovered_interval_ns": self.gap_max_ns,
            "gap_histogram_upper_exclusive_ns": [*GAP_LIMITS_NS, None],
            "gap_histogram_counts": self.gap_histogram,
            "largest_uncovered_intervals": [
                {"duration_ns": duration, "start_ns": start, "end_ns": end}
                for duration, start, end in sorted(self.largest_gaps, reverse=True)
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope-label", required=True)
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--measured-process-wall-seconds", type=float)
    parser.add_argument("--iteration-start-regex")
    parser.add_argument("--refresh-kernel-regex", default=r"PointTerms\(")
    args = parser.parse_args()
    if args.output.resolve() == args.database.resolve():
        parser.error("Output must differ from the read-only SQLite input")
    if (
        args.measured_process_wall_seconds is not None
        and args.measured_process_wall_seconds <= 0
    ):
        parser.error("Measured process wall time must be positive")
    connection = sqlite3.connect(
        args.database.resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    available = {kind: name for kind, name in GPU_TABLES.items() if name in tables}
    if "kernel" not in available:
        parser.error("SQLite contains no individual CUDA kernel table")
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    device_predicate = "" if args.device is None else " AND deviceId=:device"
    params = {"device": args.device}
    bounds = [
        connection.execute(
            f"SELECT MIN(start),MAX(end) FROM {table} WHERE end>start{device_predicate}",
            params,
        ).fetchone()
        for table in available.values()
    ]
    valid_bounds = [(start, end) for start, end in bounds if start is not None]
    if not valid_bounds:
        parser.error("No recorded GPU activities match the device selection")
    start = (
        args.start_ns if args.start_ns is not None else min(x[0] for x in valid_bounds)
    )
    end = args.end_ns if args.end_ns is not None else max(x[1] for x in valid_bounds)
    if end <= start:
        parser.error("Scope end must be after its start")
    params.update(start=start, end=end)
    predicate = f"end>:start AND start<:end{device_predicate}"
    duration = "MIN(end,:end)-MAX(start,:start)"
    kernels = []
    kernel_total_ns = 0
    for row in connection.execute(
        f"SELECT demangledName,COUNT(*),SUM({duration}),MIN({duration}),"
        f"MAX({duration}) FROM {available['kernel']} WHERE {predicate} "
        "GROUP BY demangledName ORDER BY 3 DESC",
        params,
    ):
        name_id, count, total, minimum, maximum = row
        kernel_total_ns += total
        kernels.append(
            {
                "name": strings.get(name_id, str(name_id)),
                "count": count,
                "summed_duration_ns": total,
                "mean_duration_ns": total / count,
                "minimum_duration_ns": minimum,
                "maximum_duration_ns": maximum,
            }
        )
    for kernel in kernels:
        kernel["fraction_of_summed_kernel_durations"] = (
            kernel["summed_duration_ns"] / kernel_total_ns if kernel_total_ns else None
        )
    kernel_columns = columns(connection, available["kernel"])
    graph_nodes = None
    if "graphNodeId" in kernel_columns:
        graph_nodes = connection.execute(
            f"SELECT COUNT(*),SUM(CASE WHEN graphNodeId IS NOT NULL AND graphNodeId!=0 "
            f"THEN 1 ELSE 0 END) FROM {available['kernel']} WHERE {predicate}",
            params,
        ).fetchone()
    copies = []
    if "memcpy" in available:
        labels = (
            dict(connection.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER"))
            if "ENUM_CUDA_MEMCPY_OPER" in tables
            else {}
        )
        copy_columns = columns(connection, available["memcpy"])
        count_expression = (
            "COALESCE(copyCount,1)" if "copyCount" in copy_columns else "1"
        )
        for row in connection.execute(
            f"SELECT copyKind,COUNT(*),SUM({count_expression}),SUM(bytes),"
            f"SUM({duration}),MIN(bytes),MAX(bytes) FROM {available['memcpy']} "
            f"WHERE {predicate} GROUP BY copyKind ORDER BY SUM(bytes) DESC",
            params,
        ):
            kind, records, operations, byte_count, elapsed, minimum, maximum = row
            copies.append(
                {
                    "direction": labels.get(kind, str(kind)),
                    "activity_records": records,
                    "reported_operation_count": operations,
                    "bytes_in_records_overlapping_scope": byte_count,
                    "summed_duration_ns_clipped_to_scope": elapsed,
                    "minimum_record_bytes": minimum,
                    "maximum_record_bytes": maximum,
                    "record_size_counts": dict(
                        connection.execute(
                            f"SELECT bytes,COUNT(*) FROM {available['memcpy']} "
                            f"WHERE {predicate} AND copyKind=:kind GROUP BY bytes "
                            "ORDER BY COUNT(*) DESC LIMIT 20",
                            {**params, "kind": kind},
                        )
                    ),
                }
            )
    api_rows = []
    for source in ("RUNTIME", "DRIVER"):
        table = f"CUPTI_ACTIVITY_KIND_{source}"
        if table not in tables:
            continue
        for name_id, count, total, maximum in connection.execute(
            f"SELECT nameId,COUNT(*),SUM({duration}),MAX({duration}) FROM {table} "
            "WHERE end>:start AND start<:end GROUP BY nameId",
            params,
        ):
            name = strings.get(name_id, str(name_id))
            category = (
                "synchronization_or_wait"
                if re.search(r"Synchronize|Wait", name, re.IGNORECASE)
                else "launch"
                if "Launch" in name
                else "memory_copy"
                if "Memcpy" in name
                else "other"
            )
            api_rows.append(
                {
                    "source": source,
                    "name": name,
                    "category": category,
                    "count": count,
                    "summed_call_duration_ns": total,
                    "mean_call_duration_ns": total / count,
                    "maximum_call_duration_ns": maximum,
                }
            )
    api_rows.sort(key=lambda row: row["summed_call_duration_ns"], reverse=True)
    api_categories: dict[str, int] = defaultdict(int)
    for row in api_rows:
        api_categories[row["category"]] += row["summed_call_duration_ns"]
    launch_correlation: dict[str, Any] = {
        "available": False,
        "reason": "Runtime correlation or graph-node columns unavailable.",
    }
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables and "graphNodeId" in kernel_columns:
        kernel_filter = "k.end>:start AND k.start<:end"
        if args.device is not None:
            kernel_filter += " AND k.deviceId=:device"
        row = connection.execute(
            "WITH api AS (SELECT correlationId,MIN(end) AS api_end,"
            "MIN(nameId) AS name_id,COUNT(*) AS records "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE end>:start AND start<:end "
            "GROUP BY correlationId) "
            "SELECT COUNT(*),MIN(k.start-api.api_end),AVG(k.start-api.api_end),"
            "MAX(k.start-api.api_end),SUM(CASE WHEN k.start<api.api_end THEN 1 ELSE 0 END) "
            f"FROM {available['kernel']} AS k JOIN api USING(correlationId) "
            "JOIN StringIds AS names ON names.id=api.name_id "
            f"WHERE {kernel_filter} AND api.records=1 AND names.value LIKE '%Launch%' "
            "AND (k.graphNodeId IS NULL OR k.graphNodeId=0)",
            params,
        ).fetchone()
        launch_correlation = {
            "available": row[0] > 0,
            "unique_runtime_launch_to_non_graph_kernel_matches": row[0],
            "minimum_api_end_to_kernel_start_ns": row[1],
            "mean_api_end_to_kernel_start_ns": row[2],
            "maximum_api_end_to_kernel_start_ns": row[3],
            "kernels_starting_before_api_return": row[4] or 0,
            "limitations": "Excludes graph nodes and ambiguous correlations; negative differences are valid asynchronous overlap, not negative execution time.",
        }
    # UNION ALL preserves node activities. Never add graph envelope durations
    # to their constituent nodes: that would double-count GPU occupancy.
    event_query = (
        " UNION ALL ".join(
            f"SELECT start,end,deviceId,'{kind}' AS kind FROM {table} WHERE {predicate}"
            for kind, table in available.items()
        )
        + " ORDER BY start,end"
    )
    combined = BusyUnion(start, end)
    device_busy: dict[int, BusyUnion] = {}
    for event_start, event_end, device, _kind in connection.execute(
        event_query, params
    ):
        combined.add(event_start, event_end)
        device_busy.setdefault(device, BusyUnion(start, end)).add(
            event_start, event_end
        )
    kernel_busy = BusyUnion(start, end)
    for event_start, event_end in connection.execute(
        f"SELECT start,end FROM {available['kernel']} WHERE {predicate} ORDER BY start,end",
        params,
    ):
        kernel_busy.add(event_start, event_end)
    iterations: dict[str, Any] = {
        "available": False,
        "reason": "No explicit iteration-start regex supplied; no solver stages inferred.",
    }
    if args.iteration_start_regex:
        marker = re.compile(args.iteration_start_regex)
        refresh_marker = re.compile(args.refresh_kernel_regex)
        active: dict[tuple[int, int], dict[str, Any]] = {}
        summaries: dict[str, list[dict[str, int]]] = {"ordinary": [], "refresh": []}
        for event_start, event_end, device, context, name_id in connection.execute(
            f"SELECT start,end,deviceId,contextId,demangledName FROM {available['kernel']} "
            f"WHERE {predicate} ORDER BY start,end",
            params,
        ):
            name = strings.get(name_id, str(name_id))
            key = (device, context)
            if marker.search(name) and event_start >= start:
                if key in active:
                    previous = active[key]
                    kind = "refresh" if previous["refresh"] else "ordinary"
                    summaries[kind].append(
                        {
                            "window_ns": event_start - previous["start"],
                            "summed_kernel_ns": previous["kernel_ns"],
                            "kernel_count": previous["kernel_count"],
                        }
                    )
                active[key] = {
                    "start": event_start,
                    "refresh": False,
                    "kernel_ns": 0,
                    "kernel_count": 0,
                }
            if key in active:
                active[key]["refresh"] |= bool(refresh_marker.search(name))
                active[key]["kernel_ns"] += min(event_end, end) - max(
                    event_start, start
                )
                active[key]["kernel_count"] += 1
        iterations = {
            "available": any(summaries.values()),
            "classification": "Heuristic kernel-marker windows, per device/context; validate single solver ownership.",
            "start_regex": args.iteration_start_regex,
            "refresh_regex": args.refresh_kernel_regex,
            "window_definition": "Marker start to next marker start; includes controller/step/gaps; final incomplete window excluded.",
            "excluded_final_open_windows": len(active),
            "groups": {
                kind: {
                    field: distribution(
                        [window[field] for window in windows],
                        "count" if field == "kernel_count" else "ns",
                    )
                    for field in ("window_ns", "summed_kernel_ns", "kernel_count")
                }
                for kind, windows in summaries.items()
            },
        }
    report = {
        "database": str(args.database.resolve()),
        "scope": {
            "label": args.scope_label,
            "start_ns": start,
            "end_ns": end,
            "window_ns": end - start,
            "default_window": "First through last recorded GPU activity, not whole process wall time.",
            "device_selection": args.device,
            "measured_process_wall_seconds_supplied_by_caller": args.measured_process_wall_seconds,
        },
        "limitations": [
            "Durations are measured trace timestamps from an instrumented run.",
            "CUDA graph node collection must be explicitly enabled with --cuda-graph-trace=node.",
            "Node collection adds profiling overhead; speed claims require separate unprofiled paired runs.",
            "Graph/kernel duration sums can overlap; kernel shares are not shares of process wall time.",
            "Recorded GPU activity coverage is not SM utilization, memory bandwidth, or achieved roofline.",
            "Uncovered intervals mean no recorded target-process activity, not an idle physical GPU.",
            "CPU API durations can overlap each other and GPU work; runtime/driver nesting can double-count.",
            "API rows include all traced CPUs/processes; device filter applies only to GPU activities.",
            "Copy bytes belong to whole records overlapping scope and cannot be time-prorated reliably.",
        ],
        "graph_node_evidence": {
            "kernel_records": graph_nodes[0] if graph_nodes else None,
            "kernel_records_with_nonzero_graph_node_id": graph_nodes[1]
            if graph_nodes
            else None,
            "warning": "A missing/nonzero count alone cannot certify complete node collection; retain profiler command.",
        },
        "summed_kernel_duration_ns": kernel_total_ns,
        "kernels": kernels,
        "copies": copies,
        "cpu_cuda_api_categories_summed_ns": dict(api_categories),
        "cpu_cuda_api_calls": api_rows,
        "direct_launch_correlation": launch_correlation,
        "recorded_gpu_activity_union_any_device": combined.finish(),
        "recorded_gpu_kernel_union_any_device": kernel_busy.finish(),
        "recorded_gpu_activity_union_by_device": {
            str(device): busy.finish() for device, busy in device_busy.items()
        },
        "iteration_windows": iterations,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
