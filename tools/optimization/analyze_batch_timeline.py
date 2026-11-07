"""Measure target-process GPU overlap from a diagnostic Nsight SQLite export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


def interval_metrics(intervals: Iterable[tuple[int, int, int]], duration: int) -> dict:
    """Sweep half-open intervals; count simultaneous streams, not graph spans."""
    if duration <= 0:
        message = "Capture duration must be positive"
        raise ValueError(message)
    events: list[tuple[int, int, int]] = []
    records = clipped = 0
    summed = 0
    for start, end, stream in intervals:
        if end < start:
            message = "Invalid reversed GPU interval"
            raise ValueError(message)
        records += 1
        clipped += int(start < 0 or end > duration)
        clipped_start, clipped_end = max(0, start), min(duration, end)
        if clipped_start >= clipped_end:
            continue
        summed += clipped_end - clipped_start
        events.extend(((clipped_start, 1, stream), (clipped_end, -1, stream)))
    events.sort()
    active: Counter[int] = Counter()
    by_stream_count: Counter[int] = Counter()
    last = 0
    longest_gap = current_gap = max_kernels = 0
    gaps: list[tuple[int, int]] = []
    index = 0
    while index < len(events):
        timestamp = events[index][0]
        elapsed = timestamp - last
        by_stream_count[len(active)] += elapsed
        if active:
            current_gap = 0
        else:
            current_gap += elapsed
            longest_gap = max(longest_gap, current_gap)
            if elapsed:
                gaps.append((last, timestamp))
        # Apply all starts/ends at a timestamp together: touching intervals
        # have zero overlap, independent of sort order for equal timestamps.
        while index < len(events) and events[index][0] == timestamp:
            _, delta, stream = events[index]
            active[stream] += delta
            if active[stream] == 0:
                del active[stream]
            index += 1
        max_kernels = max(max_kernels, sum(active.values()))
        last = timestamp
    by_stream_count[0] += duration - last
    longest_gap = max(longest_gap, current_gap + duration - last)
    if duration > last:
        gaps.append((last, duration))
    busy = duration - by_stream_count[0]
    overlap = sum(value for count, value in by_stream_count.items() if count >= 2)
    return {
        "records": records,
        "clipped_records": clipped,
        "capture_duration_ns": duration,
        "summed_interval_ns": summed,
        "union_busy_ns": busy,
        "recorded_coverage_percent": 100 * busy / duration,
        "distinct_stream_overlap_ns": overlap,
        "distinct_stream_overlap_percent": 100 * overlap / duration,
        "maximum_concurrent_intervals": max_kernels,
        "maximum_simultaneous_streams": max(by_stream_count),
        "longest_no_target_activity_ns": longest_gap,
        "longest_gaps_ns": [
            {"start": start, "end": end, "duration": end - start}
            for start, end in sorted(
                gaps, key=lambda gap: gap[1] - gap[0], reverse=True
            )[:20]
        ],
        "duration_ns_by_simultaneous_stream_count": dict(
            sorted(by_stream_count.items())
        ),
    }


def analyze(database: Path, worker_pid: int) -> dict:
    with sqlite3.connect(database) as connection:
        global_pids = {
            row[0]
            for row in connection.execute(
                "SELECT globalPid FROM PROCESSES WHERE pid=?", (worker_pid,)
            )
        }
        kernel_pids = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL"
            )
        }
        selected = global_pids & kernel_pids
        if len(selected) != 1:
            message = "Expected one actual worker process with kernel records"
            raise ValueError(message)
        global_pid = selected.pop()
        global_vid = global_pid & ~((1 << 48) - 1)
        durations = connection.execute(
            "SELECT duration FROM ANALYSIS_DETAILS WHERE globalVid=?", (global_vid,)
        ).fetchall()
        if len(durations) != 1:
            message = "Expected one capture duration for the worker's target"
            raise ValueError(message)
        duration = durations[0][0]
        devices = connection.execute(
            "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=?",
            (global_pid,),
        ).fetchall()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        device_results = []
        for (device,) in devices:

            def intervals(names, device_id=device):
                for name in names:
                    if name not in tables:
                        continue
                    # The table names are fixed literals, never CLI input.
                    for start, end, context, stream in connection.execute(
                        f"SELECT start,end,contextId,streamId FROM {name} "
                        "WHERE globalPid=? AND deviceId=?",
                        (global_pid, device_id),
                    ):
                        yield start, end, (context << 32) | stream

            kernel = interval_metrics(
                intervals(["CUPTI_ACTIVITY_KIND_KERNEL"]), duration
            )
            activity = interval_metrics(
                intervals(
                    [
                        "CUPTI_ACTIVITY_KIND_KERNEL",
                        "CUPTI_ACTIVITY_KIND_MEMCPY",
                        "CUPTI_ACTIVITY_KIND_MEMSET",
                    ]
                ),
                duration,
            )
            for gap in activity["longest_gaps_ns"]:
                gap["overlapping_target_cuda_api_calls"] = connection.execute(
                    "SELECT s.value,r.start,r.end,r.globalTid "
                    "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                    "JOIN StringIds s ON r.nameId=s.id "
                    "WHERE r.globalTid>=? AND r.globalTid<? AND r.end>? "
                    "AND r.start<? ORDER BY r.start LIMIT 32",
                    (global_pid, global_pid + (1 << 24), gap["start"], gap["end"]),
                ).fetchall()
            device_results.append(
                {"device": device, "kernels": kernel, "all_activity": activity}
            )
        api_rows = connection.execute(
            "SELECT s.value,COUNT(*),SUM(r.end-r.start) "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
            "WHERE r.globalTid>=? AND r.globalTid<? GROUP BY s.value "
            "ORDER BY SUM(r.end-r.start) DESC",
            (global_pid, global_pid + (1 << 24)),
        ).fetchall()
    return {
        "scope": "Instrumented target-process activity; not total GPU utilization or clean throughput",
        "worker_pid": worker_pid,
        "global_pid": global_pid,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "concurrent_kernel_evidence": "Positive distinct-stream kernel overlap measures actual overlapping kernel intervals; graph spans alone are not used",
        "cuda_api_rows_name_count_summed_ns": api_rows,
        "api_scope": "API sums may overlap across threads and GPU activity; they are not wall-time fractions. Gap intersections are associations, not causal proof.",
        "devices": device_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.worker_pid)
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
