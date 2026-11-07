"""Intersect full-round target-process GPU graph spans and NVTX CPU phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from tools.optimization.analyze_batch_timeline import interval_metrics


def union(intervals, low, high):
    """Clip and merge half-open intervals, including touching endpoints."""
    result = []
    for original_start, original_end in sorted(intervals):
        start, end = original_start, original_end
        if end < start:
            message = "Reversed interval"
            raise ValueError(message)
        start, end = max(low, start), min(high, end)
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersection(left, right):
    """Intersect sorted, disjoint half-open interval unions."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def complement(intervals, low, high):
    result = []
    cursor = low
    for start, end in union(intervals, low, high):
        if start > cursor:
            result.append((cursor, start))
        cursor = end
    if cursor < high:
        result.append((cursor, high))
    return result


def presence_bins(graphs, phases, low, high):
    """Partition no-graph time by the exact set of active CPU phase labels."""
    events = defaultdict(list)
    for name, intervals in {"GRAPH": graphs, **phases}.items():
        for start, end in union(intervals, low, high):
            events[start].append((name, 1))
            events[end].append((name, -1))
    events[high]
    active = Counter()
    bins = Counter()
    previous = low
    for at, changes in sorted(events.items()):
        if not active["GRAPH"] and at > previous:
            key = tuple(
                sorted(
                    name for name, count in active.items() if count and name != "GRAPH"
                )
            )
            bins[key] += at - previous
        for name, delta in changes:
            active[name] += delta
        previous = at
    return [
        {"phases_present": list(names), "duration_ns": value}
        for names, value in sorted(bins.items(), key=lambda row: -row[1])
    ]


def analyze(database, worker_pid):
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        processes = connection.execute(
            "SELECT globalPid FROM PROCESSES WHERE pid=?", (worker_pid,)
        ).fetchall()
        if len(processes) != 1:
            message = "Exactly one recorded worker process is required"
            raise ValueError(message)
        pid = processes[0][0]
        phases = defaultdict(list)
        jobs = []
        for start, end, text in connection.execute(
            "SELECT start,end,COALESCE(text,s.value) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId=s.id WHERE globalTid>=? AND globalTid<? AND end IS NOT NULL",
            (pid, pid + (1 << 24)),
        ):
            if text and text.startswith("job:"):
                jobs.append((start, end, text))
            elif text in (
                "setup",
                "nativecall",
                "post-native",
                "gatewait:initial",
                "gatewait:post-native",
            ):
                phases[text].append((start, end))
        if len(jobs) != 6:
            message = "Expected exactly six measured job NVTX ranges"
            raise ValueError(message)
        low = min(row[0] for row in jobs)
        high = max(row[1] for row in jobs)
        span = high - low
        graph_records = connection.execute(
            "SELECT start,end,deviceId,contextId,streamId FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE globalPid=?",
            (pid,),
        ).fetchall()
        graphs = union([(row[0], row[1]) for row in graph_records], low, high)
        no_graph = complement(graphs, low, high)
        phase_unions = {name: union(rows, low, high) for name, rows in phases.items()}
        graph_metrics = interval_metrics(
            (
                (start - low, end - low, (device << 64) | (context << 32) | stream)
                for start, end, device, context, stream in graph_records
            ),
            span,
        )
        graph_metrics = {
            key.replace("union_busy", "graph_inflight_union")
            .replace("recorded_coverage", "recorded_graph_inflight_coverage")
            .replace("no_target_activity", "no_target_graph_inflight"): value
            for key, value in graph_metrics.items()
        }
        api = defaultdict(list)
        for start, end, name in connection.execute(
            "SELECT r.start,r.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
            "WHERE r.globalTid>=? AND r.globalTid<? AND r.end>? AND r.start<?",
            (pid, pid + (1 << 24), low, high),
        ):
            api[name].append((start, end))
        api_unions = {name: union(rows, low, high) for name, rows in api.items()}
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        other_gpu = []
        other_counts = {}
        for name in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        ):
            if name in tables:
                rows = connection.execute(
                    f"SELECT start,end FROM {name} WHERE globalPid=?", (pid,)
                ).fetchall()
                other_counts[name] = len(rows)
                other_gpu.extend(rows)
        other_union = union(other_gpu, low, high)
        evidence_files = [
            database.parent / name
            for name in (
                "configuration.json",
                "worker-status.json",
                "independent-terminal-and-artifact-proof.json",
                "evaluation/batch-summary.json",
            )
        ]
        evidence_files.append(database.parent.parent / "manifest.json")
        gaps = []
        for start, end in sorted(
            no_graph, key=lambda row: row[1] - row[0], reverse=True
        )[:20]:
            gap = [(start, end)]
            associations = [
                {
                    "api": name,
                    "union_intersection_ns": duration(intersection(gap, intervals)),
                }
                for name, intervals in api_unions.items()
            ]
            gaps.append(
                {
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "relative_start_seconds": (start - low) / 1e9,
                    "phase_intersections_ns": {
                        name: duration(intersection(gap, intervals))
                        for name, intervals in phase_unions.items()
                    },
                    "cuda_api_associations": sorted(
                        (row for row in associations if row["union_intersection_ns"]),
                        key=lambda row: -row["union_intersection_ns"],
                    )[:10],
                }
            )
        bins = presence_bins(graphs, phase_unions, low, high)
        if sum(row["duration_ns"] for row in bins) != duration(no_graph):
            message = "Exclusive phase bins failed to partition no-graph time"
            raise AssertionError(message)
        return {
            "scope": "Instrumented single warmed six-case round; GRAPH-INFLIGHT spans, never kernel busy/occupancy/roofline or clean throughput",
            "worker_pid": worker_pid,
            "global_pid": pid,
            "sqlite_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evidence_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in evidence_files
                if path.is_file()
            },
            "window": {
                "start_ns": low,
                "end_ns": high,
                "duration_ns": span,
                "definition": "min start / max end of six target-worker job NVTX ranges",
            },
            "jobs": [{"start_ns": s, "end_ns": e, "name": n} for s, e, n in jobs],
            "graph_inflight_metrics": graph_metrics,
            "graph_context_streams": sorted(
                {(d, c, s) for _, _, d, c, s in graph_records}
            ),
            "no_graph_inflight_ns": duration(no_graph),
            "other_target_cuda_record_counts": other_counts,
            "no_graph_with_recorded_other_target_cuda_activity_ns": duration(
                intersection(no_graph, other_union)
            ),
            "tail_after_last_graph_ns": high - graphs[-1][1] if graphs else span,
            "cpu_phase_union_ns": {
                name: duration(rows) for name, rows in phase_unions.items()
            },
            "no_graph_phase_intersections_ns": {
                name: duration(intersection(no_graph, rows))
                for name, rows in phase_unions.items()
            },
            "exclusive_joint_phase_presence_during_no_graph": bins,
            "largest_clipped_no_graph_gaps": gaps,
            "interpretation_limits": [
                "CPU phases from different workers overlap; phase/API associations are not additive causes",
                "Graph spans may contain internal bubbles or dependencies and do not measure node execution",
                "A target no-graph interval may contain other target CUDA activity or unrelated GPU workloads",
                "Profiler API and capture tooling affect measured timings; no clean speedup claim",
            ],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.out.write_text(
        json.dumps(analyze(arguments.sqlite, arguments.worker_pid), indent=2) + "\n"
    )
