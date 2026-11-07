"""Analyze one warmed whole-VMEC node trace without roofline inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from tools.optimization.analyze_batch_round import (
    complement,
    duration,
    intersection,
    union,
)
from tools.optimization.analyze_batch_timeline import interval_metrics


def grouped(rows, low, high):
    totals = defaultdict(lambda: {"calls": 0, "clipped_duration_ns": 0})
    for start, end, label in rows:
        elapsed = max(0, min(high, end) - max(low, start))
        if elapsed:
            totals[label]["calls"] += 1
            totals[label]["clipped_duration_ns"] += elapsed
    return [
        {"name": label, **record}
        for label, record in sorted(
            totals.items(), key=lambda item: -item[1]["clipped_duration_ns"]
        )
    ]


def analyze(database, worker_pid):
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        matches = connection.execute(
            "SELECT globalPid FROM PROCESSES WHERE pid=?", (worker_pid,)
        ).fetchall()
        if len(matches) != 1:
            msg = "Exactly one target process is required"
            raise ValueError(msg)
        pid = matches[0][0]
        nvtx = connection.execute(
            "SELECT start,end,COALESCE(n.text,s.value) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE globalTid>=? AND globalTid<? AND end IS NOT NULL",
            (pid, pid + (1 << 24)),
        ).fetchall()
        calls = [row for row in nvtx if row[2] == "whole vmecpp.run"]
        if len(calls) != 1:
            msg = "Exactly one full-call NVTX interval is required"
            raise ValueError(msg)
        low, high, _ = calls[0]
        span = high - low
        kernels = connection.execute(
            "SELECT start,end,s.value,deviceId,contextId,streamId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
            "ON k.demangledName=s.id WHERE globalPid=?",
            (pid,),
        ).fetchall()
        api = connection.execute(
            "SELECT start,end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId=s.id WHERE globalTid>=? AND globalTid<?",
            (pid, pid + (1 << 24)),
        ).fetchall()
        copies = connection.execute(
            "SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE globalPid=?",
            (pid,),
        ).fetchall()
        memsets = connection.execute(
            "SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE globalPid=?",
            (pid,),
        ).fetchall()
        kernel_union = union([(r[0], r[1]) for r in kernels], low, high)
        activities = union(
            [(r[0], r[1]) for r in [*kernels, *copies, *memsets]], low, high
        )
        no_activity = complement(activities, low, high)
        api_union = union([(r[0], r[1]) for r in api], low, high)
        stage_rows = [r for r in nvtx if r[2].startswith("GPU context ns=")]
        stages = []
        for original_start, original_end, label in stage_rows:
            start, end = max(low, original_start), min(high, original_end)
            if end <= start:
                continue
            stages.append(
                {
                    "label": label,
                    "start_relative_ns": start - low,
                    "end_relative_ns": end - low,
                    "host_context_lifetime_ns": end - start,
                    "kernel_union_ns": duration(union(kernel_union, start, end)),
                    "activity_union_ns": duration(union(activities, start, end)),
                    "kernels": grouped(kernels_as_named(kernels), start, end),
                    "runtime_api": grouped(api, start, end),
                }
            )
        gaps = []
        for start, end in sorted(no_activity, key=lambda r: r[1] - r[0], reverse=True)[
            :20
        ]:
            gaps.append(
                {
                    "start_relative_ns": start - low,
                    "end_relative_ns": end - low,
                    "duration_ns": end - start,
                    "target_api_intersections": grouped(api, start, end),
                    "host_phase_intersections": grouped(nvtx, start, end),
                }
            )
        return {
            "scope": "Measured diagnostic, target process only; node trace instrumented",
            "limitations": [
                "Recorded kernel union is not occupancy or achieved roofline",
                "GPU context lifetime excludes preceding CPU setup and can restart",
                "API and GPU durations overlap; their sums are not additive causes",
                "No-target-activity gaps do not prove the whole GPU idle",
                "No CPU controller NVTX interposer was installed",
            ],
            "main_table_eligible": False,
            "worker_pid": worker_pid,
            "global_pid": pid,
            "whole_call_start_ns": low,
            "whole_call_end_ns": high,
            "whole_call_duration_ns": span,
            "kernel_intervals": interval_metrics(
                (
                    (s - low, e - low, (d << 64) | (c << 32) | stream)
                    for s, e, _, d, c, stream in kernels
                ),
                span,
            ),
            "all_activity_union_ns": duration(activities),
            "no_target_activity_ns": duration(no_activity),
            "runtime_api_union_ns": duration(api_union),
            "no_activity_intersect_runtime_api_ns": duration(
                intersection(no_activity, api_union)
            ),
            "kernels": grouped(kernels_as_named(kernels), low, high),
            "runtime_api": grouped(api, low, high),
            "copy_records": len(copies),
            "copy_bytes": sum(r[2] for r in copies),
            "copy_summed_duration_ns": sum(r[1] - r[0] for r in copies),
            "copies_by_kind": {
                str(kind): {
                    "records": sum(r[3] == kind for r in copies),
                    "bytes": sum(r[2] for r in copies if r[3] == kind),
                    "summed_duration_ns": sum(
                        r[1] - r[0] for r in copies if r[3] == kind
                    ),
                }
                for kind in sorted({r[3] for r in copies})
            },
            "memset_records": len(memsets),
            "memset_bytes": sum(r[2] for r in memsets),
            "host_phases": grouped(nvtx, low, high),
            "stages": stages,
            "largest_no_target_activity_gaps": gaps,
            "sqlite_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        }


def kernels_as_named(kernels):
    return [(row[0], row[1], row[2]) for row in kernels]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(analyze(args.sqlite, args.worker_pid), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
