"""Quantify device work and uncovered stream time from a captured replay trace."""

from __future__ import annotations

# This CLI reports measured profiler evidence.
# ruff: noqa: T201
import argparse
import itertools
import json
import sqlite3
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-calls", type=int, default=11)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    uploads = connection.execute(
        "SELECT start, deviceId, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "WHERE copyKind=1 ORDER BY start"
    ).fetchall()
    streams = {(row[1], row[2]) for row in uploads}
    if len(streams) != 1 or args.skip_calls < 0:
        parser.error("requires one upload stream and nonnegative skip count")
    device, stream = streams.pop()
    events = []
    for table, category in (
        ("KERNEL", "kernels"),
        ("MEMSET", "memset"),
        ("MEMCPY", "copies"),
    ):
        extra = ",copyKind" if category == "copies" else ""
        rows = connection.execute(
            f"SELECT start,end{extra} FROM CUPTI_ACTIVITY_KIND_{table} "
            "WHERE deviceId=? AND streamId=? ORDER BY start",
            (device, stream),
        ).fetchall()
        for row in rows:
            kind = category
            if category == "copies":
                kind = {1: "h2d", 2: "d2h"}.get(row[2], "other_copies")
            events.append((row[0], row[1], kind))
    connection.close()
    events.sort()
    if any(b[0] < a[1] for a, b in itertools.pairwise(events)):
        parser.error("overlapping stream events require union accounting")
    intervals = []
    for first, last in zip(
        uploads[args.skip_calls : -1],
        uploads[args.skip_calls + 1 :],
        strict=True,
    ):
        start, end = first[0], last[0]
        times = {"h2d": 0, "d2h": 0, "kernels": 0, "memset": 0}
        for event_start, event_end, kind in events:
            duration = min(end, event_end) - max(start, event_start)
            if duration > 0:
                times[kind] = times.get(kind, 0) + duration
        times["uncovered_stream_time"] = end - start - sum(times.values())
        times["period"] = end - start
        intervals.append({name: value / 1000 for name, value in times.items()})
    if not intervals:
        parser.error("no complete post-warmup intervals")
    means = {
        name: statistics.mean(row[name] for row in intervals) for name in intervals[0]
    }
    result = {
        "measurement": "derived from executed Nsight Systems device events",
        "scope": "captured replay; profiler timing, not unprofiled solver time",
        "uncovered_time_meaning": (
            "no recorded kernel/copy/memset on this stream; includes host packing, "
            "launch, synchronization and scheduling gaps, not attributed separately"
        ),
        "excluded_initial_calls": args.skip_calls,
        "complete_intervals": len(intervals),
        "device": device,
        "stream": stream,
        "mean_microseconds": means,
        "fractions_of_mean_period": {
            name: value / means["period"]
            for name, value in means.items()
            if name != "period"
        },
        "intervals_microseconds": intervals,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"intervals": len(intervals), "mean_microseconds": means}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
