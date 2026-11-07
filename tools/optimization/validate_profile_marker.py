"""Require recorded target NVTX scope and contained CUDA kernel intervals."""

import sqlite3


def validate_marker(database, worker_pid):
    """A callback return sign is never evidence that a range was recorded."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        processes = connection.execute(
            "SELECT globalPid FROM PROCESSES WHERE pid=?", (worker_pid,)
        ).fetchall()
        if len(processes) != 1:
            msg = "Exactly one actual target process is required"
            raise ValueError(msg)
        global_pid = processes[0][0]
        ranges = connection.execute(
            "SELECT start,end FROM NVTX_EVENTS n LEFT JOIN StringIds s "
            "ON n.textId=s.id WHERE globalTid>=? AND globalTid<? "
            "AND COALESCE(n.text,s.value)=?",
            (global_pid, global_pid + (1 << 24), "whole vmecpp.run"),
        ).fetchall()
        if len(ranges) != 1:
            msg = "Exactly one recorded target whole-call range required"
            raise ValueError(msg)
        low, high = ranges[0]
        if not isinstance(low, int) or not isinstance(high, int) or high <= low:
            msg = "Whole-call range must be complete and positive"
            raise ValueError(msg)
        kernels = connection.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=?",
            (global_pid,),
        ).fetchall()
        activities = list(kernels)
        available_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        optional_tables = ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")
        for table in optional_tables:
            if table not in available_tables:
                continue
            activities.extend(
                connection.execute(
                    f"SELECT start,end FROM {table} WHERE globalPid=?", (global_pid,)
                ).fetchall()
            )
        if not kernels:
            msg = "Actual target CUDA kernels are required"
            raise ValueError(msg)
        if any(
            not isinstance(start, int)
            or not isinstance(end, int)
            or not low <= start < end <= high
            for start, end in activities
        ):
            msg = "Target CUDA activity is outside whole-call NVTX range"
            raise ValueError(msg)
    return {
        "exact": True,
        "worker_pid": worker_pid,
        "global_pid": global_pid,
        "recorded_whole_call_ranges": 1,
        "start_ns": low,
        "end_ns": high,
        "actual_target_kernel_records": len(kernels),
        "all_target_kernels_contained": True,
        "actual_target_activity_records": len(activities),
        "all_target_cuda_activities_contained": True,
        "optional_activity_tables_present": [
            name for name in optional_tables if name in available_tables
        ],
        "optional_activity_tables_absent": [
            name for name in optional_tables if name not in available_tables
        ],
        "activity_scope": "Recorded activity rows only; Nsight omits tables for absent event types",
        "scope": "Recorded diagnostic intervals, not occupancy or throughput",
    }
