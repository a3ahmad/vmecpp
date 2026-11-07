"""Require actual graph-node activity for every successful target host launch."""

import sqlite3

from tools.optimization.validate_profile_marker import validate_marker


def validate_graph_nodes(database, worker_pid):
    marker = validate_marker(database, worker_pid)
    gp = marker["global_pid"]
    low, high = marker["start_ns"], marker["end_ns"]
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        graphs = (
            connection.execute(
                "SELECT start,end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE globalPid=?",
                (gp,),
            ).fetchall()
            if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables
            else []
        )
        if graphs:
            message = "Node-mode diagnostic unexpectedly contains whole-graph spans"
            raise ValueError(message)
        launches = connection.execute(
            "SELECT r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId "
            "WHERE r.globalTid>=? AND r.globalTid<? AND r.start>=? AND r.end<=? "
            "AND s.value LIKE 'cudaGraphLaunch%' AND r.returnValue=0",
            (gp, gp + (1 << 24), low, high),
        ).fetchall()
        if not launches:
            message = "Actual successful target host graph launches required"
            raise ValueError(message)
        launch_ids = {row[0] for row in launches}
        kernel_nodes = connection.execute(
            "SELECT correlationId,graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? "
            "AND graphNodeId IS NOT NULL AND graphNodeId!=0",
            (gp,),
        ).fetchall()
        if not kernel_nodes:
            message = "Actual replayed graph kernel nodes required; direct kernels do not suffice"
            raise ValueError(message)
        node_correlations = {row[0] for row in kernel_nodes}
        all_node_records = len(kernel_nodes)
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table in tables:
                rows = connection.execute(
                    f"SELECT correlationId FROM {table} WHERE globalPid=? AND graphNodeId IS NOT NULL AND graphNodeId!=0",
                    (gp,),
                ).fetchall()
                all_node_records += len(rows)
                node_correlations.update(row[0] for row in rows)
        missing = launch_ids - node_correlations
        if missing:
            message = "Host graph launch lacks recorded node activity"
            raise ValueError(message)
    return {
        "exact": True,
        "worker_pid": worker_pid,
        "successful_host_graph_launch_records": len(launches),
        "distinct_host_graph_launch_correlations": len(launch_ids),
        "actual_graph_kernel_node_records": len(kernel_nodes),
        "actual_graph_node_activity_records": all_node_records,
        "every_successful_host_launch_has_node_activity": True,
        "uncovered_host_launches": 0,
        "whole_graph_trace_records": 0,
        "scope": "Recorded per-launch node visibility, not a proof of every possible internal node or occupancy",
    }
