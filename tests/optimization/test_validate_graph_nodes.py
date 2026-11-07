"""Node visibility must cover host graph launches, not merely direct kernels."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.optimization.validate_graph_nodes import validate_graph_nodes


class GraphNodes(unittest.TestCase):
    def probe(self, case):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "trace.sqlite"
            with sqlite3.connect(p) as c:
                c.executescript(
                    "CREATE TABLE PROCESSES(pid,globalPid);CREATE TABLE StringIds(id,value);CREATE TABLE NVTX_EVENTS(start,end,globalTid,text,textId);CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid,correlationId,graphNodeId);CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,globalPid,correlationId,graphNodeId);CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start,end,globalPid,correlationId,graphNodeId);CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId,nameId,returnValue);CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start,end,globalPid);INSERT INTO PROCESSES VALUES(12,201326592);INSERT INTO NVTX_EVENTS VALUES(10,100,201326592,'whole vmecpp.run',NULL);INSERT INTO StringIds VALUES(1,'cudaGraphLaunch_v10000');INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(20,22,201326592,7,1,0);INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(40,42,201326592,8,1,0);INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(80,81,201326592,9,NULL);"
                )
                if case != "direct-only":
                    c.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(30,31,201326592,7,1)"
                    )
                    if case != "missing-launch":
                        c.execute(
                            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(50,51,201326592,8,2)"
                        )
                if case == "graph-mode":
                    c.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(25,35,201326592)"
                    )
            return validate_graph_nodes(p, 12)

    def test_positive_every_launch_has_actual_node(self):
        assert self.probe("valid")["every_successful_host_launch_has_node_activity"]

    def test_direct_only_graph_mode_or_uncovered_launch_fails(self):
        for case in ("direct-only", "graph-mode", "missing-launch"):
            with self.assertRaisesRegex(ValueError, "graph|launch"):  # noqa: PT027
                self.probe(case)


if __name__ == "__main__":
    unittest.main()
