"""Pause only the coordinator, draining its active child before diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pause", "resume"))
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    proc = Path(f"/proc/{args.pid}")
    if b"paired.py" not in (proc / "cmdline").read_bytes():
        parser.error("target is not a paired benchmark coordinator")
    if args.action == "resume":
        state = json.loads(args.audit.read_text())
        if state["parent_pid"] != args.pid:
            parser.error("audit PID differs")
        state["resumed_at"] = time.time()
        state["phase"] = "resumed"
        args.audit.write_text(json.dumps(state, indent=2))
        os.kill(args.pid, signal.SIGCONT)
        return 0
    os.kill(args.pid, signal.SIGSTOP)
    for _ in range(100):
        if (proc / "stat").read_text().split()[2] == "T":
            break
        time.sleep(0.01)
    else:
        os.kill(args.pid, signal.SIGCONT)
        parser.error("coordinator did not stop")
    children = [
        int(value) for value in (proc / f"task/{args.pid}/children").read_text().split()
    ]
    state = {
        "parent_pid": args.pid,
        "paused_at": time.time(),
        "phase": "paused",
        "exclude_affected_process_wall_only": True,
        "children": [],
    }
    for pid in children:
        child = Path(f"/proc/{pid}")
        if child.exists():
            state["children"].append(
                {
                    "pid": pid,
                    "command": (child / "cmdline").read_bytes().decode().split("\0"),
                }
            )
    args.audit.write_text(json.dumps(state, indent=2))
    deadline = time.monotonic() + 300
    for pid in children:
        child = Path(f"/proc/{pid}")
        while child.exists() and (child / "stat").read_text().split()[2] != "Z":
            if time.monotonic() >= deadline:
                os.kill(args.pid, signal.SIGCONT)
                state["phase"] = "drain_timeout_resumed"
                args.audit.write_text(json.dumps(state, indent=2))
                return 1
            time.sleep(1)
    state["child_finished_at"] = time.time()
    args.audit.write_text(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
