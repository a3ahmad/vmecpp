"""Incrementally retrieve pilot results until its lifecycle state is destroyed."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ssh = shlex.join(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            f"UserKnownHostsFile={args.known_hosts}",
            "-p",
            str(args.port),
        ]
    )
    while json.loads(args.state.read_text()).get("phase") != "destroyed":
        try:
            result = subprocess.run(
                [
                    "rsync",
                    "-az",
                    "--partial",
                    "-e",
                    ssh,
                    f"root@{args.host}:/opt/vmecpp-experiment/results/",
                    str(args.out) + "/",
                ],
                timeout=120,
                check=False,
            )
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
        (args.out / "last-sync.json").write_text(
            json.dumps(
                {
                    "completed_at": time.time(),
                    "exit_status": status,
                }
            )
        )
        for _ in range(4):
            if json.loads(args.state.read_text()).get("phase") == "destroyed":
                return 0
            time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
