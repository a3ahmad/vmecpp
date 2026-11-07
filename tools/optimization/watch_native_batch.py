"""Bounded read-only native-batch watcher; retrieve archives only after terminal."""

# CLI progress is intentional.
# ruff: noqa: T201
import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def command_for(args):
    """Build an explicit SSH status-read command without opening a connection."""
    remote = [
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "nice",
        "-n",
        str(args.remote_nice),
        "taskset",
        "-c",
        str(args.remote_cpu),
        args.remote_python,
        "-B",
        "-",
    ]
    return [
        str(args.ssh_executable),
        "-F",
        str(args.ssh_config),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "SendEnv=-*",
        args.ssh_host,
        shlex.join(remote),
    ]


def program_for(remote_root):
    """Encode the supplied status directory as data in the read-only program."""
    return """
import json
from pathlib import Path
root=Path(ROOT_LITERAL)
status=json.loads((root/'campaign-status.json').read_text())
process=Path('/proc')/str(status['pid'])
live=False
if process.exists():
    fields=(process/'stat').read_text().rsplit(')',1)[1].split()
    live=fields[19]==status['birth'] and fields[0]!='Z'
status['matching_coordinator_birth_live']=live
print(json.dumps(status))
""".replace("ROOT_LITERAL", json.dumps(remote_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ssh-executable", type=Path, required=True)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--remote-cpu", type=int, required=True)
    parser.add_argument("--remote-nice", type=int, choices=range(20), required=True)
    parser.add_argument("--expected-evaluations", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.ssh_host.startswith("-"):
        parser.error("SSH host must be a hostname or configured alias")
    if args.remote_cpu < 0 or args.expected_evaluations < 1:
        parser.error("CPU must be nonnegative and evaluation count positive")
    command = command_for(args)
    program = program_for(args.remote_root)
    if not args.execute:
        print(
            "PLAN ONLY: small status reads every45seconds, max40minutes; no GPU/archive work"
        )
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    environment = {
        name: value
        for name, value in os.environ.items()
        if not any(
            token in name.upper()
            for token in (
                "VAST",
                "API_KEY",
                "TOKEN",
                "SECRET",
                "PASSWORD",
                "CREDENTIAL",
                "ACCESS_KEY",
                "PRIVATE_KEY",
            )
        )
    }
    state = {"phase": "watching", "started_at": time.time()}
    started, previous = time.monotonic(), -1
    while time.monotonic() - started < 2400:
        try:
            response = subprocess.run(
                command,
                input=program,
                capture_output=True,
                text=True,
                env=environment,
                check=True,
                timeout=25,
            )
            actual = json.loads(response.stdout)
            (args.out / "latest-remote-status.json").write_text(
                json.dumps(actual, indent=2) + "\n"
            )
            state.update(
                remote_phase=actual["phase"],
                completed_evaluations=actual["completed_evaluations"],
                last_error=None,
            )
            if actual["completed_evaluations"] != previous:
                print(
                    f"Native batch {actual['completed_evaluations']}/"
                    f"{args.expected_evaluations} phase={actual['phase']}",
                    flush=True,
                )
                previous = actual["completed_evaluations"]
            terminal = (
                actual["phase"] == "complete"
                and actual["completed_evaluations"] == args.expected_evaluations
                and actual.get("final_metadata")
                and actual.get("owned_children_reaped")
                and not actual["matching_coordinator_birth_live"]
            )
            failed = (
                actual["phase"] == "failed"
                and actual.get("owned_children_reaped")
                and not actual["matching_coordinator_birth_live"]
            )
            if terminal or failed:
                state.update(
                    phase="terminal_ready_for_retrieval", finished_at=time.time()
                )
                (args.out / "watcher-status.json").write_text(
                    json.dumps(state, indent=2) + "\n"
                )
                print("Terminal native batch: " + actual["phase"], flush=True)
                return 0
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            state["last_error"] = type(error).__name__
        state["updated_at"] = time.time()
        (args.out / "watcher-status.json").write_text(
            json.dumps(state, indent=2) + "\n"
        )
        time.sleep(45)
    state["phase"] = "bounded_watch_expired"
    (args.out / "watcher-status.json").write_text(json.dumps(state, indent=2) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
