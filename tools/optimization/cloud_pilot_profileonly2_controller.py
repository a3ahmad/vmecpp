"""Control an existing one-hour pilot; never create or inspect provider resources.

The owner uploads/extracts the frozen payload before execution. Plan is the default.
Only the existing local lease.py destroy operation receives inherited credentials.
"""

from __future__ import annotations

# CLI output is intentional; unexpected failures expose only their class name.
# ruff: noqa: T201, BLE001
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

SSH_ENVIRONMENT = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "APPDATA",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
}
SUPERVISOR = r"""
import contextlib, ctypes, json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
claim = json.loads((root / "controller-launch.json").read_text())
nonce = claim["nonce"]
def save(value):
    temporary = root / "supervisor-status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(root / "supervisor-status.json")
def children(parent):
    result = []
    path = Path(f"/proc/{parent}/task/{parent}/children")
    with contextlib.suppress(FileNotFoundError, ProcessLookupError):
        for value in path.read_text().split():
            pid = int(value)
            stat = Path(f"/proc/{pid}/stat")
            with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                start = stat.read_text().rsplit(")", 1)[1].split()[19]
                result.append((pid, start))
                result.extend(children(pid))
    return result
def cleanup():
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, start in reversed(children(os.getpid())):
            with contextlib.suppress(ProcessLookupError, FileNotFoundError):
                current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                if current == start:
                    os.kill(pid, sig)
        time.sleep(1)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
    return not children(os.getpid())
if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
    raise RuntimeError("Cannot establish owned-child supervision")
env = {key: value for key, value in os.environ.items()
       if key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "CUDA_VISIBLE_DEVICES")}
env["PYTHONPATH"] = str(root)
env["PYTHONDONTWRITEBYTECODE"] = "1"
state = {"nonce": nonce, "pid": os.getpid(), "phase": "running", "started_at": time.time(),
         "process_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
         "expires_at": claim["expires_at"], "reserve_seconds": claim["reserve_seconds"], "stages": []}
save(state)
code = 1
try:
    for action in ("bootstrap", "run"):
        remaining = claim["expires_at"] - claim["reserve_seconds"] - time.time()
        if remaining <= 0:
            code = 124
            break
        command = [sys.executable, str(root / "tools/optimization/cloud_pilot.py"), action,
                   "--root", str(root), "--execute", "--lease-started-at", str(claim["created_at"]),
                   "--lease-expires-at", str(claim["expires_at"])]
        if action == "run":
            command += ["--max-pairs", str(claim["max_pairs"])]
            if claim["include_optional_tools"]:
                command.append("--include-optional-tools")
        with (root / "logs" / ("supervisor-" + action + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            stage = {"action": action, "pid": process.pid, "started_at": time.time()}
            state["stages"].append(stage)
            save(state)
            try:
                code = process.wait(timeout=max(1, remaining))
            except subprocess.TimeoutExpired:
                code = 124
                cleanup()
            stage.update(return_code=code, finished_at=time.time())
            save(state)
        if code != 0:
            break
except BaseException as error:
    state["failure_type"] = type(error).__name__
    code = 1
finally:
    clean = cleanup()
    state.update(phase="terminal" if clean else "cleanup_unconfirmed", return_code=code,
                 owned_children_reaped=clean, finished_at=time.time())
    save(state)
raise SystemExit(code if clean else 125)
"""


class TransientTransport(RuntimeError):
    """An SSH/scp failure whose completion state is unknown."""


class DuplicateLaunch(RuntimeError):
    """Another nonce already owns the remote root."""


class StartupUnconfirmed(RuntimeError):
    """No matching live supervisor appeared within the bounded startup window."""


def transport_environment(environment: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in environment.items()
        if key.upper() in SSH_ENVIRONMENT
    }


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lease_identity(state: dict, now: float, reserve: int) -> dict:
    if not re.fullmatch(r"vmecpp-opt-[0-9]+", str(state.get("label", ""))):
        message = "Lease label is not an optimization lease"
        raise ValueError(message)
    result = {
        name: state[name]
        for name in ("label", "instance_id", "created_at", "expires_at")
    }
    if not isinstance(result["instance_id"], int) or result["instance_id"] <= 0:
        message = "Lease has no confirmed instance"
        raise ValueError(message)
    if not 0 < result["expires_at"] - result["created_at"] <= 3600:
        message = "Pilot requires an actual lease window of at most one hour"
        raise ValueError(message)
    if reserve < 600 or result["expires_at"] - now <= reserve:
        message = "Insufficient actual lease time after retrieval reserve"
        raise ValueError(message)
    if not isinstance(state.get("watchdog_pid"), int) or state["watchdog_pid"] <= 0:
        message = "Independent lease watchdog is required"
        raise ValueError(message)
    return result


def launch_program(configuration: dict) -> str:
    return (
        "import json,os,subprocess,sys\nfrom pathlib import Path\n"
        + "cfg=json.loads("
        + repr(json.dumps(configuration))
        + ")\n"
        + "root=Path(cfg['root']); claim=root/'controller-launch.json'\n"
        + "if claim.exists():\n"
        + " old=json.loads(claim.read_text()); print(json.dumps({'duplicate':old['nonce']!=cfg['nonce'],'nonce':old['nonce'],'pid':old.get('pid'),'phase':'claimed'})); raise SystemExit(0)\n"
        + "assert sys.version_info >= (3,12)\n"
        + "assert (root/'payload-manifest.json').is_file() and not (root/'bootstrap-status.json').exists() and not (root/'pilot-status.json').exists()\n"
        + "fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\n"
        + "with os.fdopen(fd,'w') as f: json.dump(cfg,f); f.flush(); os.fsync(f.fileno())\n"
        + "source=root/'controller-supervisor.py'\nsource.write_text("
        + repr(SUPERVISOR)
        + ")\n"
        + "(root/'logs').mkdir(exist_ok=True)\n"
        + "env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','LC_ALL','TMPDIR')}\n"
        + "with (root/'logs/controller-supervisor.log').open('x') as log:\n"
        + " p=subprocess.Popen([sys.executable,str(source),str(root)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
        + "cfg['pid']=p.pid; temp=claim.with_suffix('.tmp'); temp.write_text(json.dumps(cfg)); temp.replace(claim)\n"
        + "print(json.dumps({'nonce':cfg['nonce'],'pid':p.pid,'phase':'launched'}))\n"
    )


def diagnostic_success(pilot, expected_prior_sha):
    evaluations = pilot.get("completed_diagnostic_evaluations", [])
    return (
        isinstance(expected_prior_sha, str)
        and isinstance(evaluations, list)
        and pilot.get("phase") == "diagnostic_complete"
        and pilot.get("diagnostic_complete") is True
        and pilot.get("diagnostic_mode") == "anchor-plus-profile"
        and pilot.get("prior_canonical_evidence_sha256") == expected_prior_sha
        and re.fullmatch(r"[0-9a-f]{64}", expected_prior_sha) is not None
        and len(evaluations) == 2
        and [item.get("label") for item in evaluations] == ["anchor", "profiled"]
        and evaluations[0].get("profiled") is False
        and evaluations[1].get("profiled") is True
        and all(item.get("gate", {}).get("exact") is True for item in evaluations)
        and pilot.get("anchor_profile_raw_comparison", {}).get("exact") is True
        and bool(pilot.get("anchor_profile_raw_comparison", {}).get("numeric_fields"))
        and all(
            pilot.get("nsys_report", {}).get(key) is True
            for key in (
                "created",
                "target_worker_terminal_reaped",
                "sqlite_export_passed",
                "actual_worker_kernel_node_evidence",
            )
        )
        and pilot.get("nsys_report", {})
        .get("actual_nvtx_marker_and_kernel_containment", {})
        .get("exact")
        is True
        and pilot.get("nsys_report", {})
        .get("actual_nvtx_marker_and_kernel_containment", {})
        .get("all_target_cuda_activities_contained")
        is True
        and all(
            pilot.get(key) is True
            for key in (
                "owned_children_reaped",
                "runtime_unchanged",
                "immutable_payload_unchanged",
            )
        )
        and "case_pairs" not in pilot
        and "canonical_parity_complete" not in pilot
    )


class Controller:
    def __init__(self, args, identity: dict, state: dict):
        self.args = args
        self.identity = identity
        self.state = state
        self.environment = transport_environment(dict(os.environ))
        self.downloaded = 0
        self.sequence = 0
        self.owns_lease = True

    def persist(self) -> None:
        self.state["updated_at"] = time.time()
        save(self.args.out / "controller-state.json", self.state)

    def timeout(self, maximum: int) -> int:
        return max(1, min(maximum, int(self.identity["expires_at"] - time.time() - 90)))

    def pause(self) -> None:
        delay = self.args.poll_seconds
        if (
            not self.state.get("launch_acknowledged")
            and "launch_started_at" in self.state
        ):
            remaining = (
                self.state["launch_started_at"]
                + self.args.startup_seconds
                - time.time()
            )
            if remaining <= 0:
                message = "Supervisor startup proof deadline reached"
                raise StartupUnconfirmed(message)
            delay = min(delay, remaining)
        time.sleep(delay)

    def options(self, scp: bool = False) -> list[str]:
        result = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "UserKnownHostsFile=" + str(self.args.known_hosts),
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "SendEnv=-*",
            "-o",
            "ConnectTimeout=15",
            "-P" if scp else "-p",
            str(self.args.ssh_port),
        ]
        if self.args.identity:
            result += ["-i", str(self.args.identity), "-o", "IdentitiesOnly=yes"]
        return result

    def rpc(self, program: str, maximum: int = 120) -> dict:
        if (
            not self.state.get("launch_acknowledged")
            and "launch_started_at" in self.state
        ):
            remaining = (
                self.state["launch_started_at"]
                + self.args.startup_seconds
                - time.time()
            )
            if remaining <= 0:
                message = "Supervisor startup proof deadline reached"
                raise StartupUnconfirmed(message)
            maximum = max(1, min(maximum, int(remaining)))
        command = [
            "ssh",
            *self.options(),
            "-T",
            self.args.ssh_user + "@" + self.args.ssh_host,
            shlex.join([self.args.remote_python, "-"]),
        ]
        try:
            result = subprocess.run(
                command,
                input=program,
                capture_output=True,
                text=True,
                env=self.environment,
                timeout=self.timeout(maximum),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TransientTransport(type(error).__name__) from None
        if result.returncode:
            msg = "ssh_exit"
            raise TransientTransport(msg)
        try:
            response = json.loads(result.stdout)
            if not isinstance(response, dict):
                raise TypeError
            return response
        except (ValueError, TypeError):
            msg = "ssh_response"
            raise TransientTransport(msg) from None

    def start(self) -> None:
        started = self.state.setdefault("launch_started_at", time.time())
        remaining = self.args.startup_seconds - (time.time() - started)
        if remaining <= 0:
            message = "Supervisor startup proof deadline reached"
            raise StartupUnconfirmed(message)
        configuration = {
            **self.identity,
            "root": self.args.remote_root,
            "nonce": self.state["nonce"],
            "reserve_seconds": self.args.reserve_seconds,
            "max_pairs": self.args.max_pairs,
            "include_optional_tools": self.args.include_optional_tools,
        }
        result = self.rpc(
            launch_program(configuration), max(1, min(15, int(remaining)))
        )
        if result.get("duplicate") or result.get("nonce") != self.state["nonce"]:
            self.owns_lease = False
            message = "Remote root already belongs to another controller"
            raise DuplicateLaunch(message)
        self.state.update(
            launcher_reported_pid=result.get("pid"), launch_claim_seen=True
        )

    def status(self) -> dict:
        program = (
            "import json\nfrom pathlib import Path\np=Path("
            + repr(self.args.remote_root + "/supervisor-status.json")
            + ")\nr=json.loads(p.read_text()) if p.is_file() else {'phase':'pending'}\n"
            + "s=Path('/proc/'+str(r.get('pid',0))+'/stat')\n"
            + "try: r['pid_alive']=s.read_text().rsplit(')',1)[1].split()[19]==r.get('process_start_ticks')\n"
            + "except FileNotFoundError: r['pid_alive']=False\nprint(json.dumps(r))\n"
        )
        result = self.rpc(
            program, 15 if not self.state.get("launch_acknowledged") else 30
        )
        if (
            result.get("phase") != "pending"
            and result.get("nonce") != self.state["nonce"]
        ):
            message = "Supervisor nonce does not match this controller"
            raise DuplicateLaunch(message)
        pid = result.get("pid")
        expected = self.state.get("launcher_reported_pid")
        confirmed = (
            isinstance(pid, int)
            and pid > 0
            and (expected is None or expected == pid)
            and (
                (result.get("phase") == "running" and result.get("pid_alive") is True)
                or (
                    result.get("phase") == "terminal"
                    and result.get("owned_children_reaped") is True
                )
            )
        )
        if confirmed:
            self.state.update(launch_acknowledged=True, supervisor_pid=pid)
        safe = {
            key: result[key]
            for key in (
                "phase",
                "nonce",
                "pid",
                "return_code",
                "started_at",
                "finished_at",
                "owned_children_reaped",
                "pid_alive",
                "process_start_ticks",
            )
            if key in result
        }
        self.state["remote_supervisor"] = safe
        return safe

    def collect(self, *, final: bool) -> None:
        if final and self.state.get("remote_supervisor", {}).get("phase") != "terminal":
            message = "Final collection requires observed supervisor terminal status"
            raise ValueError(message)
        tool = self.args.remote_root + "/tools/optimization/cloud_pilot.py"
        program = (
            "import subprocess,sys\nr=subprocess.run("
            + repr(
                [
                    self.args.remote_python,
                    tool,
                    "collect",
                    "--root",
                    self.args.remote_root,
                ]
            )
            + ",stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)\nsys.stdout.write(r.stdout); raise SystemExit(r.returncode)\n"
        )
        metadata = self.rpc(program, 180)
        remote = PurePosixPath(metadata.get("archive", ""))
        if remote.parent != PurePosixPath(
            self.args.remote_root
        ) / "collections" or not re.fullmatch(
            r"pilot-results-[0-9]+\.tar\.gz", remote.name
        ):
            message = "Unexpected collection archive path"
            msg = "collection_path"
            raise TransientTransport(msg)
        sha = metadata.get("sha256", "")
        size = metadata.get("bytes")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", sha)
            or not isinstance(size, int)
            or size < 0
            or size > self.args.max_archive_bytes
        ):
            message = "Invalid or oversized collection metadata"
            msg = "collection_metadata"
            raise TransientTransport(msg)
        if self.downloaded + size > self.args.max_download_bytes:
            message = "Pilot retrieval transfer budget reached"
            msg = "collection_budget"
            raise TransientTransport(msg)
        self.sequence += 1
        name = f"snapshot-{self.sequence:04d}"
        archive = self.args.out / (name + ".tar.gz")
        destination = self.args.out / name
        host = (
            "[" + self.args.ssh_host + "]"
            if ":" in self.args.ssh_host
            else self.args.ssh_host
        )
        command = [
            "scp",
            *self.options(True),
            self.args.ssh_user + "@" + host + ":" + str(remote),
            str(archive),
        ]
        self.downloaded += size
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self.environment,
                timeout=self.timeout(180),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TransientTransport(type(error).__name__) from None
        if result.returncode or not archive.is_file() or digest(archive) != sha:
            msg = "collection_transfer"
            raise TransientTransport(msg)
        try:
            verification = subprocess.run(
                [
                    str(self.args.verifier_python),
                    str(self.args.cloud_pilot_tool),
                    "verify-collection",
                    "--root",
                    self.args.remote_root,
                    "--archive",
                    str(archive),
                    "--expected-sha256",
                    sha,
                    "--destination",
                    str(destination),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=self.environment,
                timeout=self.timeout(180),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TransientTransport(type(error).__name__) from None
        if verification.returncode:
            msg = "collection_verification"
            raise TransientTransport(msg)
        try:
            proof = json.loads(verification.stdout)
        except (ValueError, TypeError):
            msg = "verification_response"
            raise TransientTransport(msg) from None
        if proof.get("exact") is not True:
            msg = "collection_verification"
            raise TransientTransport(msg)
        try:
            manifest = json.loads((destination / "retrieval-manifest.json").read_text())
            actual_files = {
                path.relative_to(destination).as_posix()
                for path in destination.rglob("*")
                if path.is_file() and path != destination / "retrieval-manifest.json"
            }
            if actual_files != set(manifest) or proof.get("verified_files") != len(
                manifest
            ):
                raise ValueError
            if final and "supervisor-status.json" not in manifest:
                raise ValueError
        except (OSError, ValueError, TypeError):
            msg = "collection_manifest_coverage"
            raise TransientTransport(msg) from None
        receipt = {
            "directory": str(destination),
            "archive_sha256": sha,
            "verified_files": proof["verified_files"],
            "final": final,
            "collected_at": time.time(),
        }
        if final:
            try:
                supervisor = json.loads(
                    (destination / "supervisor-status.json").read_text()
                )
            except (OSError, ValueError):
                msg = "missing_terminal_proof"
                raise TransientTransport(msg) from None
            if (
                supervisor.get("nonce") != self.state["nonce"]
                or supervisor.get("phase") != "terminal"
                or not supervisor.get("owned_children_reaped")
            ):
                message = "Final collection lacks matching terminal supervisor proof"
                msg = "mismatched_terminal_proof"
                raise TransientTransport(msg)
            self.state["final_retrieval_verified"] = True
            self.state["supervisor_return_code"] = supervisor.get("return_code")
            pilot = destination / "pilot-status.json"
            pilot_state = json.loads(pilot.read_text()) if pilot.is_file() else {}
            prior = destination / "results/provenance/prior-canonical-evidence.json"
            configuration = destination / "results/provenance/pilot-config.json"
            coverage = json.loads((destination / "retrieval-manifest.json").read_text())
            if any(
                str(path.relative_to(destination)) not in coverage
                for path in (prior, configuration)
            ):
                message = (
                    "Diagnostic collection lacks manifest-covered prior evidence/config"
                )
                raise TransientTransport(message)
            expected_prior = json.loads(configuration.read_text())[
                "prior_canonical_evidence_sha256"
            ]
            if digest(prior) != expected_prior:
                message = "Diagnostic prior evidence receipt hash mismatch"
                raise TransientTransport(message)
            self.state["diagnostic_evaluations"] = len(
                pilot_state.get("completed_diagnostic_evaluations", [])
            )
            self.state["pilot_succeeded"] = supervisor.get(
                "return_code"
            ) == 0 and diagnostic_success(pilot_state, expected_prior)
        self.state.setdefault("snapshots", []).append(receipt)
        self.state["downloaded_bytes"] = self.downloaded

    def destroy(self) -> bool:
        while time.time() < self.identity["expires_at"] + 120:
            lease = json.loads(self.args.lease_state.read_text())
            if any(lease.get(key) != value for key, value in self.identity.items()):
                self.state["teardown_error"] = "lease_identity_changed"
                return False
            if (
                lease.get("phase") == "destroyed"
                and lease.get("teardown_verified") is True
            ):
                self.state["teardown_verified"] = True
                return True
            try:
                # This one LOCAL command intentionally inherits the provider credential.
                subprocess.run(
                    [
                        str(self.args.tools_python),
                        str(self.args.lease_tool),
                        "destroy",
                        "--state",
                        str(self.args.lease_state),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=90,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                self.state["teardown_error"] = "destroy_retry"
            self.persist()
            time.sleep(5)
        return False

    def run(self) -> int:
        next_collection = 0.0
        try:
            while time.time() < self.identity["expires_at"] - 90:
                lease = json.loads(self.args.lease_state.read_text())
                if any(lease.get(key) != value for key, value in self.identity.items()):
                    self.owns_lease = False
                    msg = "lease_identity_changed"
                    raise DuplicateLaunch(msg)
                if lease.get("phase") == "destroyed":
                    break
                try:
                    if not self.state.get("launch_acknowledged"):
                        self.state["phase"] = "launch_attempted"
                        self.persist()
                        self.start()
                    supervisor = self.status()
                    if not self.state.get("launch_acknowledged"):
                        self.state["phase"] = "startup_pending"
                        self.persist()
                        self.pause()
                        continue
                    terminal = supervisor.get("phase") == "terminal"
                    if terminal or time.time() >= next_collection:
                        self.collect(final=terminal)
                        next_collection = time.time() + self.args.collect_seconds
                    self.state["phase"] = "collecting_final" if terminal else "running"
                    self.persist()
                    if self.state.get("final_retrieval_verified"):
                        break
                except TransientTransport as error:
                    self.state.update(
                        phase="transport_retry", last_transport_error=str(error)
                    )
                    self.persist()
                self.pause()
        except DuplicateLaunch:
            self.owns_lease = False
            self.state["phase"] = "duplicate_launch_refused"
        except BaseException as error:
            self.state.update(
                phase="controller_error", failure_type=type(error).__name__
            )
        finally:
            if self.owns_lease:
                self.destroy()
            evidence_complete = bool(
                self.state.get("final_retrieval_verified")
                and self.state.get("teardown_verified")
            )
            self.state["evidence_and_teardown_complete"] = evidence_complete
            success = evidence_complete and bool(self.state.get("pilot_succeeded"))
            self.state["complete"] = success
            if self.owns_lease:
                self.state["phase"] = "complete" if success else "incomplete"
            self.persist()
        return 0 if self.state.get("complete") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--lease-state", type=Path)
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--remote-root")
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tools-python", type=Path)
    parser.add_argument("--verifier-python", type=Path)
    parser.add_argument(
        "--lease-tool", type=Path, default=Path(__file__).with_name("lease.py")
    )
    parser.add_argument(
        "--cloud-pilot-tool",
        type=Path,
        default=Path(__file__).with_name("cloud_pilot_profileonly2.py"),
    )
    parser.add_argument("--reserve-seconds", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--collect-seconds", type=int, default=120)
    parser.add_argument("--startup-seconds", type=int, default=60)
    parser.add_argument("--max-pairs", type=int, choices=(2,), default=2)
    parser.add_argument("--include-optional-tools", action="store_true")
    parser.add_argument("--max-archive-bytes", type=int, default=1024**3)
    parser.add_argument("--max-download-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args()
    if args.include_optional_tools:
        parser.error("Profile-only diagnostic excludes optional probes")
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "creates_or_inspects_leases": False,
                    "pipeline": [
                        "once-only detached bootstrap/run supervisor",
                        "incremental collection and Python3.12 verification",
                        "terminal full retrieval",
                        "local label-checked lease.py destroy",
                    ],
                    "retrieval_reserve_seconds": args.reserve_seconds,
                    "requires": [
                        "existing one-hour labeled lease and watchdog",
                        "uploaded frozen payload",
                        "pinned known-hosts",
                        "local provider credential only for destroy",
                    ],
                },
                indent=2,
            )
        )
        return 0
    required = (
        args.lease_state,
        args.ssh_host,
        args.known_hosts,
        args.remote_root,
        args.out,
        args.tools_python,
        args.verifier_python,
    )
    if not all(required):
        parser.error(
            "execution requires lease-state, SSH endpoint, known-hosts, remote-root, out and both local Python paths"
        )
    if not re.fullmatch(r"/opt/vmecpp-pilot-[A-Za-z0-9_-]+", args.remote_root):
        parser.error("remote-root must be a fresh /opt/vmecpp-pilot-* directory")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.:-]*", args.ssh_host
    ) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", args.ssh_user):
        parser.error("invalid SSH endpoint")
    if (
        not 1 <= args.ssh_port <= 65535
        or args.poll_seconds < 1
        or args.collect_seconds < 1
        or not 15 <= args.startup_seconds <= 120
    ):
        parser.error("invalid port or polling interval")
    if (
        args.max_archive_bytes <= 0
        or args.max_download_bytes <= 0
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", args.remote_python)
    ):
        parser.error("invalid transfer budget or remote Python")
    for path in (
        args.lease_state,
        args.known_hosts,
        args.tools_python,
        args.verifier_python,
        args.lease_tool,
        args.cloud_pilot_tool,
    ):
        if not path.is_file():
            parser.error("a required local file is absent")
    args.out = args.out.resolve()
    identity = lease_identity(
        json.loads(args.lease_state.read_text()), time.time(), args.reserve_seconds
    )
    version = subprocess.run(
        [
            str(args.verifier_python),
            "-c",
            "import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=transport_environment(dict(os.environ)),
        timeout=15,
        check=False,
    )
    if version.returncode:
        parser.error("verifier-python must be Python3.12 or newer")
    args.out.mkdir(parents=True, exist_ok=False)
    state = {
        "nonce": uuid.uuid4().hex,
        "phase": "prepared",
        "lease": identity,
        "final_retrieval_verified": False,
        "teardown_verified": False,
        "complete": False,
    }
    controller = Controller(args, identity, state)
    controller.persist()
    return controller.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Pilot controller failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
