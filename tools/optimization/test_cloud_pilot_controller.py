"""CPU-only lifecycle tests; no provider, SSH or GPU operations are executed."""

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from tools.optimization import cloud_pilot_controller as controller


@contextlib.contextmanager
def raises(error_type):
    try:
        yield
    except error_type:
        return
    message = "Expected controller rejection"
    raise AssertionError(message)


def fixture(root):
    identity = {
        "label": "vmecpp-opt-1",
        "instance_id": 12,
        "created_at": 900,
        "expires_at": 4500,
    }
    lease = root / "lease.json"
    lease.write_text(json.dumps({**identity, "phase": "created", "watchdog_pid": 77}))
    args = types.SimpleNamespace(
        out=root,
        lease_state=lease,
        reserve_seconds=600,
        ssh_host="example.invalid",
        ssh_port=22,
        ssh_user="root",
        known_hosts=root / "known",
        identity=None,
        remote_python="python3",
        remote_root="/opt/vmecpp-pilot-test",
        max_pairs=1,
        include_optional_tools=False,
        max_archive_bytes=1024**3,
        max_download_bytes=4 * 1024**3,
        verifier_python=Path(sys.executable),
        cloud_pilot_tool=Path(__file__).with_name("cloud_pilot.py"),
        tools_python=Path(sys.executable),
        lease_tool=root / "lease.py",
        poll_seconds=1,
        collect_seconds=120,
        startup_seconds=60,
    )
    state = {
        "nonce": "nonce",
        "phase": "prepared",
        "final_retrieval_verified": False,
        "teardown_verified": False,
        "complete": False,
    }
    return controller.Controller(args, identity, state)


class ControllerTest(unittest.TestCase):
    def test_default_plan_never_calls_subprocess_or_reads_a_lease(self):
        with (
            mock.patch.object(sys, "argv", ["controller"]),
            mock.patch.object(controller.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            assert controller.main() == 0
        run.assert_not_called()
        assert json.loads(output.getvalue())["execute"] is False

    def test_transport_environment_is_allowlisted(self):
        result = controller.transport_environment(
            {
                "PATH": "/bin",
                "HOME": "/safe",
                "VAST_KEY": "dummy-only",
                "VAST_API_KEY": "dummy-only",
                "AWS_SECRET_ACCESS_KEY": "dummy-only",
                "UNRECOGNIZED_PROVIDER_TOKEN": "dummy-only",
                "SSH_AUTH_SOCK": "/agent",
                "LD_PRELOAD": "/bad.so",
            }
        )
        assert result == {"PATH": "/bin", "HOME": "/safe", "SSH_AUTH_SOCK": "/agent"}

    def test_real_lease_duration_and_retrieval_reserve_are_enforced(self):
        state = {
            "label": "vmecpp-opt-123",
            "instance_id": 12,
            "created_at": 900,
            "expires_at": 4500,
            "watchdog_pid": 77,
        }
        assert controller.lease_identity(state, 1000, 600)["expires_at"] == 4500
        for changed in (
            {"expires_at": 4501},
            {"label": "unrelated"},
            {"instance_id": 0},
            {"watchdog_pid": None},
        ):
            with raises(ValueError):
                controller.lease_identity({**state, **changed}, 1000, 600)
        with raises(ValueError):
            controller.lease_identity(state, 3950, 600)

    def test_supervisor_and_launch_program_compile(self):
        compile(controller.SUPERVISOR, "supervisor", "exec")
        compile(
            controller.launch_program({"root": "/opt/vmecpp-pilot-test", "nonce": "n"}),
            "launch",
            "exec",
        )

    def test_real_local_supervisor_reaps_owned_detached_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = root / "tools/optimization/cloud_pilot.py"
            tool.parent.mkdir(parents=True)
            (root / "logs").mkdir()
            tool.write_text(
                "import subprocess,sys\nfrom pathlib import Path\nr=Path(sys.argv[sys.argv.index('--root')+1])\nif sys.argv[1]=='bootstrap':\n p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)\n (r/'orphan-pid').write_text(str(p.pid))\n"
            )
            claim = {
                "nonce": "local-only",
                "created_at": time.time(),
                "expires_at": time.time() + 900,
                "reserve_seconds": 600,
                "max_pairs": 1,
                "include_optional_tools": False,
            }
            (root / "controller-launch.json").write_text(json.dumps(claim))
            source = root / "controller-supervisor.py"
            source.write_text(controller.SUPERVISOR)
            result = subprocess.run(
                [sys.executable, str(source), str(root)],
                capture_output=True,
                timeout=15,
                check=False,
            )
            assert result.returncode == 0
            state = json.loads((root / "supervisor-status.json").read_text())
            assert state["phase"] == "terminal"
            assert state["owned_children_reaped"] is True
            assert [stage["action"] for stage in state["stages"]] == [
                "bootstrap",
                "run",
            ]
            with raises(ProcessLookupError):
                os.kill(int((root / "orphan-pid").read_text()), 0)

    def test_lost_ack_retry_never_launches_a_second_supervisor(self):
        if sys.version_info < (3, 12):
            self.skipTest("Remote launch correctly requires Python3.12")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "payload-manifest.json").write_text("{}")
            configuration = {"root": str(root), "nonce": "same"}
            with (
                mock.patch.object(
                    subprocess, "Popen", return_value=types.SimpleNamespace(pid=123)
                ) as spawn,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                exec(controller.launch_program(configuration), {})
                with raises(SystemExit):
                    exec(controller.launch_program(configuration), {})
                with raises(SystemExit):
                    exec(
                        controller.launch_program({**configuration, "nonce": "other"}),
                        {},
                    )
            assert spawn.call_count == 1
            responses = [json.loads(line) for line in output.getvalue().splitlines()]
            assert responses[1]["duplicate"] is False
            assert responses[2]["duplicate"] is True

    def test_ssh_timeout_is_transient_and_does_not_log_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))
            with (
                mock.patch.object(
                    controller.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired(["ssh"], 10),
                ),
                raises(controller.TransientTransport),
            ):
                item.rpc("print('{}')")

    def test_ssh_and_scp_do_not_receive_provider_credentials(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(
                os.environ,
                {"VAST_KEY": "dummy-only", "OTHER_PROVIDER_KEY": "dummy-only"},
            ),
        ):
            item = fixture(Path(temporary))
            with mock.patch.object(
                controller.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "{}", ""),
            ) as run:
                item.rpc("print('{}')")
            environment = run.call_args.kwargs["env"]
            assert "VAST_KEY" not in environment
            assert "OTHER_PROVIDER_KEY" not in environment
            assert "StrictHostKeyChecking=yes" in run.call_args.args[0]

    def test_final_collection_cannot_start_before_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))
            with mock.patch.object(item, "rpc") as rpc, raises(ValueError):
                item.collect(final=True)
            rpc.assert_not_called()

    def test_transport_failure_retries_then_collects_terminal_before_destroy(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))
            attempts = []
            collections = []

            def start():
                attempts.append(1)
                if len(attempts) == 1:
                    msg = "TimeoutExpired"
                    raise controller.TransientTransport(msg)
                item.state["launch_acknowledged"] = True

            def status():
                value = {
                    "phase": "running" if not collections else "terminal",
                    "nonce": "nonce",
                }
                item.state["remote_supervisor"] = value
                return value

            def collect(*, final):
                collections.append(final)
                if final:
                    item.state.update(
                        final_retrieval_verified=True, pilot_succeeded=True
                    )

            def destroy():
                assert item.state["final_retrieval_verified"]
                item.state["teardown_verified"] = True

            with (
                mock.patch.object(controller.time, "time", return_value=1000),
                mock.patch.object(controller.time, "sleep"),
                mock.patch.object(item, "start", side_effect=start),
                mock.patch.object(item, "status", side_effect=status),
                mock.patch.object(item, "collect", side_effect=collect),
                mock.patch.object(item, "destroy", side_effect=destroy),
            ):
                assert item.run() == 0
            assert attempts == [1, 1]
            assert collections == [False, True]
            assert item.state["complete"]

    def test_retrieval_timeout_uses_retry_not_early_destroy(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))
            item.state["launch_acknowledged"] = True
            item.state["remote_supervisor"] = {"phase": "terminal"}
            calls = []

            def collect(*, final):
                assert final
                calls.append(1)
                if len(calls) == 1:
                    msg = "scp_timeout"
                    raise controller.TransientTransport(msg)
                item.state.update(final_retrieval_verified=True, pilot_succeeded=True)

            def destroy():
                assert len(calls) == 2
                item.state["teardown_verified"] = True

            with (
                mock.patch.object(controller.time, "time", return_value=1000),
                mock.patch.object(controller.time, "sleep"),
                mock.patch.object(item, "status", return_value={"phase": "terminal"}),
                mock.patch.object(item, "collect", side_effect=collect),
                mock.patch.object(item, "destroy", side_effect=destroy),
            ):
                assert item.run() == 0

    def test_bare_launch_claim_never_acknowledges_and_tears_down_within60(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))
            item.args.poll_seconds = 120
            clock = [1000]
            destroyed_at = []

            def rpc(program, _maximum=120):
                if "cfg=json.loads" in program:
                    return {"phase": "claimed", "nonce": "nonce", "pid": None}
                return {"phase": "pending", "pid_alive": False}

            def destroy():
                destroyed_at.append(clock[0])
                item.state["teardown_verified"] = True

            with (
                mock.patch.object(
                    controller.time, "time", side_effect=lambda: clock[0]
                ),
                mock.patch.object(
                    controller.time,
                    "sleep",
                    side_effect=lambda seconds: clock.__setitem__(
                        0, clock[0] + seconds
                    ),
                ),
                mock.patch.object(item, "rpc", side_effect=rpc),
                mock.patch.object(item, "collect") as collect,
                mock.patch.object(item, "destroy", side_effect=destroy),
            ):
                assert item.run() == 1
            assert destroyed_at == [1060]
            assert not item.state.get("launch_acknowledged")
            assert item.state["failure_type"] == "StartupUnconfirmed"
            collect.assert_not_called()

    def test_matching_live_supervisor_is_required_for_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))
            item.state["launcher_reported_pid"] = 123
            with mock.patch.object(
                item,
                "rpc",
                return_value={
                    "phase": "running",
                    "nonce": "nonce",
                    "pid": 123,
                    "pid_alive": False,
                },
            ):
                item.status()
            assert not item.state.get("launch_acknowledged")
            with mock.patch.object(
                item,
                "rpc",
                return_value={
                    "phase": "running",
                    "nonce": "nonce",
                    "pid": 123,
                    "pid_alive": True,
                },
            ):
                item.status()
            assert item.state["launch_acknowledged"] is True

    def test_expiry_without_final_retrieval_is_never_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))

            def destroy():
                item.state["teardown_verified"] = True

            with (
                mock.patch.object(controller.time, "time", return_value=4420),
                mock.patch.object(item, "destroy", side_effect=destroy),
                mock.patch.object(item, "start") as start,
            ):
                assert item.run() == 1
            start.assert_not_called()
            assert item.state["complete"] is False

    def test_destroy_uses_local_inheritance_and_verifies_lease_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))

            def destroy(*_arguments, **kwargs):
                assert "env" not in kwargs
                value = json.loads(item.args.lease_state.read_text())
                value.update(phase="destroyed", teardown_verified=True)
                item.args.lease_state.write_text(json.dumps(value))

            with (
                mock.patch.object(controller.time, "time", return_value=1000),
                mock.patch.object(controller.time, "sleep"),
                mock.patch.object(
                    controller.subprocess, "run", side_effect=destroy
                ) as run,
            ):
                assert item.destroy()
            assert run.call_count == 1
            assert "destroy" in run.call_args.args[0]
            assert item.state["teardown_verified"]

    def test_changed_lease_identity_is_never_destroyed(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = fixture(Path(temporary))
            value = json.loads(item.args.lease_state.read_text())
            value["instance_id"] = 99
            item.args.lease_state.write_text(json.dumps(value))
            with (
                mock.patch.object(controller.time, "time", return_value=1000),
                mock.patch.object(controller.subprocess, "run") as run,
            ):
                assert not item.destroy()
            run.assert_not_called()

    def test_collection_invokes_real_python312_manifest_verifier(self):
        if sys.version_info < (3, 12):
            self.skipTest("Safe collection verification requires Python3.12")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = fixture(root)
            item.state["remote_supervisor"] = {"phase": "terminal"}
            content = {
                "supervisor-status.json": json.dumps(
                    {
                        "phase": "terminal",
                        "nonce": "nonce",
                        "return_code": 0,
                        "owned_children_reaped": True,
                    }
                ).encode(),
                "pilot-status.json": json.dumps(
                    {
                        "phase": "complete",
                        "canonical_parity_complete": True,
                        "case_pairs": [{"gate": {"exact": True}}],
                    }
                ).encode(),
            }
            content["retrieval-manifest.json"] = json.dumps(
                {
                    name: hashlib.sha256(data).hexdigest()
                    for name, data in content.items()
                }
            ).encode()
            source = root / "source.tar.gz"
            with tarfile.open(source, "w:gz") as archive:
                for name, data in content.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            metadata = {
                "archive": item.args.remote_root
                + "/collections/pilot-results-1.tar.gz",
                "bytes": source.stat().st_size,
                "sha256": controller.digest(source),
            }
            real_run = subprocess.run

            def run(command, **kwargs):
                assert "VAST_KEY" not in kwargs["env"]
                if command[0] == "scp":
                    shutil.copy2(source, command[-1])
                    return subprocess.CompletedProcess(command, 0)
                return real_run(command, **kwargs)

            with (
                mock.patch.object(item, "rpc", return_value=metadata),
                mock.patch.object(controller.subprocess, "run", side_effect=run),
            ):
                item.collect(final=True)
            assert item.state["final_retrieval_verified"] is True
            assert item.state["pilot_succeeded"] is True
            assert item.state["canonical_case_pairs"] == 1
            assert item.state["teardown_verified"] is False


if __name__ == "__main__":
    unittest.main()
