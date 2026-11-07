"""Isolated native four-solve-worker baseline versus queued-adapter comparison."""

# Explicit terminal receipts and interruption cleanup are required by this runner.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from tools.optimization import batch_throughput as batch
from tools.optimization import native_batch_campaign as base
from tools.optimization.batch_solve_queue import QueuedCaptureDispatcher, pinned_pool

SOURCE = Path("/bulk/vmecpp-optimization-20261103/batch18-qualification")
ROOT = SOURCE.parent / "queue18-comparison72"
ROLES = {
    1: {
        "role": "baseline",
        "adapter_workers": 4,
        "native_workers": 4,
        "adapter_cpus": [0, 1, 2, 3],
        "native_cpus": [0, 1, 2, 3],
    },
    4: {
        "role": "queued",
        "adapter_workers": 6,
        "native_workers": 4,
        "adapter_cpus": [4, 5, 6, 7, 8, 9],
        "native_cpus": [0, 1, 2, 3],
    },
}


def service(root, label):
    role = ROLES[label]
    config = json.loads((root / "run/launch.json").read_text())
    status = {
        "phase": "running",
        "pid": os.getpid(),
        "birth": base.birth(os.getpid()),
        "role_label": label,
        "resource_role": role,
        "nonce": config["nonce"],
        "actual_process_affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    base.save(root / "run" / f"service-{label}.json", status)
    if status["actual_process_affinity"] != list(range(10)) or status["nice"] != 0:
        message = "Persistent role did not receive its fixed mask and priority"
        raise RuntimeError(message)
    original_rounds = batch.run_rounds
    if label == 4:

        def factory(original, **options):
            return QueuedCaptureDispatcher(
                original, native_workers=4, native_cpus=role["native_cpus"], **options
            )

        cast(Any, batch).CaptureDispatcher = factory

    def coordinated(
        groups, dispatcher, evaluate, finalize, actual_workers, *, verify_artifacts
    ):
        if actual_workers != role["adapter_workers"]:
            message = "Actual role worker count disagrees with frozen resource plan"
            raise RuntimeError(message)

        def run(groups, dispatcher, evaluate, finalize, internal_label, **options):
            if internal_label != label:
                message = "Internal role label changed"
                raise RuntimeError(message)
            options["executor_factory"] = pinned_pool(role["adapter_cpus"])
            return original_rounds(
                groups,
                dispatcher,
                evaluate,
                finalize,
                role["adapter_workers"],
                **options,
            )

        try:
            return base.controlled_rounds(
                root,
                config,
                label,
                run,
                groups,
                dispatcher,
                evaluate,
                finalize,
                verify_artifacts=verify_artifacts,
            )
        finally:
            if label == 4:
                dispatcher.close()
                base.save(
                    root / f"queue-native-activity-{label}.json",
                    {
                        "resource_role": role,
                        "all_native_tasks_terminal": all(
                            f.done() for f in dispatcher.futures
                        ),
                        "pool_closed": dispatcher.closed,
                        "records": dispatcher.queue_records,
                    },
                )

    batch.run_rounds = coordinated
    sys.argv = [
        "native-queue-role",
        "--execute",
        "--backend",
        "gpu",
        "--workers",
        str(role["adapter_workers"]),
        "--warmup-rounds",
        "1",
        "--rounds",
        "5",
        "--replicates",
        "1",
        "--manifest",
        str(SOURCE / "fixtures/remote-manifest.json"),
        "--references",
        str(SOURCE / "references"),
        "--stella",
        str(SOURCE / "stella"),
        "--out",
        str(root / "evals" / f"workers{label}"),
    ]
    code = 1
    try:
        code = batch.main()
        return code
    except BaseException as error:
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        status.update(phase="terminal", exit_status=code, finished_at=time.time())
        base.save(root / "run" / f"service-{label}.json", status)


def report(root):
    original = ORIGINAL_REPORT(root, output=root / "internal-role-numeric-summary.json")
    result = dict(original)
    result["scope"] = (
        "Measured six-case batch throughput: baseline four inline solve workers versus four native solve workers plus six adapters; CPU resource allocation differs"
    )
    result["resource_roles"] = ROLES
    result["internal_directory_role_labels_are_not_worker_counts"] = True
    result["cpu_resources_differ"] = True
    queue = json.loads((root / "queue-native-activity-4.json").read_text())
    native_tids = {row["native_tid"] for row in queue["records"]}
    adapter_tids = {row["adapter_tid"] for row in queue["records"]}
    resource_exact = (
        queue["all_native_tasks_terminal"]
        and queue["pool_closed"]
        and len(queue["records"]) == 36
        and len(native_tids) == 4
        and len(adapter_tids) == 6
        and not native_tids & adapter_tids
    )
    resource_exact = resource_exact and all(
        len(row["before_affinity"]) == 1
        and row["before_affinity"] == row["after_affinity"]
        and row["before_affinity"][0] in ROLES[4]["native_cpus"]
        and len(row["adapter_affinity"]) == 1
        and row["adapter_affinity"][0] in ROLES[4]["adapter_cpus"]
        and all(
            after > before
            for after, before in zip(
                row["activity_after"], row["activity_before"], strict=True
            )
        )
        for row in queue["records"]
    )
    for label, role in ROLES.items():
        summary = json.loads(
            (root / "evals" / f"workers{label}" / "batch-summary.json").read_text()
        )
        resource_exact = (
            resource_exact and summary["workers_requested"] == role["adapter_workers"]
        )
        resource_exact = resource_exact and all(
            row["worker_affinity"]["start_affinity"]
            == row["worker_affinity"]["end_affinity"]
            and row["worker_affinity"]["start_affinity"][0] in role["adapter_cpus"]
            for row in summary["records"]
        )
    result["actual_resource_role_gate_exact"] = resource_exact
    result["batch_table_qualified"] = bool(
        result["batch_table_qualified"] and resource_exact
    )
    original["batch_table_qualified"] = result["batch_table_qualified"]
    pairs = []
    for row in original["measured_pairs"]:
        pairs.append(
            {
                "pair": row["pair"],
                "baseline_seconds": row["workers1_seconds"],
                "queued_seconds": row["workers4_seconds"],
                "baseline_evaluations_per_second": row[
                    "workers1_evaluations_per_second"
                ],
                "queued_evaluations_per_second": row["workers4_evaluations_per_second"],
                "queued_over_baseline_throughput_ratio": row[
                    "batch_throughput_ratio_4_over_1"
                ],
            }
        )
    result["measured_pairs"] = pairs
    for row in result["per_request_latencies_separate"]:
        label = row.pop("workers")
        row.update(ROLES[label])
    original["role_label_legend"] = ROLES
    original["directory_labels_not_worker_counts"] = True
    base.save(root / "internal-role-numeric-summary.json", original)
    base.save(root / "batch-qualification-summary.json", result)
    return result


ORIGINAL_REPORT = base.report


def stage():
    if ROOT.exists() and (ROOT / "stage-ready.json").exists():
        message = "Queue comparison already staged; never overwrite frozen inputs"
        raise RuntimeError(message)
    old = json.loads((SOURCE / "stage-ready.json").read_text())
    previous = json.loads((SOURCE / "run/launch.json").read_text())
    base.check_files(previous["files"])
    environment = dict(old["service_environment"])
    environment["PYTHONPATH"] = str(ROOT) + ":" + str(SOURCE / "python")
    files = dict(previous["files"])
    files.update(
        {str(p): base.digest(p) for p in (ROOT / "tools/optimization").glob("*.py")}
    )
    topology = old["affinity"]
    topology["service_cpus"] = list(range(10))
    plan = {
        "scope": "PLAN: old solver18 runtime unchanged; extra adapter CPU roles explicitly part of comparison",
        "crosshost_gate_evaluations": 12,
        "paired_campaign_evaluations": 72,
        "warmup_pairs": 1,
        "measured_pairs": 5,
        "resource_roles": ROLES,
        "alternating_role_labels": [base.paired_order(p) for p in range(6)],
        "original_root_read_only": str(SOURCE),
        "new_root": str(ROOT),
        "internal_role_labels": {
            "1": "baseline four inline workers",
            "4": "queued four native plus six adapter workers",
        },
        "estimate_not_measurement_minutes": [14, 18],
        "deadline_seconds": 2100,
    }
    base.save(ROOT / "campaign-plan.json", plan)
    files[str(ROOT / "campaign-plan.json")] = base.digest(ROOT / "campaign-plan.json")
    base.save(
        ROOT / "stage-ready.json",
        {
            "files": files,
            "python": old["python"],
            "service_environment": environment,
            "affinity": topology,
            "resource_roles": ROLES,
            "all_original_files_exact": True,
        },
    )
    return plan


def campaign():
    base.REMOTE_ROOT = ROOT
    cast(Any, base).report = report
    original_launch = base.service_launch

    def launch(environment, cpus, python, arguments):
        rewritten = [
            arg.replace(
                "tools.optimization.native_batch_campaign",
                "tools.optimization.native_queue_campaign",
            )
            for arg in arguments
        ]
        return original_launch(environment, cpus, python, rewritten)

    base.service_launch = launch
    return base.campaign(ROOT, 2100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--service", type=int, choices=(1, 4))
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.root.resolve() != ROOT:
        parser.error("Only the new isolated queue-comparison root is permitted")
    if arguments.stage:
        print(json.dumps(stage(), indent=2))
    elif arguments.service:
        raise SystemExit(service(ROOT, arguments.service))
    elif arguments.execute:
        raise SystemExit(campaign())
    else:
        print(
            json.dumps(
                {
                    "plan_only": True,
                    "resource_roles": ROLES,
                    "paired_evaluations": 72,
                    "crosshost_gate_evaluations": 12,
                }
            )
        )
