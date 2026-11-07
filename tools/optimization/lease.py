"""Bounded Vast pilot lifecycle with an independent expiry watchdog.

Credentials are read only from environment variables and never serialized. Only
instances carrying the exact state-file label can be destroyed.
"""

from __future__ import annotations

# SDK exceptions may contain request credentials; always redact their details.
# ruff: noqa: BLE001, T201
import argparse
import contextlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


class LeasePreflightError(RuntimeError):
    """A safe, fixed diagnostic emitted before any instance is created."""


def pilot_image(digest: str | None) -> str:
    """Use the inspected immutable image when supplied; retain old CLI defaults."""
    if digest is None:
        return "vastai/base-image:cuda-12.9-mini-py312-2026-10-23"
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        message = "Image digest must be sha256 followed by 64 lowercase hex digits"
        raise LeasePreflightError(message)
    return "vastai/base-image@" + digest


def save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def client():
    credential = os.environ.get("VAST_KEY") or os.environ.get("VAST_API_KEY")
    if not credential:
        message = "Vast credential is absent"
        raise RuntimeError(message)
    return importlib.import_module("vastai.sdk").VastAI(
        api_key=credential,
        raw=True,
        quiet=True,
        explain=False,
    )


def destroy(api, path: Path) -> None:
    state = json.loads(path.read_text())
    matches = [
        row for row in api.show_instances() if row.get("label") == state["label"]
    ]
    for row in matches:
        api.destroy_instance(row["id"])
    for _ in range(6):
        remaining = [
            row
            for row in api.show_instances()
            if row.get("label") == state["label"]
            and row.get("actual_status") != "destroyed"
        ]
        if not remaining:
            state.update(
                phase="destroyed", destroyed_at=time.time(), teardown_verified=True
            )
            save(path, state)
            return
        time.sleep(10)
    message = "Instance destruction not yet verified; watchdog remains active"
    raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "watchdog", "destroy", "inspect"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--exclude-offer", type=int)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument(
        "--gpu", choices=("A100 SXM4", "A100 PCIE"), default="A100 SXM4"
    )
    parser.add_argument("--offer-id", type=int)
    parser.add_argument("--image-digest")
    args = parser.parse_args()
    if not 0.25 <= args.hours <= 4:
        parser.error("pilot duration must be between 0.25 and 4 hours")
    if args.image_digest is not None and args.action != "create":
        parser.error("--image-digest applies only to create")
    image = pilot_image(args.image_digest)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        api = client()
        if args.action == "create":
            if args.state.exists():
                message = "State already exists"
                raise LeasePreflightError(message)
            if any(
                str(row.get("label", "")).startswith("vmecpp-opt-")
                for row in api.show_instances()
            ):
                message = "An optimization lease already exists"
                raise LeasePreflightError(message)
            offers = api.search_offers(
                query={
                    "num_gpus": {"eq": 1},
                    "gpu_name": {"eq": args.gpu},
                    "verified": {"eq": True},
                    "rentable": {"eq": True},
                    "rented": {"eq": False},
                    "reliability": {"gte": 0.98},
                    "cuda_max_good": {"gte": 12.9},
                    "dph_total": {"lte": 1},
                    "cpu_cores_effective": {"gte": 8},
                    "inet_down": {"gte": 500},
                    "inet_down_cost": {"lte": 1},
                    "inet_up_cost": {"lte": 1},
                    "storage_cost": {"lte": 1},
                },
                order="dph_total",
                storage=30,
                limit=5,
            )
            offers = [offer for offer in offers if offer["dph_total"] <= 1]
            if not offers:
                message = "No qualifying offer below USD 1/hour"
                raise LeasePreflightError(message)
            offers = [offer for offer in offers if offer["id"] != args.exclude_offer]
            if args.offer_id is not None:
                offers = [offer for offer in offers if offer["id"] == args.offer_id]
            if not offers:
                message = "The requested offer is no longer available"
                raise LeasePreflightError(message)
            offer = offers[0]
            now = time.time()
            state = {
                "label": f"vmecpp-opt-{int(now)}",
                "phase": "creating",
                "created_at": now,
                "expires_at": now + args.hours * 3600,
                "offer_id": offer["id"],
                "gpu_name": offer.get("gpu_name"),
                "image": image,
                "cpu_name": offer.get("cpu_name"),
                "quoted_hourly_rate": offer["dph_total"],
                "disk_GB": 30,
                "external_volume_created": False,
                "maximum_pilot_hours": args.hours,
                "gpu_rate_ceiling_estimate_usd": offer["dph_total"] * args.hours,
            }
            save(args.state, state)
            with args.state.with_suffix(".watchdog.log").open("a") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "watchdog",
                        "--state",
                        str(args.state.resolve()),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            state["watchdog_pid"] = process.pid
            save(args.state, state)
            result = api.create_instance(
                id=offer["id"],
                image=image,
                disk=30,
                label=state["label"],
                ssh=True,
                direct=True,
                cancel_unavail=True,
            )
            instance = result.get("new_contract")
            if not instance:
                state["phase"] = "creation_unconfirmed"
                save(args.state, state)
                message = "Lease creation unconfirmed; watchdog will reconcile by label"
                raise RuntimeError(message)
            state.update(instance_id=instance, phase="created")
            save(args.state, state)
            public_key = Path.home() / ".ssh/id_ed25519.pub"
            api.attach_ssh(instance, public_key.read_text().strip())
        elif args.action == "destroy":
            destroy(api, args.state)
        elif args.action == "watchdog":
            while True:
                state = json.loads(args.state.read_text())
                if state.get("phase") == "destroyed":
                    return 0
                if time.time() >= state["expires_at"]:
                    try:
                        destroy(api, args.state)
                        return 0
                    except Exception:
                        time.sleep(30)
                else:
                    time.sleep(min(30, max(1, state["expires_at"] - time.time())))
        else:
            state = json.loads(args.state.read_text())
            rows = [
                row
                for row in api.show_instances()
                if row.get("label") == state["label"]
            ]
            fields = (
                "id",
                "label",
                "actual_status",
                "ssh_host",
                "ssh_port",
                "public_ipaddr",
                "dph_total",
                "gpu_name",
                "status_msg",
                "cur_state",
                "intended_status",
            )
            args.state.with_suffix(".connection.json").write_text(
                json.dumps(
                    [{key: row.get(key) for key in fields} for row in rows], indent=2
                )
            )
    print(f"Lease action completed: {args.action}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeasePreflightError as error:
        print(f"Lease preflight: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as error:
        print(f"Lease action failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None
