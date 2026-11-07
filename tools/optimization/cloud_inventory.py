"""Read Vast inventory using an environment credential; never create a lease."""

from __future__ import annotations

# SDK exceptions may contain request credentials; always redact their details.
# ruff: noqa: BLE001, T201
import contextlib
import importlib
import io
import json
import os
from pathlib import Path


def main() -> int:
    credential = os.environ.get("VAST_KEY") or os.environ.get("VAST_API_KEY")
    if not credential:
        message = "Vast credential environment variable is absent"
        raise SystemExit(message)
    sdk = importlib.import_module("vastai.sdk")
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            client = sdk.VastAI(api_key=credential, raw=True, quiet=True)
            user = client.show_user()
            instances = client.show_instances()
            offers = client.search_offers(
                query={
                    "num_gpus": {"eq": 1},
                    "gpu_name": {
                        "in": [
                            "A100 SXM4",
                            "A100 PCIE",
                            "H100 SXM",
                            "H100 PCIE",
                            "H200",
                        ]
                    },
                    "dph_total": {"lte": 3},
                    "reliability": {"gte": 0.98},
                    "verified": {"eq": True},
                    "rentable": {"eq": True},
                    "rented": {"eq": False},
                },
                order="dph_total",
                limit=20,
                storage=30,
            )
    except Exception as error:
        # Do not stringify SDK exceptions: request details may include credentials.
        status = getattr(getattr(error, "response", None), "status_code", None)
        message = f"Vast inventory failed: {type(error).__name__}, HTTP {status}"
        raise SystemExit(message) from None
    safe_fields = (
        "id",
        "gpu_name",
        "num_gpus",
        "gpu_ram",
        "dph_total",
        "storage_cost",
        "cpu_name",
        "cpu_cores_effective",
        "cpu_ram",
        "reliability",
        "inet_down",
        "inet_up",
        "inet_down_cost",
        "inet_up_cost",
        "cuda_max_good",
    )
    report = {
        "measurement": "measured inventory; no lease created",
        "authenticated": isinstance(user, dict) and not user.get("error", False),
        "existing_instances": [
            {key: row.get(key) for key in ("id", "label", "actual_status", "dph_total")}
            for row in instances
        ],
        "offers": [{key: row.get(key) for key in safe_fields} for row in offers],
    }
    out = Path("build/optimization/vast-inventory.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"Authenticated: {report['authenticated']}; qualifying offers: {len(offers)}")
    return 0 if report["authenticated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
