"""Run the official harness with an explicitly recorded VMEC thread override.

The harness request remains single threaded for metric comparison. This probe is an
alternative runtime configuration and must not be treated as an official request/metric
identity match merely because its numeric metrics match.
"""

from __future__ import annotations

import argparse
import importlib
import json
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True)
    args, harness_args = parser.parse_known_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    vmecpp = importlib.import_module("vmecpp")
    original_run = vmecpp.run

    def overridden_run(*positional, **keywords):
        keywords["max_threads"] = args.threads
        return original_run(*positional, **keywords)

    vmecpp.__dict__["run"] = overridden_run
    harness = Path(__file__).with_name("benchmark.py")
    sys.argv = [str(harness), *harness_args]
    status = 0
    try:
        runpy.run_path(str(harness), run_name="__main__")
    except SystemExit as error:
        status = error.code or 0
    out = Path(harness_args[harness_args.index("--out") + 1])
    (out / "runtime-override.json").write_text(
        json.dumps(
            {
                "classification": "alternative runtime configuration; parity unproven",
                "vmec_max_threads": args.threads,
                "harness_request_threads": 1,
                "warning": "Compare numeric metrics; request hash alone is insufficient",
            },
            indent=2,
        )
    )
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
