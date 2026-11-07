"""Current-payload controller interface guard; creates no lease or provider API."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.optimization import cloud_pilot_controller


def main():
    if "--include-optional-tools" in sys.argv:
        message = "Current canonical pilot excludes legacy optional solver8 probes; leave --include-optional-tools OFF"
        raise ValueError(message)
    if "--cloud-pilot-tool" not in sys.argv:
        current = Path(__file__).with_name("cloud_pilot_current.py")
        if not current.is_file():
            current = Path(__file__).with_name("cloud_pilot.py")
        sys.argv.extend(["--cloud-pilot-tool", str(current)])
    return cloud_pilot_controller.main()


if __name__ == "__main__":
    raise SystemExit(main())
