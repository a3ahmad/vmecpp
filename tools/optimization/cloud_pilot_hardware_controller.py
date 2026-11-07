"""Hardware-timing payload controller; no provider or lease creation operations."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.optimization import cloud_pilot_controller


def main():
    if "--include-optional-tools" in sys.argv:
        message = "Hardware pilot excludes legacy optional probes"
        raise ValueError(message)
    if "--cloud-pilot-tool" not in sys.argv:
        tool = Path(__file__).with_name("cloud_pilot_hardware.py")
        if not tool.is_file():
            tool = Path(__file__).with_name("cloud_pilot.py")
        sys.argv.extend(["--cloud-pilot-tool", str(tool)])
    return cloud_pilot_controller.main()


if __name__ == "__main__":
    raise SystemExit(main())
