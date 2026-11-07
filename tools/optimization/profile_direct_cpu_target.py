"""Pure stdlib direct-child route/provider/FENV proof; no application CUDA calls."""

import ctypes
import importlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    root, destination = (Path(value).resolve() for value in sys.argv[1:])
    sys.path.insert(0, str(root))
    pilot = importlib.import_module("tools.optimization.cloud_pilot")
    proof = pilot.direct_profile_worker_guard(root, "candidate")
    guard = ctypes.CDLL(str(root / "work/bin/pilot-cpu-guard.so"))
    pushed = guard.VmecPilotRangePush()
    time.sleep(0.01)
    popped = guard.VmecPilotRangePop()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "guard": proof,
                "push_return": pushed,
                "pop_return": popped,
                "application_CUDA_calls": 0,
                "equilibrium_calls": 0,
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
