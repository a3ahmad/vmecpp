"""Reuse the retained full18-output native inverse control; PLAN by default.

Site-bound assets are hash-pinned. Root owns GPU admission and external timeout.
This script never claims that isolated GEMM timings equal full transform time.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--driver-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    pins = {
        "build/optimization/inverse-device-timing1/build1/inverse-device-timing": "cee696780bddc80c8a0c6292e124cf01acb609e0f6cd316456fe129e0928ee70",
        "build/optimization/gb202-packed-sm1201/build2/resident.so": "60d24820102c953dedc85a8487ec3023fdf7de921ff7e82ff25294718309e078",
        "build/optimization/public-stage-capture-run1/actual7/generated-public/ns99/inverse.bin": "febee8b11a4de04589321e40715bafa01cf5e5a748cc9ba6eca4ec1f68132af6",
        "build/optimization/a100-fast-component-batch1/package2/payload/lib/ld-linux-x86-64.so.2": "c20a2dc8917c755f02b94049356320fe1f62ac7d9f8994731f807d9df39302da",
    }
    for name, expected in pins.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    binary, resident, capture, loader = (root / name for name in pins)
    driver = args.driver_dir.resolve(strict=True)
    assert driver.is_dir()
    report = args.report.resolve()
    assert report.parent.is_dir()
    assert not report.exists(), "Fresh report required"
    libraries = f"{driver}:/usr/lib/wsl/lib:{loader.parent}:/usr/local/cuda-12.9/lib64"
    command = [str(loader), "--library-path", libraries, str(binary), str(capture),
               str(report), "100", "8", str(resident), str(resident)]
    if not args.execute:
        print(json.dumps({"PLAN": True, "command": command, "GPU_calls": 0,
                          "external_owned_timeout_required_seconds": 60,
                          "scope": "same native60d module in both arms; outputclear+ToroidalPartials+inverse, all18arrays exact"}, indent=2))  # noqa: T201
        return
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                   "HOME": os.environ.get("HOME", "/tmp"), "CUDA_VISIBLE_DEVICES": "0",
                   "LD_LIBRARY_PATH": libraries}
    os.execve(str(loader), command, environment)


if __name__ == "__main__":
    main()
