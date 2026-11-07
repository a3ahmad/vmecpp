"""Exact frozen Nsight routing/preload guard; no arbitrary inherited preload."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROUTING_LIBRARY = (
    "libToolsInjection64.so",
    "d16d1cbf42e9b9ec8e5750ad1113f513601e20b9a561e54a90c5a8c546518ca3",
)
ORDERED_PRELOAD_LIBRARIES = (
    (
        "libToolsInjectionProxy64.so",
        "a5cdecae0b5195503736b7d4772427bcbb96e50b682488a960fe0fdf5a1eb5ea",
    ),
    (
        "libLinuxKeyboardInterceptorProxy.so",
        "1a5c1b1c7be874be5bd02a6b30834a2062f169e6e28371c2ebe695672a0e44fa",
    ),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_profile_environment(root, base, inherited):
    """Preserve measured proxy order and append only the canonical app shim."""
    directory = (root / "work/nsys/target-linux-x64").resolve()
    expected = {}
    for name, sha in (ROUTING_LIBRARY, *ORDERED_PRELOAD_LIBRARIES):
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory)
            or not path.is_file()
            or digest(path) != sha
        ):
            message = "Frozen Nsight library missing or mismatched: " + name
            raise ValueError(message)
        expected[name] = path
    result = dict(base)
    for key in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH"):
        value = inherited.get(key, "")
        if (
            not value
            or not Path(value).is_absolute()
            or Path(value).resolve() != expected[ROUTING_LIBRARY[0]]
        ):
            message = "Unexpected or absent exact Nsight routing path"
            raise ValueError(message)
        result[key] = str(expected[ROUTING_LIBRARY[0]])
    entries = inherited.get("LD_PRELOAD", "").replace(":", " ").split()
    wanted = [expected[name] for name, _ in ORDERED_PRELOAD_LIBRARIES]
    if (
        len(entries) != len(wanted)
        or any(not Path(value).is_absolute() for value in entries)
        or [Path(value).resolve() for value in entries] != wanted
    ):
        message = "Unexpected frozen Nsight preload set or order"
        raise ValueError(message)
    shim = (root / "work/lib/solver-shim.so").resolve()
    if not shim.is_file() or base.get("LD_PRELOAD") != str(shim):
        message = "Only the canonical application shim may follow profiler proxies"
        raise ValueError(message)
    result["LD_PRELOAD"] = ":".join([*(str(path) for path in wanted), str(shim)])
    result.update(
        {key: value for key, value in inherited.items() if key.startswith("NSYS_")}
    )
    return result
