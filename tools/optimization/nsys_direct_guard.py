"""Validate frozen direct Nsight child state without changing its environment."""

from pathlib import Path

from tools.optimization.nsys_injection_guard import (
    ORDERED_PRELOAD_LIBRARIES,
    ROUTING_LIBRARY,
    digest,
)

REFERENCE_SHA = "144e4f204cbb0569edd25710cf8ef4c0e618b0e7a60e33cf3a556066e8a06da8"


def validate_direct_environment(root, expected, actual, runtime_files):
    """Read only: preserve the tool's private keys, loader paths and ordering."""
    directory = (root / "work/nsys/target-linux-x64").resolve()
    paths = {}
    for name, sha in (ROUTING_LIBRARY, *ORDERED_PRELOAD_LIBRARIES):
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory)
            or not path.is_file()
            or digest(path) != sha
        ):
            message = "Exact frozen Nsight library hash required: " + name
            raise ValueError(message)
        paths[name] = path
    shim = (root / "work/lib/solver-shim.so").resolve()
    if not shim.is_file() or digest(shim) != runtime_files.get(
        "work/lib/solver-shim.so"
    ):
        message = "Exact frozen canonical application shim required"
        raise ValueError(message)
    wanted = [shim, *(paths[name] for name, _ in ORDERED_PRELOAD_LIBRARIES)]
    raw = actual.get("LD_PRELOAD", "")
    entries = raw.replace(":", " ").split()
    if (
        len(entries) != 3
        or any(not Path(value).is_absolute() for value in entries)
        or [Path(value).resolve() for value in entries] != wanted
    ):
        message = "Only exact APPFIRST canonical shim and two frozen proxies allowed"
        raise ValueError(message)
    for key in ("CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH"):
        value = actual.get(key, "")
        if (
            not value
            or not Path(value).is_absolute()
            or Path(value).resolve() != paths[ROUTING_LIBRARY[0]]
        ):
            message = "Exact frozen Nsight injection routing required"
            raise ValueError(message)
    flags = {key: value for key, value in expected.items() if key.startswith("VMECPP_")}
    observed = {
        key: value for key, value in actual.items() if key.startswith("VMECPP_")
    }
    if len(flags) != 9 or observed != flags:
        message = "Actual direct child must retain exactly the nine frozen solver flags"
        raise ValueError(message)
    for key in (
        "OPENBLAS_CORETYPE",
        "NPY_DISABLE_CPU_FEATURES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "CGA_OPTIMIZATION_WRAPPER_SHA256",
        "CGA_OPTIMIZATION_CONFIGURATION_SHA256",
    ):
        if actual.get(key) != expected.get(key):
            message = "Actual direct child environment differs: " + key
            raise ValueError(message)
    provider = Path(flags["VMECPP_FORWARD_REFERENCE_SO"])
    if not provider.is_file() or digest(provider) != REFERENCE_SHA:
        message = "Actual reference provider is not frozen144e"
        raise ValueError(message)
    if not actual.get("NSYS_PROFILING_SESSION_ID"):
        message = "Actual Nsight session provenance missing"
        raise ValueError(message)
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5].removesuffix(" (deleted)")).resolve()
            if path in wanted:
                mapped.add(path)
    if set(wanted) != mapped:
        message = "Actual direct child preloads are not mapped"
        raise ValueError(message)
    return {
        "exact": True,
        "environment_rewritten": False,
        "validated_raw_LD_PRELOAD": raw,
        "actual_APPFIRST_paths": [str(path) for path in wanted],
        "actual_mapped_preloads": sorted(str(path) for path in mapped),
        "frozen_library_sha256": {name: digest(path) for name, path in paths.items()},
        "application_shim_sha256": digest(shim),
        "reference_provider_sha256": digest(provider),
        "nine_flags_exact": True,
        "actual_NSYS_key_names_only": sorted(
            key for key in actual if key.startswith("NSYS_")
        ),
    }
