"""Research wrapper around the retained capture worker; no qualification waiver."""

# Scientific modules are imported only in the explicitly owned worker.
# ruff: noqa: PLC0415

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == args.sha
    cfg = json.loads(args.config.read_text())
    for name in cfg["preflight_files"]:
        assert (
            hashlib.sha256(Path(name).read_bytes()).hexdigest() == cfg["files"][name]
        ), name
    source = Path(cfg["capture_source"])
    sys.path[:0] = [str(source.parent), str(source.parent / "ownership"), cfg["repo"]]
    spec = importlib.util.spec_from_file_location("retained_capture", source)
    assert spec is not None
    assert spec.loader is not None
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    # This module belongs to the source-bound capture worker's ownership folder.
    from first_gate import link_parent  # pyright: ignore[reportMissingImports]

    link_parent("worker")
    root = Path(cfg["run_root"])
    root.mkdir(exist_ok=False)
    target = cfg["base"]
    target["files"] = cfg["files"]
    target["fixture"] = cfg["case"]["seed"]
    target["track"] = cfg["case"]["track"]
    target["selected_backend"] = cfg["backend"]
    target["stats"] = cfg["stats"]
    target["run_root"] = str(root)
    original_measure = capture.measured_call
    outputs = []

    def measured_and_saved(function, clocks, *positional, **keywords):
        output = original_measure(function, clocks, *positional, **keywords)
        if hasattr(output, "wout"):
            import numpy as np

            index = len(outputs)
            outputs.append(output)
            # Save immediately after the original timed call, even if metrics fail.
            output.wout.save(root / f"saved_wout_{index}.nc")
            arrays = {}
            for section in ("wout", "jxbout", "mercier", "threed1_volumetrics"):
                for name, value in getattr(output, section).model_dump().items():
                    if isinstance(value, int | float | bool | np.ndarray | np.number):
                        arrays[f"{section}/{name}"] = np.ascontiguousarray(
                            np.asarray(value)
                        )
            np.savez(root / f"saved_arrays_{index}.npz", **arrays)
        return output

    capture.measured_call = measured_and_saved  # pyright: ignore[reportAttributeAccessIssue]
    result = {"scope": "ALTERNATIVE_RESEARCH_NOT_QUALIFIED", "complete": False}
    try:
        capture.worker(target, "selected")
        result["complete"] = True
    except BaseException as failure:
        result["failure_type"] = type(failure).__name__
        traceback.print_exc()
        raise
    finally:
        result["saved_vmec_outputs"] = len(outputs)
        result["PID"] = os.getpid()
        save(root / "research-worker.json", result)


if __name__ == "__main__":
    main()
