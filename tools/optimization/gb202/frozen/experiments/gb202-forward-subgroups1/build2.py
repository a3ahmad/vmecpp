"""Offline SM120 forward recurrence subgroup experiment, default disabled."""

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "source"
    source.mkdir(exist_ok=True)
    parent = BASE / "forward-clear-elide1/source"
    for path in parent.iterdir():
        if path.is_file():
            shutil.copyfile(path, source / path.name)
    shutil.copyfile(ROOT / "subgroups_kernel.h", source / "subgroups_kernel.h")
    forward = source / "gpu_forward.cu"
    text = forward.read_text()
    point = "namespace {\nstruct ForwardPointers"
    assert text.count(point) == 1
    text = text.replace(point, "#ifndef VMECPP_FORWARD_RECURRENCE_SUBGROUPS\n#define VMECPP_FORWARD_RECURRENCE_SUBGROUPS 0\n#endif\nconstexpr int kForwardPartialGroups = kForwardGroups * (VMECPP_FORWARD_RECURRENCE_SUBGROUPS ? 2 : 1);\n\n" + point)
    old = "ForwardPartials<<<dim3((tasks + 127) / 128, kForwardGroups)"
    assert text.count(old) == 3
    text = text.replace(old, "ForwardPartials<<<dim3((tasks + 127) / 128, kForwardPartialGroups)")
    forward.write_text(text)
    header = source / "gpu_forward_groups_kernels.h"
    text = header.read_text()
    begin = text.index("__global__ void ForwardPartials")
    end = text.index("__global__ void ForwardFinish")
    text = text[:begin] + '#if VMECPP_FORWARD_RECURRENCE_SUBGROUPS\n#include "subgroups_kernel.h"\n#else\n' + text[begin:end] + "#endif\n\n" + text[end:]
    header.write_text(text)
    owner_path = BASE / "resident-solver-energy-ragged-clean1/compiler_owner.py"
    spec = importlib.util.spec_from_file_location("existing_owner", owner_path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    out = ROOT / "build2"
    out.mkdir()
    receipt = {"GPU_executions": 0, "default_OFF": True, "CPU_nice": os.getpriority(os.PRIO_PROCESS, 0), "jobs": [],
               "source_inputs": {str(p): sha(p) for p in source.iterdir() if p.is_file()},
               "current_baseline_resident": sha(BASE / "gb202-packed-sm1201/build2/resident.so")}
    def run(label, command):
        job = owner.run_job(command, label, 180, dict(os.environ), out)
        receipt["jobs"].append(job)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        assert job["direct_exit_status"] == 0 and job["owned_children_reaped"]
        assert not job["cleanup_actions"] and not job.get("error")
    for arm, enabled in (("on", "1"),):
        target = out / arm
        target.mkdir()
        run(arm + "-compile", ["/usr/local/cuda-12.9/bin/nvcc", "-std=c++20", "-O3", "--fmad=false",
            "-DVMECPP_FORWARD_SPLIT_GROUPS", "-DVMECPP_FORWARD_ELIDE_OUTPUT_CLEAR=1",
            "-DVMECPP_FORWARD_RECURRENCE_SUBGROUPS=" + enabled, "-Xcompiler=-fPIC", "-Xptxas=-v",
            "-gencode", "arch=compute_120,code=sm_120", "-I" + str(source), "-c", str(forward),
            "-o", str(target / "forward.o")])
        objects = BASE / "gb202-packed-sm1201/build2/objects"
        run(arm + "-link", ["g++", "-Wl,--no-undefined", "-shared", str(objects / "resident.o"),
            str(objects / "fourier.o"), str(target / "forward.o"), str(objects / "constraint.o"),
            "-L/usr/local/cuda-12.9/lib64", "-lcudart", "-Wl,-rpath,/usr/local/cuda-12.9/lib64",
            "-o", str(target / "resident.so")])
    receipt["outputs"] = {str(p.relative_to(out)): sha(p) for p in out.glob("*/resident.so")}
    receipt["unchanged_objects"] = {str(objects / n): sha(objects / n) for n in ("resident.o", "fourier.o", "constraint.o")}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["outputs"]))


if __name__ == "__main__":
    main()
