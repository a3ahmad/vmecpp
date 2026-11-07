"""Wire an explicit experimental GPU transform into the isolated source clone."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    args = parser.parse_args()
    directory = args.source.resolve() / "gpu_experiment"
    directory.mkdir(exist_ok=False)
    original = args.kernel_source.read_text()
    start = original.index("struct KernelParams {")
    params = original[start : original.index("};", start) + 2]
    start = original.index("__global__ void FourierToRealKernel(")
    end = original.index(
        "// =============================================================================",
        start,
    )
    kernel = original[start:end]
    kernel = kernel.replace(
        "atomicAdd(&lv[idx_kl], -(lmksc_n * sinmu_val + lmkcs_n * cosmu_val));",
        "lv[idx_kl] -= lmksc_n * sinmu_val + lmkcs_n * cosmu_val;",
    )
    kernel = re.sub(
        r"atomicAdd\(&([^,]+),\s*(.*?)\);", r"\1 += \2;", kernel, flags=re.DOTALL
    )
    if "atomicAdd" in kernel:
        parser.error("unconverted atomic operation")
    (directory / "generated_fourier_kernel.cuh").write_text(
        "// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH\n"
        "// SPDX-License-Identifier: MIT\n" + params + "\n" + kernel
    )
    for name in ("gpu_fourier.h", "gpu_fourier.cu"):
        shutil.copyfile(Path(__file__).with_name(name), directory / name)
    path = args.source / "src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc"
    text = path.read_text()
    text = text.replace(
        "#include <cstdio>",
        "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n"
        "#ifdef VMECPP_GPU_FOURIER_EXPERIMENT\n"
        f'#include "{directory.as_posix()}/gpu_fourier.h"\n#endif',
        1,
    )
    inputs = [
        f"physical_x.{name}"
        for name in ("rmncc", "rmnss", "zmnsc", "zmncs", "lmnsc", "lmncs")
    ]
    inputs += ["xmpq", "rp.sqrtSF"]
    inputs += [
        f"fb.{name}"
        for name in (
            "cosmu",
            "sinmu",
            "cosmum",
            "sinmum",
            "cosnv",
            "sinnv",
            "cosnvn",
            "sinnvn",
        )
    ]
    outputs = [
        f"m_geometry.{name}"
        for name in (
            "r1_e",
            "r1_o",
            "ru_e",
            "ru_o",
            "rv_e",
            "rv_o",
            "z1_e",
            "z1_o",
            "zu_e",
            "zu_o",
            "zv_e",
            "zv_o",
            "lu_e",
            "lu_o",
            "lv_e",
            "lv_o",
            "rCon",
            "zCon",
        )
    ]
    dimensions = "r.nsMinF, r.nsMaxF, r.nsMinF1, r.nsMaxF1, r.nsMaxFIncludingLcfs, s.mpol, s.ntor, s.nZeta, s.nThetaEff, s.nThetaReduced, s.nnyq2, r.nsMaxF1, 0"
    call = '\n#ifdef VMECPP_GPU_FOURIER_EXPERIMENT\n  const char* gpu_transform = std::getenv("VMECPP_GPU_FOURIER");\n  if (gpu_transform != nullptr && std::strcmp(gpu_transform, "1") == 0) {\n'
    call += (
        "    const GpuFourierInput input{{"
        + ",".join(f"{v}.data()" for v in inputs)
        + "},{"
        + ",".join(f"{v}.size()" for v in inputs)
        + "},{"
        + dimensions
        + "}};\n"
    )
    call += (
        "    GpuFourierOutput output{{"
        + ",".join(f"{v}.data()" for v in outputs)
        + "},{"
        + ",".join(f"{v}.size()" for v in outputs)
        + "}};\n"
    )
    call += "    RunGpuFourier(input, output);\n    return;\n  }\n#endif\n"
    marker = "  // can safely assume lthreed == true in here"
    if text.count(marker) != 1:
        parser.error("unexpected CPU transform layout")
    text = text.replace(marker, call + marker, 1)
    text.encode("ascii")
    path.write_text(text)
    cmake = args.source / "CMakeLists.txt"
    extra = f"""
option(VMECPP_GPU_FOURIER_EXPERIMENT "Build experimental strict FP64 transform" OFF)
if(VMECPP_GPU_FOURIER_EXPERIMENT)
  set_property(SOURCE "${{PROJECT_SOURCE_DIR}}/src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc" APPEND PROPERTY COMPILE_DEFINITIONS VMECPP_GPU_FOURIER_EXPERIMENT=1)
  set_source_files_properties("{directory.as_posix()}/gpu_fourier.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  target_sources(vmecpp_core PRIVATE "{directory.as_posix()}/gpu_fourier.o")
  target_link_libraries(vmecpp_core PRIVATE "{args.cuda.resolve().as_posix()}/lib64/libcudart.so")
endif()
"""
    cmake.write_text(cmake.read_text() + extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
