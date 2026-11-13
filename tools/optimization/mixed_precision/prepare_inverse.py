"""Generate isolated FP32-local inverse kernels with an unchanged double ABI."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parent / "gb202" / "frozen"


def main():
    original = (FROZEN / "2d399832145f-generated_fourier_kernel.cuh").read_text()
    pieces = [
        "// SPDX-License-Identifier: MIT\n"
        "// Research-only device-time dispatch; one equilibrium per process.\n"
        "namespace vmecpp_mixed_inverse {\n"
        "__device__ int precision_fp32 = 0;\n"
        "__device__ unsigned long long precision_counts[2] = {};\n"
        "struct FloatReadView {\n"
        "  const double* data_;\n"
        "  __device__ float operator[](int index) const {\n"
        "    return static_cast<float>(data_[index]);\n"
        "  }\n};\n"
    ]
    proofs = []
    names = ("ToroidalPartialsKernel", "FourierToRealKernelSeparable")
    for name in names:
        start = original.index("__global__ void " + name + "(")
        brace = original.index("{", start)
        depth = 1
        end = brace + 1
        while depth:
            depth += (original[end] == "{") - (original[end] == "}")
            end += 1
        signature = original[start:brace].strip()
        body = original[brace + 1 : end - 1]
        inputs = re.findall(r"const double\* __restrict__ (\w+)", signature)
        arguments = re.findall(r"(?:__restrict__ |KernelParams )(\w+)", signature)
        for bits in (64, 32):
            helper = signature.replace("__global__", "__device__ __forceinline__")
            helper = helper.replace(name, name + "Fp" + str(bits))
            transformed = body
            if bits == 32:
                transformed = re.sub(r"\bdouble\b", "float", body)
                transformed = re.sub(
                    r"(?<![\w.])(\d+\.\d+)(?![\w.])", r"\1f", transformed
                )
                views = []
                for item in inputs:
                    helper = re.sub(r"\b" + item + r"\b", item + "_fp64", helper)
                    views.append(
                        "  const FloatReadView " + item + "{" + item + "_fp64};"
                    )
                transformed = "\n" + "\n".join(views) + transformed
            pieces.append(helper + " {" + transformed + "\n}\n")
        wrapper = signature.replace(name, name + "Dispatch")
        call = ", ".join(arguments)
        pieces.append(
            wrapper + " {\n"
            "  const int mode = precision_fp32 != 0;\n"
            "  if (blockIdx.x == 0 && threadIdx.x == 0)\n"
            "    atomicAdd(&precision_counts[mode], 1ULL);\n"
            "  if (mode) " + name + "Fp32(" + call + ");\n"
            "  else " + name + "Fp64(" + call + ");\n}\n"
        )
        proofs.append(
            {
                "kernel": name,
                "fp64_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "fp64_body_literal_copy": True,
                "float_input_views": inputs,
            }
        )
    pieces.append("}  // namespace vmecpp_mixed_inverse\n")
    (ROOT / "inverse_mixed_kernels.cuh").write_text("\n".join(pieces), encoding="ascii")
    host = (FROZEN / "d5f9016c843e-gpu_fourier.cu").read_text()
    host = host.replace(
        '#include "gpu_fourier.h"',
        '#include "gpu_fourier.h"\n#include "inverse_mixed_kernels.cuh"',
    )
    host = host.replace(
        "ToroidalPartialsKernel<<<",
        "vmecpp_mixed_inverse::ToroidalPartialsKernelDispatch<<<",
    )
    host = host.replace(
        "FourierToRealKernel<<<",
        "vmecpp_mixed_inverse::FourierToRealKernelSeparableDispatch<<<",
    )
    host += """
extern "C" cudaError_t SetGpuInverseTransformPrecision(int bits, void* stream) {
  if (bits != 32 && bits != 64) return cudaErrorInvalidValue;
  if (stream != nullptr) return cudaErrorInvalidValue;
  static int current_bits = 64;
  if (bits == current_bits) return cudaSuccess;
  const int flag = bits == 32;
  const auto status = cudaMemcpyToSymbol(vmecpp_mixed_inverse::precision_fp32,
                                       &flag, sizeof(flag));
  if (status == cudaSuccess) current_bits = bits;
  return status;
}
extern "C" cudaError_t GetGpuInverseTransformPrecisionCounts(
    unsigned long long* m_counts) {
  if (m_counts == nullptr) return cudaErrorInvalidValue;
  return cudaMemcpyFromSymbol(m_counts, vmecpp_mixed_inverse::precision_counts,
                              2 * sizeof(unsigned long long));
}
"""
    (ROOT / "gpu_fourier_mixed.cu").write_text(host, encoding="ascii")
    (ROOT / "inverse-source-proof.json").write_text(json.dumps(proofs, indent=2) + "\n")


if __name__ == "__main__":
    main()
