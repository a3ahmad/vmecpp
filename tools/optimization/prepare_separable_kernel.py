"""Reuse toroidal sums across theta points while retaining reduction order."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-components", action="store_true")
    args = parser.parse_args()
    text = args.kernel.read_text()
    start_marker = "    // Accumulate from toroidal modes n."
    end_marker = "    // Compute output index."
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        parser.error("unexpected inverse-transform layout")
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    sums = text[start:end]
    names = [
        "rmkcc",
        "rmkcc_n",
        "rmkss",
        "rmkss_n",
        "zmksc",
        "zmksc_n",
        "zmkcs",
        "zmkcs_n",
        "lmksc",
        "lmksc_n",
        "lmkcs",
        "lmkcs_n",
    ]
    task_count = "(params.ns_max_f1 - params.ns_min_f1) * params.mpol * params.n_zeta"
    partials = (
        "    const int task_count = " + task_count + ";\n"
        "    const int task = ((jF - params.ns_min_f1) * params.mpol + m) * params.n_zeta + k;\n"
    )
    partials += "".join(
        f"    const double {name} = partials[{i} * task_count + task];\n"
        for i, name in enumerate(names)
    )
    stage = """__global__ void ToroidalPartialsKernel(
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ cosnv, const double* __restrict__ sinnv,
    const double* __restrict__ cosnvn, const double* __restrict__ sinnvn,
    double* __restrict__ m_partials, KernelParams params) {
  const int task = blockIdx.x * blockDim.x + threadIdx.x;
"""
    stage += "  const int task_count = " + task_count + ";\n"
    stage += """  if (task >= task_count) { return; }
  const int k = task % params.n_zeta;
  const int m = (task / params.n_zeta) % params.mpol;
  const int jF = task / (params.n_zeta * params.mpol) + params.ns_min_f1;
  if (jF == 0 && m > 1) { return; }
  const int ntorp1 = params.ntor + 1;
  const int nnyq2p1 = params.nnyq2 + 1;
"""
    stage += sums
    stage += "".join(
        f"  m_partials[{i} * task_count + task] = {name};\n"
        for i, name in enumerate(names)
    )
    stage += "}\n"
    if args.split_components:
        signature = stage[: stage.index("  const int task =")]
        stage = signature + "  const int task_count = " + task_count + ";\n"
        stage += """  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 12 * task_count) { return; }
  const int component = index / task_count;
  const int task = index % task_count;
  const int k = task % params.n_zeta;
  const int m = (task / params.n_zeta) % params.mpol;
  const int jF = task / (params.n_zeta * params.mpol) + params.ns_min_f1;
  if (jF == 0 && m > 1) { return; }
  const double* coefficient = rmncc;
  switch (component / 2) {
    case 1: coefficient = rmnss; break;
    case 2: coefficient = zmnsc; break;
    case 3: coefficient = zmncs; break;
    case 4: coefficient = lmnsc; break;
    case 5: coefficient = lmncs; break;
  }
  const double* basis = cosnv;
  switch (component % 4) {
    case 1: basis = sinnvn; break;
    case 2: basis = sinnv; break;
    case 3: basis = cosnvn; break;
  }
  double sum = 0.0;
  for (int n = 0; n <= params.ntor; ++n) {
    const int idx_mn = ((jF - params.ns_min_f1) * params.mpol + m) * (params.ntor + 1) + n;
    const int idx_kn = k * (params.nnyq2 + 1) + n;
    sum += coefficient[idx_mn] * basis[idx_kn];
  }
  m_partials[index] = sum;
}
"""
    text = text[:start] + partials + text[end:]
    signature = "__global__ void FourierToRealKernelFlat("
    if text.count(signature) != 1:
        parser.error("expected the dense launch kernel")
    text = text.replace(
        signature,
        stage + "__global__ void FourierToRealKernelSeparable(\n"
        "    const double* __restrict__ partials,",
        1,
    )
    text.encode("ascii")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "generated_fourier_kernel.cuh").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
