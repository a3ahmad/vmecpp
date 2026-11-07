"""Prepare a denser GPU launch layout without changing per-output arithmetic."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--register-outputs", action="store_true")
    parser.add_argument("--unroll-n", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--unroll-m", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    text = args.kernel.read_text()
    marker = "  const int jF = blockIdx.x + params.ns_min_f1;"
    if text.count(marker) != 1:
        parser.error("unexpected inverse-transform kernel layout")
    text = text.replace("FourierToRealKernel(", "FourierToRealKernelFlat(", 1)
    text = text.replace(
        marker,
        "  const int point_index = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  const int jF = point_index / (params.n_zeta * params.n_theta_reduced) + params.ns_min_f1;",
        1,
    )
    text = text.replace(
        "  const int k = blockIdx.y;",
        "  const int k = (point_index / params.n_theta_reduced) % params.n_zeta;",
        1,
    )
    text = text.replace(
        "  const int l = threadIdx.x;",
        "  const int l = point_index % params.n_theta_reduced;",
        1,
    )
    text = text.replace(
        "// Use atomic adds since multiple m values write to the same output location.",
        "// Each thread owns one output and accumulates modes in the reference order.",
    )
    if args.register_outputs:
        expressions = {
            "r1": "rmkcc * cosmu_val + rmkss * sinmu_val",
            "ru": "rmkcc * sinmum_val + rmkss * cosmum_val",
            "rv": "rmkcc_n * cosmu_val + rmkss_n * sinmu_val",
            "z1": "zmksc * sinmu_val + zmkcs * cosmu_val",
            "zu": "zmksc * cosmum_val + zmkcs * sinmum_val",
            "zv": "zmksc_n * sinmu_val + zmkcs_n * cosmu_val",
            "lu": "lmksc * cosmum_val + lmkcs * sinmum_val",
            "lv": "lmksc_n * sinmu_val + lmkcs_n * cosmu_val",
        }
        fields = [f"{name}_{parity}" for name in expressions for parity in ("e", "o")]
        declarations = "".join(f"  double {name}_sum = 0.0;\n" for name in fields)
        declarations += "  double r_constraint = 0.0;\n  double z_constraint = 0.0;\n"
        text = text.replace(
            "  const int ntorp1 =", declarations + "  const int ntorp1 =", 1
        )
        start = text.index("    // Select even/odd arrays based on m parity.")
        end = text.index("    // Constraint force computation.", start)
        accumulation = "    if (m_even) {\n"
        for parity in ("e", "o"):
            if parity == "o":
                accumulation += "    } else {\n"
            for name, expression in expressions.items():
                operation = "-=" if name == "lv" else "+="
                accumulation += f"      {name}_{parity}_sum {operation} {expression};\n"
        accumulation += "    }\n\n"
        text = text[:start] + accumulation + text[end:]
        text = text.replace("r_con[idx_con] +=", "r_constraint +=")
        text = text.replace("z_con[idx_con] +=", "z_constraint +=")
        end = text.rfind("\n}")
        stores = "\n  const int output_index = ((jF - params.ns_min_f1) * params.n_zeta + k) * params.n_theta_eff + l;\n"
        stores += "".join(f"  {name}[output_index] = {name}_sum;\n" for name in fields)
        stores += (
            "  if (params.ns_min_f <= jF && jF < params.ns_max_f_including_lcfs) {\n"
        )
        stores += "    const int index = ((jF - params.ns_min_f) * params.n_zeta + k) * params.n_theta_eff + l;\n"
        stores += (
            "    r_con[index] = r_constraint;\n    z_con[index] = z_constraint;\n  }\n"
        )
        text = text[:end] + stores + text[end:]
    if args.unroll_n != 1:
        text = text.replace(
            "    for (int n = 0; n < ntorp1; ++n) {",
            f"#pragma unroll {args.unroll_n}\n    for (int n = 0; n < ntorp1; ++n) {{",
            1,
        )
    if args.unroll_m != 1:
        text = text.replace(
            "  for (int m = 0; m < params.mpol; ++m) {",
            f"#pragma unroll {args.unroll_m}\n  for (int m = 0; m < params.mpol; ++m) {{",
            1,
        )
    text.encode("ascii")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "generated_fourier_kernel.cuh").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
