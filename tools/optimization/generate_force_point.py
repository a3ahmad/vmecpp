"""Extract the reference MHD expressions into an independent CPU/GPU point kernel."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

FIELDS = (
    "r12",
    "ru12",
    "zu12",
    "rs",
    "zs",
    "tau",
    "gsqrt",
    "bsupu",
    "bsupv",
    "total_pressure",
    "r1_e",
    "r1_o",
    "ru_e",
    "ru_o",
    "zu_e",
    "zu_o",
    "z1_o",
    "rv_e",
    "rv_o",
    "zv_e",
    "zv_o",
    "sqrt_sf",
    "sqrt_sh",
)
FORCES = (
    "armn_e",
    "armn_o",
    "azmn_e",
    "azmn_o",
    "brmn_e",
    "brmn_o",
    "bzmn_e",
    "bzmn_o",
    "crmn_e",
    "crmn_o",
    "czmn_e",
    "czmn_o",
)
PRODUCTS = ("P", "rup", "zup", "rsp", "zsp", "taup", "gbubu", "gbubv", "gbvbv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    relative = "src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc"
    original = subprocess.run(
        ["git", "-C", str(args.source), "show", "v0.4.11:" + relative],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    start = original.index("void IdealMhdModel::computeMHDForces()")
    end = original.index("bool IdealMhdModel::shouldUpdateRadialPreconditioner", start)
    body = original[start:end]
    pattern = r"(" + "|".join(FORCES) + r")\[idx_f\]\s*(\+=|=)\s*(.*?);"
    statements = re.findall(pattern, body, flags=re.DOTALL)
    if len(statements) != 16 or {x[0] for x in statements} != set(FORCES):
        parser.error("unexpected reference MHD expression layout")
    expressions = []
    for name, operation, original_expression in statements:
        expression = original_expression
        for product in PRODUCTS:
            member = product.lower() + "_"
            expression = expression.replace(
                f"m_ls_.{product}_i[kl]", f"inside.{member}"
            )
            expression = expression.replace(f"{product}_o[kl]", f"outside.{member}")
        expression = expression.replace("m_fc_.deltaS", "p.delta_s_")
        for field in FIELDS:
            expression = expression.replace(f"{field}[idx_g]", f"v.{field}_[idx_g]")
        if "m_ls_" in expression or "[kl]" in expression or "m_fc_" in expression:
            parser.error("unconverted reference MHD operand")
        expressions.append(f"  m_output.{name}_ {operation} {expression};\n")
    text = """// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
// Generated from v0.4.11 computeMHDForces; arithmetic order is unchanged.
#ifndef VMECPP_EXPERIMENT_MHD_POINT_H_
#define VMECPP_EXPERIMENT_MHD_POINT_H_
#ifdef __CUDACC__
#define VMECPP_MHD_INLINE __host__ __device__ __forceinline__
#else
#define VMECPP_MHD_INLINE inline
#endif
namespace vmecpp_experiment {
struct MhdForceView {
"""
    text += "".join(f"  const double* {field}_;\n" for field in FIELDS)
    text += """};
struct MhdForceParameters {
  int ns_min_f_;
  int ns_min_f1_;
  int ns_min_h_;
  int ns_max_h_;
  int angular_size_;
  double delta_s_;
};
struct MhdPointForces {
"""
    text += "".join(f"  double {force}_;\n" for force in FORCES)
    text += "};\nstruct MhdHalfProducts {\n"
    text += "".join(f"  double {product.lower()}_ = 0.0;\n" for product in PRODUCTS)
    text += """};
VMECPP_MHD_INLINE MhdHalfProducts ComputeHalfProducts(
    const MhdForceView& v, const MhdForceParameters& p, int jH, int kl) {
  const int index = (jH - p.ns_min_h_) * p.angular_size_ + kl;
  MhdHalfProducts result;
  result.p_ = v.r12_[index] * v.total_pressure_[index];
  result.rup_ = v.ru12_[index] * result.p_;
  result.zup_ = v.zu12_[index] * result.p_;
  result.rsp_ = v.rs_[index] * result.p_;
  result.zsp_ = v.zs_[index] * result.p_;
  result.taup_ = v.tau_[index] * v.total_pressure_[index];
  result.gbubu_ = v.gsqrt_[index] * v.bsupu_[index] * v.bsupu_[index];
  result.gbubv_ = v.gsqrt_[index] * v.bsupu_[index] * v.bsupv_[index];
  result.gbvbv_ = v.gsqrt_[index] * v.bsupv_[index] * v.bsupv_[index];
  return result;
}
VMECPP_MHD_INLINE void ComputeMhdForcePoint(
    const MhdForceView& v, const MhdForceParameters& p, int jF, int kl,
    MhdPointForces& m_output) {
  MhdHalfProducts inside;
  MhdHalfProducts outside;
  double sqrtSHi = 1.0;
  double sqrtSHo = 1.0;
  if (jF > p.ns_min_f_ || p.ns_min_f_ > 0) {
    const int jH = jF > p.ns_min_f_ ? jF - 1 : p.ns_min_h_;
    inside = ComputeHalfProducts(v, p, jH, kl);
    sqrtSHi = v.sqrt_sh_[jH - p.ns_min_h_];
  }
  if (jF < p.ns_max_h_) {
    outside = ComputeHalfProducts(v, p, jF, kl);
    sqrtSHo = v.sqrt_sh_[jF - p.ns_min_h_];
  }
  const double sFull = v.sqrt_sf_[jF - p.ns_min_f1_] * v.sqrt_sf_[jF - p.ns_min_f1_];
  const int idx_g = (jF - p.ns_min_f1_) * p.angular_size_ + kl;
"""
    text += "".join(expressions)
    text += """}
}  // namespace vmecpp_experiment
#undef VMECPP_MHD_INLINE
#endif  // VMECPP_EXPERIMENT_MHD_POINT_H_
"""
    text.encode("ascii")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "mhd_point.h").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
