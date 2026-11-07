"""Extract unchanged Jacobian and metric expressions for the resident GPU path."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

SURFACE_FIELDS = (
    ("r1e", "r1_e"),
    ("r1o", "r1_o"),
    ("z1e", "z1_e"),
    ("z1o", "z1_o"),
    ("rue", "ru_e"),
    ("ruo", "ru_o"),
    ("zue", "zu_e"),
    ("zuo", "zu_o"),
    ("rve", "rv_e"),
    ("rvo", "rv_o"),
    ("zve", "zv_e"),
    ("zvo", "zv_o"),
)
HALF_FIELDS = ("r12", "ru12", "zu12", "rs", "zs", "tau", "gsqrt", "guu", "guv", "gvv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = subprocess.run(
        [
            "git",
            "-C",
            str(args.source),
            "show",
            "v0.4.11:src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    jacobian = original.split("void IdealMhdModel::computeJacobian()", 1)[1]
    jacobian = jacobian.split("void IdealMhdModel::computeMetricElements()", 1)[0]
    metric = original.split("void IdealMhdModel::computeMetricElements()", 1)[1]
    metric = metric.split("void IdealMhdModel::updateDifferentialVolume()", 1)[0]
    jacobian = jacobian[jacobian.index("// R on half-grid") :]
    jacobian = jacobian[: jacobian.index("// hand over to next iteration")]
    jacobian = re.sub(
        r"\s*if \(tau_val [<>] (?:minTau|maxTau).*?\n\s*\}",
        "",
        jacobian,
        flags=re.DOTALL,
    )
    if "minTau" in jacobian or "maxTau" in jacobian:
        parser.error("unconverted Jacobian reduction")
    pattern = r"(gsqrt|guu|guv|gvv)\[iHalf\]\s*(\+=|=)\s*(.*?);"
    statements = re.findall(pattern, metric, flags=re.DOTALL)
    if len(statements) != 5:
        parser.error("unexpected metric expression count")
    expressions = (
        jacobian
        + "\n"
        + "\n".join(
            f"{name}[iHalf] {operation} {expression};"
            for name, operation, expression in statements
        )
    )
    for short, _ in SURFACE_FIELDS:
        expressions = expressions.replace(f"m_ls_.{short}_i[kl]", f"inside.{short}_")
    for field in HALF_FIELDS:
        expressions = expressions.replace(f"{field}[iHalf]", f"m_output.{field}_")
    expressions = expressions.replace("m_fc_.deltaS", "p.delta_s_")
    expressions = expressions.replace("dSHalfDsInterp", "p.d_s_half_ds_interp_")
    if "m_ls_" in expressions or "[iHalf]" in expressions or "m_fc_" in expressions:
        parser.error("unconverted point expression")
    text = """// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// SPDX-License-Identifier: MIT
// Generated from v0.4.11. Preserve every parenthesis and arithmetic order.
#ifndef VMECPP_EXPERIMENT_GEOMETRY_POINT_H_
#define VMECPP_EXPERIMENT_GEOMETRY_POINT_H_
#ifdef __CUDACC__
#define VMECPP_GEOMETRY_POINT_INLINE __host__ __device__ __forceinline__
#else
#define VMECPP_GEOMETRY_POINT_INLINE inline
#endif
namespace vmecpp_experiment {
struct GeometryPointView {
"""
    text += "".join(f"  const double* {field}_;\n" for _, field in SURFACE_FIELDS)
    text += """  const double* sqrt_sf_;
  const double* sqrt_sh_;
};
struct GeometryPointParameters {
  int angular_size_;
  int ns_min_f1_;
  int ns_min_h_;
  double delta_s_;
  double d_s_half_ds_interp_;
};
struct GeometryPoint {
"""
    text += "".join(f"  double {field}_;\n" for field in HALF_FIELDS)
    text += """};
VMECPP_GEOMETRY_POINT_INLINE void ComputeGeometryPoint(
    const GeometryPointView& v, const GeometryPointParameters& p,
    int jH, int kl, GeometryPoint& m_output) {
  const int inner_index = (jH - p.ns_min_f1_) * p.angular_size_ + kl;
  const int outer_index = (jH + 1 - p.ns_min_f1_) * p.angular_size_ + kl;
  struct Surface {
"""
    text += "".join(f"    double {short}_;\n" for short, _ in SURFACE_FIELDS)
    text += "  };\n  const Surface inside{\n"
    text += ",\n".join(f"    v.{field}_[inner_index]" for _, field in SURFACE_FIELDS)
    text += "\n  };\n"
    text += "".join(
        f"  const double {short}_o = v.{field}_[outer_index];\n"
        for short, field in SURFACE_FIELDS
    )
    text += """  const double sqrtSH = v.sqrt_sh_[jH - p.ns_min_h_];
  const double sF_i = v.sqrt_sf_[jH - p.ns_min_f1_] *
                      v.sqrt_sf_[jH - p.ns_min_f1_];
  const double sF_o = v.sqrt_sf_[jH + 1 - p.ns_min_f1_] *
                      v.sqrt_sf_[jH + 1 - p.ns_min_f1_];
"""
    text += expressions
    text += """
}
}  // namespace vmecpp_experiment
#undef VMECPP_GEOMETRY_POINT_INLINE
#endif  // VMECPP_EXPERIMENT_GEOMETRY_POINT_H_
"""
    text.encode("ascii")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
