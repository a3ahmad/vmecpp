"""Fuse per-mode staging/Thomas solves and parallelize independent constraints."""

from __future__ import annotations

import argparse
from pathlib import Path

from parallel_pointwise import function_bounds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    path = args.source / (
        "src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc"
    )
    text = path.read_text()
    if "RadialExecutor& FourierExecutor()" not in text:
        parser.error("requires the persistent-worker experiment")
    if "VMECPP_PARALLEL_PRECONDITIONER" in text:
        parser.error("experiment already applied")
    start, end = function_bounds(
        text, "absl::Status IdealMhdModel::applyRZPreconditioner("
    )
    body = text[start:end]
    gather_marker = "  // gather everything into HandoverStorage (flat layout)"
    scatter_marker = "  // re-distribute solution back into threads"
    gather = body[body.index(gather_marker) : body.index(scatter_marker)]
    scatter = body[body.index(scatter_marker) :]
    mode_loop = "for (int mn = 0; mn < s_.mnsize; ++mn)"
    first, last = function_bounds(gather, mode_loop)
    gather_mode = gather[first:last]
    first, last = function_bounds(scatter, mode_loop)
    scatter_mode = scatter[first:last]
    first, last = function_bounds(body, "for (int mn = mnmin; mn < mnmax; ++mn)")
    solve_mode = body[first:last]
    branch = """
  const char* parallel_preconditioner =
      std::getenv("VMECPP_PARALLEL_PRECONDITIONER");
  if (parallel_preconditioner != nullptr &&
      std::strcmp(parallel_preconditioner, "1") == 0 &&
      r_.get_num_threads() == 1) {
    // Systems are independent. Keep each Thomas dependency chain unchanged.
    // Staging one mode immediately before solving retains its rows in cache.
    constexpr int kModeBlockSize = 16;
    const int blocks = (s_.mnsize + kModeBlockSize - 1) / kModeBlockSize;
    const int ns = m_h_.all_ar.cols();
    FourierExecutor().Run(0, blocks, [&, this](int block) {
      const int first_mode = block * kModeBlockSize;
      const int last_mode = std::min(first_mode + kModeBlockSize, s_.mnsize);
      for (int mn = first_mode; mn < last_mode; ++mn) {
        for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
"""
    branch += gather_mode + "\n        }\n" + solve_mode
    branch += "\n        for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {\n"
    branch += scatter_mode + "\n        }\n      }\n    });\n"
    branch += "    return absl::OkStatus();\n  }\n\n"
    body = body.replace(gather_marker, branch + gather_marker, 1)
    text = text[:start] + body + text[end:]

    start, end = function_bounds(text, "void vmecpp::deAliasConstraintForce(")
    body = text[start:end]
    loop = "for (int jF = std::max(jMin, rp.nsMinF); jF < rp.nsMaxF; ++jF)"
    first, last = function_bounds(body, loop)
    inner = body[first:last]
    inner = inner.replace("m_gsc", "local_gsc").replace("m_gcs", "local_gcs")
    branch = """
  const char* parallel_constraint = std::getenv("VMECPP_PARALLEL_CONSTRAINT");
  if (parallel_constraint != nullptr &&
      std::strcmp(parallel_constraint, "1") == 0 &&
      rp.get_num_threads() == 1) {
    FourierExecutor().Run(std::max(jMin, rp.nsMinF), rp.nsMaxF, [&](int jF) {
      // Reuse thread-local scratch; radial outputs have no shared reductions.
      thread_local std::vector<double> local_gsc;
      thread_local std::vector<double> local_gcs;
      local_gsc.resize(s_.ntor + 1);
      local_gcs.resize(s_.ntor + 1);
"""
    branch += inner
    branch += """
      if (jF == rp.nsMaxF - 1 && s_.mpol > 2) {
        // Preserve even the final scratch state of the original serial loop.
        std::copy(local_gsc.begin(), local_gsc.end(), m_gsc.begin());
        std::copy(local_gcs.begin(), local_gcs.end(), m_gcs.begin());
      }
    });
    return;
  }

"""
    position = body.index("  " + loop)
    body = body[:position] + branch + body[position:]
    text = text[:start] + body + text[end:]
    text.encode("ascii")
    path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
