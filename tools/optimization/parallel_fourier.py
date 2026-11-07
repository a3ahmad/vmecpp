"""Parallelize independent radial transforms without splitting solver reductions."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    path = args.source / (
        "src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc"
    )
    text = path.read_text()
    if "DeterministicFourierThreads" in text:
        parser.error("patch is already applied")
    helper = """
// Independent radial transforms retain each output's original summation order.
int DeterministicFourierThreads() {
  const char* value = std::getenv("VMECPP_DETERMINISTIC_FOURIER_THREADS");
  if (value == nullptr) { return 1; }
  char* end = nullptr;
  const long count = std::strtol(value, &end, 10);
  if (*end != '\\0' || count < 1 || count > 32) {
    throw std::runtime_error("Fourier thread count must be in [1, 32]");
  }
  return static_cast<int>(count);
}
"""
    text = text.replace("namespace {\n", "namespace {\n" + helper, 1)
    text = text.replace("#include <span>", "#include <span>\n#include <stdexcept>", 1)
    start = text.index("void vmecpp::ForcesToFourier3DSymmFastPoloidal(")
    end = text.index("IdealMhdModel::IdealMhdModel(", start)
    section = text[start:end]
    pragma = (
        "#ifdef _OPENMP\n"
        "#pragma omp parallel for schedule(static) num_threads(DeterministicFourierThreads()) if(DeterministicFourierThreads() > 1)\n"
        "#endif\n"
    )
    loops = [
        "  for (int jF = rp.nsMinF; jF < jMaxRZ; ++jF) {",
        "  for (int jF = jMaxRZ; jF < rp.nsMaxFIncludingLcfs; ++jF) {",
        "  for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {",
    ]
    for loop in loops:
        if section.count(loop) != 1:
            parser.error("unexpected transform loop layout")
        section = section.replace(loop, pragma + loop, 1)
    text = text[:start] + section + text[end:]
    text.encode("ascii")
    path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
