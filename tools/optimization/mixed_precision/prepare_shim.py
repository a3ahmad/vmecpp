"""Create the isolated experimental shim from the pinned retained source."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
retained = root.parent / "gb202"
manifest = json.loads((retained / "sources.json").read_text())
name = manifest["runtime_shim"]["source"]
source = (retained / name).read_bytes()
assert hashlib.sha256(source).hexdigest() == manifest["files"][name]["sha256"]
text = source.decode("ascii")
assert text.count("struct Binding {") == 1
text = text.replace("#include <span>", "#include <span>\n#include <string>")
text = text.replace(
    "struct Binding {", '#include "precision_trial_hooks.inc"\n\nstruct Binding {'
)
anchor = "  RunGpuSolver(binding.context_, control, scalars);"
assert text.count(anchor) == 1
text = text.replace(
    anchor, "  PrecisionBeforeForce(binding.owner_, m_fc.ns, iter1, iter2);\n" + anchor
)
anchor = "        iterations_before_checkpointing, verbose);\n    for (int i = 0; i < 6; ++i) {"
assert text.count(anchor) == 1
text = text.replace(
    anchor,
    "        iterations_before_checkpointing, verbose);\n    PrecisionMaterialized(m_fc, result.ok(), result.ok() && !*result &&\n        !materialized_need_restart &&\n        m_fc.restart_reason != RestartReason::BAD_JACOBIAN);\n    for (int i = 0; i < 6; ++i) {",
)
anchor = "  return result;\n}\n\nvoid Vmec::RestartIteration"
assert text.count(anchor) == 1
text = text.replace(
    anchor,
    '  if (PrecisionMode() != nullptr && precision_trial.ns_ >= 0)\n    PrecisionSnapshot("loop_end");\n'
    + anchor,
)
output = root / "resident_solver_shim.cc"
assert not output.exists(), "Use a fresh output; do not overwrite a measured source"
output.write_text(text, encoding="ascii", newline="\n")
print(  # noqa: T201 - Print the explicit source-generation receipt.
    json.dumps(
        {
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "GPU_calls": 0,
        }
    )
)
