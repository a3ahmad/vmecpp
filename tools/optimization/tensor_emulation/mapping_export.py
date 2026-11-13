"""CPU-only actual-capture GEMM export, with ordered FP64 references.

GEMM reassociation is an alternative configuration, never the identity path. Requires
NumPy; does not call BLAS, CUDA or load a solver DSO.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from mapping_dense2d import export_dense_inverse
from mapping_poloidal import export_poloidal


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(path, inputs, outputs, magic):
    data = path.read_bytes()
    assert data[:8] == magic
    dims = struct.unpack_from("<13i", data, 8)
    offset = 60
    sizes = struct.unpack_from(f"<{inputs + outputs}Q", data, offset)
    offset += 8 * len(sizes)
    assert all(x <= 10000000 for x in sizes)
    assert offset + 8 * sum(sizes) == len(data)
    arrays = []
    for size in sizes:
        arrays.append(np.frombuffer(data, dtype="<f8", count=size, offset=offset))
        offset += 8 * size
    assert all(np.isfinite(x).all() for x in arrays)
    return dims, arrays[:inputs], arrays[inputs:]


def ordered(a, b, pair=False, subtract=False):
    """Serial reduction; each NumPy operation is a separate FP64 operation."""
    c = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
    stride = 2 if pair else 1
    for k in range(0, a.shape[1], stride):
        term = a[:, k, None] * b[None, k, :]
        if pair:
            term = term + a[:, k + 1, None] * b[None, k + 1, :]
        c = c - term if subtract else c + term
    return c


class Export:
    def __init__(self, out, enabled):
        self.out = out
        self.enabled = set(enabled)
        self.jobs = []
        self.artifacts = []
        self.repeat_rows = [1]

    def write(self, family, name, a, b, c, **metadata):
        if family not in self.enabled:
            return
        assert a.ndim == b.ndim == c.ndim == 2
        m, k = a.shape
        assert b.shape[0] == k
        assert c.shape == (m, b.shape[1])
        n = b.shape[1]
        path = self.out / (name + ".bin")
        with path.open("wb") as stream:
            stream.write(b"TEGEMM1\0")
            stream.write(struct.pack("<3Q", m, n, k))
            for values in (a, b, c):
                stream.write(np.asarray(values, dtype="<f8").tobytes(order="F"))
        self.jobs.append(
            dict(
                name=name,
                family=family,
                file=path.name,
                sha256=digest(path),
                M=m,
                N=n,
                K=k,
                transA="N",
                transB="N",
                lda=m,
                ldb=k,
                ldc=m,
                alpha=1,
                beta=0,
                reference="original ordered FP64 contraction; GEMM may reassociate",
                **metadata,
            )
        )


def inverse(d, x, expected, export, prefix):
    lo, _hi, low1, high1, lcfs, mpol, ntor, nz, eff, theta, nyq, _ns, _free = d
    radial = high1 - low1
    mode = np.arange(mpol)
    jf = np.arange(low1, high1)
    valid = ~((jf[:, None] == 0) & (mode[None, :] > 1))
    coeff = [a.reshape(radial, mpol, ntor + 1) for a in x[:6]]
    bases = [a.reshape(nz, nyq + 1)[:, : ntor + 1].T for a in x[12:16]]
    # channel, coefficient channel, basis: cosnv/sinnv/cosnvn/sinnvn.
    routes = [
        (0, 0),
        (0, 3),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 3),
        (5, 1),
        (5, 2),
    ]
    partial = np.zeros((12, radial, mpol, nz))
    for channel, (source, basis) in enumerate(routes):
        a = coeff[source].reshape(-1, ntor + 1)
        c = ordered(a, bases[basis])
        partial[channel] = c.reshape(radial, mpol, nz)
        partial[channel, ~valid] = 0.0
        rows = np.flatnonzero(valid.ravel())
        export.write(
            "inverse-toroidal",
            f"{prefix}-inv-toroidal-{channel:02}",
            a[rows],
            bases[basis],
            c[rows],
            row_order="original (jF-low1)*mpol+m, inactive axis rows removed",
            source_coefficient=source,
            source_basis=basis,
            original_active_rows=rows.tolist(),
            reduction="n ascending",
        )
    poloidal = [a[: mpol * theta].reshape(mpol, theta) for a in x[8:12]]
    # r1,ru,rv,z1,zu,zv,lu,lv: partial pair and poloidal pair.
    routes = [
        (0, 2, 0, 1),
        (0, 2, 3, 2),
        (1, 3, 0, 1),
        (4, 6, 1, 0),
        (4, 6, 2, 3),
        (5, 7, 1, 0),
        (8, 10, 2, 3),
        (9, 11, 1, 0),
    ]
    actual = [np.zeros(a.shape, dtype=np.float64) for a in expected]
    for channel, (p, q, u, v) in enumerate(routes):
        for parity in (0, 1):
            modes = np.arange(parity, mpol, 2)
            a = np.empty((radial * nz, 2 * len(modes)))
            b = np.empty((2 * len(modes), theta))
            c = np.zeros((radial, nz, theta))
            for index, m in enumerate(modes):
                left = partial[p, :, m, :]
                right = partial[q, :, m, :]
                a[:, 2 * index] = left.ravel()
                a[:, 2 * index + 1] = right.ravel()
                b[2 * index] = poloidal[u][m]
                b[2 * index + 1] = poloidal[v][m]
                term = (
                    left[:, :, None] * poloidal[u][m]
                    + right[:, :, None] * poloidal[v][m]
                )
                c = c - term if channel == 7 else c + term
            if channel == 7:
                b = -b
            export.write(
                "inverse-main",
                f"{prefix}-inv-main-{channel}-{parity}",
                a,
                b,
                c.reshape(radial * nz, theta),
                row_order="(jF-low1)*nZeta+k",
                reduction="m ascending within parity, paired products before accumulation",
                sign="original subtract" if channel == 7 else "original add",
            )
            actual[2 * channel + parity].reshape(radial, nz, eff)[:, :, :theta] = c
    # Original constraint expression: factor is applied AFTER the two-product
    # addition. Folding it into GEMM operands changes rounding, so do not export
    # it as if it were the same contraction.
    for m in range(mpol):
        active = (jf >= lo) & (jf < lcfs) & valid[:, m]
        sqrt_s = np.ones(radial) if m % 2 == 0 else x[7]
        factor = x[6][m] * sqrt_s
        for out_channel, p, q, u, v in [(16, 0, 2, 0, 1), (17, 4, 6, 1, 0)]:
            term = (
                partial[p, :, m, :, None] * poloidal[u][m]
                + partial[q, :, m, :, None] * poloidal[v][m]
            ) * factor[:, None, None]
            for r in np.flatnonzero(active):
                target = actual[out_channel].reshape(lcfs - lo, nz, eff)[
                    jf[r] - lo, :, :theta
                ]
                target += term[r]
    export_dense_inverse(d, x, expected, export, prefix, export.repeat_rows, ordered)
    return compare(actual, expected)


def forward(d, x, expected, export, prefix):
    lo, hi, _low1, _high1, lcfs, mpol, ntor, nz, eff, theta, nyq, ns, _free = d
    radial = lcfs - lo
    jf = np.arange(lo, lcfs)
    rz = (jf < hi) & (jf < ns - 1)
    inputs = []
    for values in x[:20]:
        padded = np.zeros((radial, nz, eff))
        rows = len(values) // (nz * eff)
        padded[:rows] = values.reshape(rows, nz, eff)
        inputs.append(padded)
    poly = [a[: mpol * theta].reshape(mpol, theta) for a in x[21:25]]
    tor = [a.reshape(nz, nyq + 1)[:, : ntor + 1] for a in x[25:29]]
    partial = np.zeros((12, radial, mpol, nz))
    for m in range(mpol):
        parity = m % 2
        valid = ~((jf == 0) & (m > 0))
        for l in range(theta):  # noqa: E741 - original poloidal index
            co, si, com, sim = (p[m, l] for p in poly)
            a = [p[:, :, l] for p in inputs]
            tr = a[parity] + x[20][m] * a[16 + parity]
            tz = a[2 + parity] + x[20][m] * a[18 + parity]
            partial[1, :, m] -= a[12 + parity] * co
            partial[3, :, m] -= a[12 + parity] * si
            partial[0, :, m] += tr * co + a[6 + parity] * sim
            partial[2, :, m] += tr * si + a[6 + parity] * com
            partial[7, :, m] -= a[14 + parity] * co
            partial[5, :, m] -= a[14 + parity] * si
            partial[4, :, m] += tz * si + a[8 + parity] * com
            partial[6, :, m] += tz * co + a[8 + parity] * sim
            partial[8, :, m] += a[4 + parity] * com
            partial[10, :, m] += a[4 + parity] * sim
            partial[11, :, m] -= a[10 + parity] * co
            partial[9, :, m] -= a[10 + parity] * si
        partial[:8, ~rz, m] = 0.0
        partial[:, ~valid, m] = 0.0
    export_poloidal(x, inputs, poly, partial, jf, rz, export, prefix, ordered, digest)
    actual = []
    for channel in range(6):
        group, part = divmod(channel, 2)
        p, q = 4 * group + 2 * part, 4 * group + 2 * part + 1
        enabled = jf >= 1 if group == 2 else rz
        a = np.empty((radial * mpol, 2 * nz))
        a[:, 0::2] = partial[p].reshape(radial * mpol, nz)
        a[:, 1::2] = partial[q].reshape(radial * mpol, nz)
        a.reshape(radial, mpol, 2 * nz)[~enabled] = 0.0
        b = np.empty((2 * nz, ntor + 1))
        b[0::2] = tor[0 if part == 0 else 1]
        b[1::2] = tor[3 if part == 0 else 2]
        c = ordered(a, b, pair=True)
        export.write(
            "forward-finish",
            f"{prefix}-forward-finish-{channel}",
            a,
            b,
            c,
            row_order="(jF-lo)*mpol+m",
            reduction="k ascending, paired products before accumulation",
            source_partial_channels=[p, q],
        )
        actual.append(c.ravel())
    return compare(actual, expected)


def compare(actual, expected):
    results = []
    for a, b in zip(actual, expected, strict=True):
        assert a.shape == b.shape
        delta = np.abs(a - b)
        results.append(
            {
                "bit_differences": int(
                    np.count_nonzero(a.view(np.uint64) != b.view(np.uint64))
                ),
                "max_absolute": float(delta.max(initial=0)),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "tools/optimization/gb202/sources.json",
    )
    parser.add_argument(
        "--ns", nargs="+", type=int, choices=[25, 51, 99], default=[25, 51, 99]
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=[
            "inverse-toroidal",
            "inverse-main",
            "forward-finish",
            "forward-poloidal",
            "forward-poloidal-packed",
            "inverse-full2d",
        ],
        default=[
            "inverse-toroidal",
            "inverse-main",
            "forward-finish",
            "forward-poloidal",
            "forward-poloidal-packed",
        ],
    )
    parser.add_argument(
        "--repeat-rows",
        nargs="+",
        type=int,
        choices=[1, 4, 16, 64],
        default=[1, 4, 16, 64],
    )
    args = parser.parse_args()
    source_manifest = json.loads(args.source_manifest.read_text())
    source_provenance = {}
    for name, value in source_manifest["files"].items():
        if any(
            original.endswith(
                (
                    "inverse-load-hoist1/source/generated_fourier_kernel.cuh",
                    "inverse-load-hoist1/source/gpu_fourier.cu",
                    "forward-clear-elide1/source/gpu_forward.cu",
                    "forward-clear-elide1/source/gpu_forward_groups_kernels.h",
                )
            )
            for original in value["originals"]
        ):
            path = args.source_manifest.parent / name
            assert digest(path) == value["sha256"]
            source_provenance[name] = value
    assert len(source_provenance) == 4
    assert not args.out.exists(), "Use a fresh output directory"
    args.out.mkdir(parents=True)
    export = Export(args.out, args.families)
    export.repeat_rows = args.repeat_rows
    provenance = []
    for ns in args.ns:
        for name, count_in, count_out, magic, function in [
            ("inverse", 16, 18, b"VMECGPU1", inverse),
            ("forward", 29, 6, b"VMECGPU2", forward),
        ]:
            path = args.captures / f"ns{ns}" / (name + ".bin")
            before = digest(path)
            d, x, expected = capture(path, count_in, count_out, magic)
            reference = function(d, x, expected, export, f"ns{ns}")
            assert digest(path) == before
            provenance.append(
                {
                    "capture": str(path.resolve()),
                    "sha256": before,
                    "dimensions": d,
                    "output_reference_comparison": reference,
                }
            )
    receipt = {
        "scope": "CPU export of actual dense contractions, not solver timing or metric qualification",
        "configuration": "GEMM reassociation alternative; original ordered path retained",
        "exporter_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "selected_source_manifest_sha256": digest(args.source_manifest),
        "selected_source_provenance": source_provenance,
        "binary_format": "TEGEMM1\\0 + uint64LE M,N,K + float64LE column-major A,B,original-ordered-C",
        "jobs": export.jobs,
        "artifacts": export.artifacts,
        "captures": provenance,
        "poloidal_exporter_sha256": digest(
            Path(__file__).with_name("mapping_poloidal.py")
        ),
        "dense2d_exporter_sha256": digest(
            Path(__file__).with_name("mapping_dense2d.py")
        ),
        "reference_all_captured_outputs_bitexact": all(
            x["bit_differences"] == 0
            for item in provenance
            for x in item["output_reference_comparison"]
        ),
    }
    (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(  # noqa: T201
        json.dumps(
            {
                "exported_jobs": len(export.jobs),
                "reference_all_captured_outputs_bitexact": receipt[
                    "reference_all_captured_outputs_bitexact"
                ],
                "manifest": str(args.out / "manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
