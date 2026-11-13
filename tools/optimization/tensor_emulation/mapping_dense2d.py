"""Full product-basis inverse research matrices; explicitly reassociated."""

import numpy as np


def export_dense_inverse(d, x, expected, export, prefix, repeats, ordered):
    if "inverse-full2d" not in export.enabled:
        return
    _lo, _hi, low1, high1, _lcfs, mpol, ntor, nz, eff, theta, nyq, _ns, _free = d
    radial = high1 - low1
    jf = np.arange(low1, high1)
    active = jf >= 1
    rows = np.flatnonzero(active)
    cosmu, sinmu = (a[: mpol * theta].reshape(mpol, theta) for a in x[8:10])
    cosnv, sinnv = (a.reshape(nz, nyq + 1)[:, : ntor + 1] for a in x[12:14])
    coefficients = [a.reshape(radial, mpol, ntor + 1) for a in x[:2]]
    captured = [
        a.reshape(radial, nz, eff)[:, :, :theta].reshape(radial, nz * theta)
        for a in expected[:2]
    ]
    for name, modes, reference in [
        ("r1-sum", np.arange(mpol), captured[0] + captured[1]),
        ("r1-even", np.arange(0, mpol, 2), captured[0]),
        ("r1-split-output", np.arange(mpol), None),
    ]:
        a = np.concatenate(
            [
                coefficient[rows][:, modes].reshape(len(rows), -1)
                for coefficient in coefficients
            ],
            axis=1,
        )
        b = np.concatenate(
            [
                (cosmu[modes, None, None, :] * cosnv.T[None, :, :, None]).reshape(
                    -1, nz * theta
                ),
                (sinmu[modes, None, None, :] * sinnv.T[None, :, :, None]).reshape(
                    -1, nz * theta
                ),
            ],
            axis=0,
        )
        c = reference[rows] if reference is not None else None
        if name == "r1-split-output":
            even_a = a.copy()
            odd_a = a.copy()
            column_parity = np.tile(np.repeat(modes % 2, ntor + 1), 2)
            even_a[:, column_parity == 1] = 0.0
            odd_a[:, column_parity == 0] = 0.0
            a = np.concatenate([even_a, odd_a], axis=0)
            c = np.concatenate([captured[0][rows], captured[1][rows]], axis=0)
        assert c is not None
        dense_ordered = ordered(a, b)
        delta = np.abs(dense_ordered - c)
        denominator = np.sum(c * c)
        # Original separable arithmetic for this value only, excluding other
        # derivative/geometry channels that share its kernels.
        separable_2_stage_ops = (
            len(rows) * len(modes) * nz * (ntor + 1) * 4
            + len(rows) * nz * theta * len(modes) * 4
        )
        epilogue_adds = len(rows) * nz * theta if name == "r1-sum" else 0
        for factor in repeats:
            repeat_a = np.tile(a, (factor, 1))
            repeat_c = np.tile(c, (factor, 1))
            export.write(
                "inverse-full2d",
                f"{prefix}-inverse-full2d-{name}-repeat{factor}",
                repeat_a,
                b,
                repeat_c,
                reduction="alternative dense K: all rmncc(m,n), then all rmnss(m,n); explicit product basis",
                row_order="repeat blocks: even radial rows then odd radial rows"
                if name == "r1-split-output"
                else "repeat blocks of actual active radial rows",
                original_radial_indices=jf[rows].tolist(),
                repeat_real_rows_factor=factor,
                repetition_scope="diagnostic repeated SAME rows, not distinct equilibria or solver throughput",
                poloidal_modes=modes.tolist(),
                output_columns="k*nThetaReduced+l",
                basis_products_precomputed=True,
                basis_product_FP64_multiplications=int(b.size),
                captured_reference="captured r1_even + captured r1_odd, one FP64 add"
                if name == "r1-sum"
                else "captured r1_even and r1_odd stacked separately"
                if name == "r1-split-output"
                else "captured r1_even directly",
                zero_mask_row_expansion=2 if name == "r1-split-output" else 1,
                zero_mask_K_fraction=0.5 if name == "r1-split-output" else 0.0,
                physical_radial_sqrt_reconstruction=False,
                original_value_separable_2_stage_ops=separable_2_stage_ops * factor,
                original_reference_epilogue_adds=epilogue_adds * factor,
                dense_nominal_2MNK=int(2 * repeat_a.shape[0] * b.shape[1] * b.shape[0]),
                dense_over_value_separable_ops=float(
                    2 * a.shape[0] * b.shape[1] * b.shape[0] / separable_2_stage_ops
                ),
                CPU_dense_reassociation_bit_differences=int(
                    np.count_nonzero(dense_ordered.view(np.uint64) != c.view(np.uint64))
                ),
                CPU_dense_reassociation_max_absolute=float(delta.max(initial=0)),
                CPU_dense_reassociation_relative_L2=float(
                    np.sqrt(np.sum(delta * delta) / denominator)
                )
                if denominator
                else 0.0,
                scope="full2D product-basis alternative; B preparation and row packing outside GEMM timing",
            )
