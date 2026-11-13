"""Algebraic forward-poloidal GEMM alternatives; original reference retained."""

import numpy as np


def export_poloidal(x, inputs, poly, partial, jf, rz, export, prefix, ordered, digest):
    if not {"forward-poloidal", "forward-poloidal-packed"} & export.enabled:
        return
    radial, mpol, nz = partial.shape[1:]
    theta = poly[0].shape[1]
    # name, even input channel, required basis IDs, physical input row group.
    fields = [
        ("armn", 0, [0, 1], "rz"),
        ("brmn", 6, [3, 2], "rz"),
        ("frcon", 16, [0, 1], "rz"),
        ("crmn", 12, [0, 1], "rz"),
        ("azmn", 2, [1, 0], "rz"),
        ("bzmn", 8, [2, 3], "rz"),
        ("fzcon", 18, [1, 0], "rz"),
        ("czmn", 14, [1, 0], "rz"),
        ("blmn", 4, [2, 3], "lambda"),
        ("clmn", 10, [1, 0], "lambda"),
    ]
    decomposed = np.zeros_like(partial)
    combinations = []
    small_flops = 0
    packed_flops = 0
    for parity in (0, 1):
        modes = np.arange(parity, mpol, 2)
        products = {}
        packed_inputs = []
        for label, base, basis_ids, group in fields:
            rows = np.flatnonzero(rz) if group == "rz" else np.arange(radial)
            a = inputs[base + parity][rows, :, :theta].reshape(len(rows) * nz, theta)
            padded = np.zeros((radial, nz, theta))
            padded[rows] = a.reshape(len(rows), nz, theta)
            packed_inputs.append(padded.reshape(radial * nz, theta))
            for basis in basis_ids:
                b = poly[basis][modes].T
                c = ordered(a, b)
                name = f"{prefix}-forward-poloidal-{label}-{parity}-{basis}"
                export.write(
                    "forward-poloidal",
                    name,
                    a,
                    b,
                    c,
                    row_order="selected radial row*nZeta+k",
                    original_radial_rows=rows.tolist(),
                    poloidal_modes=modes.tolist(),
                    reduction="l ascending, single product",
                    basis_index=basis,
                    source_input_index=base + parity,
                    scope="one constituent product; epilogue not included",
                )
                small_flops += 2 * a.shape[0] * b.shape[1] * theta
                values = np.zeros((radial, nz, len(modes)))
                values[rows] = c.reshape(len(rows), nz, len(modes))
                products[label, basis] = (values, name)
        packed_a = np.concatenate(packed_inputs, axis=0)
        packed_b = np.concatenate([basis[modes].T for basis in poly], axis=1)
        packed_c = ordered(packed_a, packed_b)
        packed_flops += 2 * packed_a.shape[0] * packed_b.shape[1] * theta
        export.write(
            "forward-poloidal-packed",
            f"{prefix}-forward-poloidal-packed-{parity}",
            packed_a,
            packed_b,
            packed_c,
            row_order="field_index*(radial*nZeta)+(jF-lo)*nZeta+k",
            field_order=[item[0] for item in fields],
            column_order="basis_index*parity_mode_count+parity_mode_index",
            poloidal_modes=modes.tolist(),
            reduction="l ascending, single product",
            extra_work="all four bases per field; only two needed, plus zero-padded R/Z LCFS rows",
            scope="one packed alternative; extraction and epilogue not included",
        )
        for field_index, (label, _base, basis_ids, _group) in enumerate(fields):
            for basis in basis_ids:
                view = packed_c[
                    field_index * radial * nz : (field_index + 1) * radial * nz,
                    basis * len(modes) : (basis + 1) * len(modes),
                ]
                assert np.array_equal(
                    view.view(np.uint64),
                    products[label, basis][0]
                    .reshape(radial * nz, len(modes))
                    .view(np.uint64),
                )
        # A specific, explicit reassociated epilogue, not the original loop DAG.
        values = [
            products["armn", 0][0]
            + products["brmn", 3][0]
            + x[20][modes] * products["frcon", 0][0],
            -products["crmn", 0][0],
            products["armn", 1][0]
            + products["brmn", 2][0]
            + x[20][modes] * products["frcon", 1][0],
            -products["crmn", 1][0],
            products["azmn", 1][0]
            + products["bzmn", 2][0]
            + x[20][modes] * products["fzcon", 1][0],
            -products["czmn", 1][0],
            products["azmn", 0][0]
            + products["bzmn", 3][0]
            + x[20][modes] * products["fzcon", 0][0],
            -products["czmn", 0][0],
            products["blmn", 2][0],
            -products["clmn", 1][0],
            products["blmn", 3][0],
            -products["clmn", 0][0],
        ]
        for channel, value in enumerate(values):
            for index, m in enumerate(modes):
                decomposed[channel, :, m] = value[:, :, index]
                decomposed[channel, (jf == 0) & (m > 0), m] = 0.0
        combinations.append(
            {
                "parity": parity,
                "modes": modes.tolist(),
                "products": {
                    f"{field}.{basis}": name
                    for (field, basis), (_value, name) in products.items()
                },
                "R_values": [
                    "(armn.0+brmn.3)+xmpq*frcon.0",
                    "(armn.1+brmn.2)+xmpq*frcon.1",
                ],
                "Z_values": [
                    "(azmn.1+bzmn.2)+xmpq*fzcon.1",
                    "(azmn.0+bzmn.3)+xmpq*fzcon.0",
                ],
                "R_derivatives": ["-crmn.0", "-crmn.1"],
                "Z_derivatives": ["-czmn.1", "-czmn.0"],
                "lambda_values": ["blmn.2", "blmn.3"],
                "lambda_derivatives": ["-clmn.1", "-clmn.0"],
            }
        )
    path = export.out / (prefix + "-forward-poloidal-reference.npz")
    np.savez(
        path,
        original_ordered_partials=partial,
        decomposed_ordered_products_partials=decomposed,
        xmpq=x[20][:mpol],
        radial_indices=jf,
    )
    delta = np.abs(decomposed - partial)
    export.artifacts.append(
        {
            "file": path.name,
            "sha256": digest(path),
            "family": "forward-poloidal",
            "small_GEMM_count": 40,
            "packed_GEMM_count": 2,
            "small_GEMM_2MNK": small_flops,
            "packed_GEMM_2MNK": packed_flops,
            "packed_over_small_FLOP_ratio": packed_flops / small_flops,
            "epilogue_required": True,
            "extra_RZ_epilogue_per_point_mode": "4 multiplies + 8 adds",
            "extra_derivative_epilogue_per_point_mode": "6 sign negations; two lambda copies",
            "packing_and_extraction_cost_not_timed": True,
            "axis_mask_required": True,
            "combinations": combinations,
            "decomposition_vs_original_bit_differences": int(
                np.count_nonzero(decomposed.view(np.uint64) != partial.view(np.uint64))
            ),
            "decomposition_vs_original_max_absolute": float(delta.max(initial=0)),
            "reference": "original serial-l FP64 partials and explicit alternative decomposed result",
        }
    )
