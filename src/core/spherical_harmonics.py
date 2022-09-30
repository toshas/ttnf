"""
Taken from the following BSD-2 source:
https://github.com/sxyu/svox2/blob/master/svox2/utils.py
"""

import torch


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def spherical_harmonics_bases(basis_dim: int, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics bases at unit directions, without taking linear combination. At each point, the final
    result may the be obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    assert basis_dim in (1, 4, 9, 16, 25)
    out = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    out[..., 0] = SH_C0

    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        out[..., 1] = -SH_C1 * y
        out[..., 2] = SH_C1 * z
        out[..., 3] = -SH_C1 * x

        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            out[..., 4] = SH_C2[0] * xy
            out[..., 5] = SH_C2[1] * yz
            out[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            out[..., 7] = SH_C2[3] * xz
            out[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                out[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                out[..., 10] = SH_C3[1] * xy * z
                out[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                out[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                out[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                out[..., 14] = SH_C3[5] * z * (xx - yy)
                out[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    out[..., 16] = SH_C4[0] * xy * (xx - yy)
                    out[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    out[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    out[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    out[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    out[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    out[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    out[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    out[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return out
