"""
Functions for zernike basis

"""

import math
import torch
import numpy as np
import utils
import torch.fft
from aotools.functions import zernikeArray


def combine_zernike_basis(coeffs, basis, return_phase=False):
    """
    Multiplies the Zernike coefficients and basis functions while preserving
    dimensions

    :param coeffs: torch tensor with coeffs, see propagation_ASM_zernike
    :param basis: the output of compute_zernike_basis, must be same length as coeffs
    :param return_phase:
    :return: A float32 tensor that combines coeffs and basis.
    """

    if len(coeffs.shape) < 3:
        coeffs = torch.reshape(coeffs, (coeffs.shape[0], 1, 1))

    # combine zernike basis and coefficients
    zernike = (coeffs * basis).sum(0, keepdim=True)

    # shape to [1, len(coeffs), H, W]
    zernike = zernike.unsqueeze(0)

    return zernike


def compute_zernike_basis(num_polynomials, field_res, dtype=torch.float32, wo_piston=False):
    """Computes a set of Zernike basis function with resolution field_res

    num_polynomials: number of Zernike polynomials in this basis
    field_res: [height, width] in px, any list-like object
    dtype: torch dtype for computation at different precision
    """

    # size the zernike basis to avoid circular masking
    zernike_diam = int(np.ceil(np.sqrt(field_res[0]**2 + field_res[1]**2)))

    # create zernike functions

    if not wo_piston:
        zernike = zernikeArray(num_polynomials, zernike_diam)
    else:  # 200427 - exclude pistorn term
        idxs = range(2, 2 + num_polynomials)
        zernike = zernikeArray(idxs, zernike_diam)

    zernike = utils.crop_image(zernike, field_res, pytorch=False)

    # convert to tensor and create phase
    zernike = torch.tensor(zernike, dtype=dtype, requires_grad=False)

    return zernike
