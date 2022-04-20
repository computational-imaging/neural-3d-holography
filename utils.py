"""
Utils
"""

import math
import random
import numpy as np

import os
import torch
import torch.nn as nn

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import GaussianBlur

from skimage.restoration import inpaint
import kornia
import torch.nn.functional as F
import torch.fft as tfft

def roll_torch(tensor, shift: int, axis: int):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=True, lf=False):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if lf:
        size_diff = np.array(field.shape[-4:-2]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-4:-2]) % 2
    else:
        if pytorch:
            if stacked_complex:
                size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-3:-1]) % 2
            else:
                size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-2:]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if lf:
            return field[(..., *crop_slices, slice(None), slice(None))]
        else:
            if pytorch and stacked_complex:
                return field[(..., *crop_slices, slice(None))]
            else:
                return field[(..., *crop_slices)]
    else:
        return field


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def phasemap_8bit(phasemap, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed

    Input
    -----
    :param phasemap: input phasemap tensor, which is supposed to be in the range of [-pi, pi].
    :param inverted: a boolean value that indicates whether the phasemap is inverted.

    Output
    ------
    :return: output phasemap, with uint8 dtype (in [0, 255])
    """

    output_phase = ((phasemap + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    return phase_out_8bit


def burst_img_processor(img_burst_list):
    img_tensor = np.stack(img_burst_list, axis=0)
    img_avg = np.mean(img_tensor, axis=0)
    return im2float(img_avg)  # changed from int8 to float32


def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def get_psnr_ssim(recon_amp, target_amp, multichannel=False):
    """get PSNR and SSIM metrics"""
    psnrs, ssims = {}, {}

    # amplitude
    psnrs['amp'] = psnr(target_amp, recon_amp)
    ssims['amp'] = ssim(target_amp, recon_amp, multichannel=multichannel)

    # linear
    target_linear = target_amp**2
    recon_linear = recon_amp**2
    psnrs['lin'] = psnr(target_linear, recon_linear)
    ssims['lin'] = ssim(target_linear, recon_linear, multichannel=multichannel)

    # srgb
    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    psnrs['srgb'] = psnr(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, multichannel=multichannel)

    return psnrs, ssims


def make_kernel_gaussian(sigma, kernel_size):

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2
    variance = sigma**2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = ((1 / (2 * math.pi * variance))
                       * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)
                                   / (2 * variance)))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    return gaussian_kernel


def pad_stacked_complex(field, pad_width, padval=0):
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, value=padval)
        imag = nn.functional.pad(imag, pad_width, value=0)
        return torch.stack((real, imag), -1)


def srgb_gamma2lin(im_in):
    """ converts from sRGB to linear color space """
    thresh = 0.04045
    if torch.is_tensor(im_in):
        low_val = im_in <= thresh
        im_out = torch.zeros_like(im_in)
        im_out[low_val] = 25 / 323 * im_in[low_val]
        im_out[torch.logical_not(low_val)] = ((200 * im_in[torch.logical_not(low_val)] + 11)
                                                / 211) ** (12 / 5)
    else:
        im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055) ** (12/5))

    return im_out


def srgb_lin2gamma(im_in):
    """ converts from linear to sRGB color space """
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out


def decompose_depthmap(depthmap_virtual_D, depth_planes_D):
    """ decompose a depthmap image into a set of masks with depth positions (in Diopter) """

    num_planes = len(depth_planes_D)

    masks = torch.zeros(depthmap_virtual_D.shape[0], len(depth_planes_D), *depthmap_virtual_D.shape[-2:],
                        dtype=torch.float32).to(depthmap_virtual_D.device)
    for k in range(len(depth_planes_D) - 1):
        depth_l = depth_planes_D[k]
        depth_h = depth_planes_D[k + 1]
        idxs = (depthmap_virtual_D >= depth_l) & (depthmap_virtual_D < depth_h)
        close_idxs = (depth_h - depthmap_virtual_D) > (depthmap_virtual_D - depth_l)

        # closer one
        mask = torch.zeros_like(depthmap_virtual_D)
        mask += idxs * close_idxs * 1
        masks[:, k, ...] += mask.squeeze(1)

        # farther one
        mask = torch.zeros_like(depthmap_virtual_D)
        mask += idxs * (~close_idxs) * 1
        masks[:, k + 1, ...] += mask.squeeze(1)

    # even closer ones
    idxs = depthmap_virtual_D >= max(depth_planes_D)
    mask = torch.zeros_like(depthmap_virtual_D)
    mask += idxs * 1
    masks[:, len(depth_planes_D) - 1, ...] += mask.clone().squeeze(1)

    # even farther ones
    idxs = depthmap_virtual_D < min(depth_planes_D)
    mask = torch.zeros_like(depthmap_virtual_D)
    mask += idxs * 1
    masks[:, 0, ...] += mask.clone().squeeze(1)

    # sanity check
    assert torch.sum(masks).item() == torch.numel(masks) / num_planes

    return masks


def prop_dist_to_diopter(prop_dists, focal_distance, prop_dist_inf, from_lens=True):
    """
    Calculates distance from the user in diopter unit given the propagation distance from the SLM.
    :param prop_dists:
    :param focal_distance:
    :param prop_dist_inf:
    :param from_lens:
    :return:
    """
    x0 = prop_dist_inf  # prop distance from SLM that correcponds to optical infinity from the user
    f = focal_distance  # focal distance of eyepiece

    if from_lens:  # distance is from the lens
        diopters = [1 / (x0 + f - x) - 1 / f for x in prop_dists]  # diopters from the user side
    else:  # distance is from the user (basically adding focal length)
        diopters = [(x - x0) / f**2 for x in prop_dists]

    return diopters


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


def laplacian(img):

    # signed angular difference
    grad_x1, grad_y1 = grad(img, next_pixel=True)  # x_{n+1} - x_{n}
    grad_x0, grad_y0 = grad(img, next_pixel=False)  # x_{n} - x_{n-1}

    laplacian_x = grad_x1 - grad_x0  # (x_{n+1} - x_{n}) - (x_{n} - x_{n-1})
    laplacian_y = grad_y1 - grad_y0

    return laplacian_x + laplacian_y


def grad(img, next_pixel=False, sovel=False):
    
    if img.shape[1] > 1:
        permuted = True
        img = img.permute(1, 0, 2, 3)
    else:
        permuted = False
    
    # set diff kernel
    if sovel:  # use sovel filter for gradient calculation
        k_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32) / 8
        k_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 8
    else:
        if next_pixel:  # x_{n+1} - x_n
            k_x = torch.tensor([[0, -1, 1]], dtype=torch.float32)
            k_y = torch.tensor([[1], [-1], [0]], dtype=torch.float32)
        else:  # x_{n} - x_{n-1}
            k_x = torch.tensor([[-1, 1, 0]], dtype=torch.float32)
            k_y = torch.tensor([[0], [1], [-1]], dtype=torch.float32)

    # upload to gpu
    k_x = k_x.to(img.device).unsqueeze(0).unsqueeze(0)
    k_y = k_y.to(img.device).unsqueeze(0).unsqueeze(0)

    # boundary handling (replicate elements at boundary)
    img_x = F.pad(img, (1, 1, 0, 0), 'replicate')
    img_y = F.pad(img, (0, 0, 1, 1), 'replicate')

    # take sign angular difference
    grad_x = signed_ang(F.conv2d(img_x, k_x))
    grad_y = signed_ang(F.conv2d(img_y, k_y))
    
    if permuted:
        grad_x = grad_x.permute(1, 0, 2, 3)
        grad_y = grad_y.permute(1, 0, 2, 3)

    return grad_x, grad_y


def signed_ang(angle):
    """
    cast all angles into [-pi, pi]
    """
    return (angle + math.pi) % (2*math.pi) - math.pi


# Adapted from https://github.com/svaiter/pyprox/blob/master/pyprox/operators.py
def soft_thresholding(x, gamma):
    """
    return element-wise shrinkage function with threshold kappa
    """
    return torch.maximum(torch.zeros_like(x),
                         1 - gamma / torch.maximum(torch.abs(x), 1e-10*torch.ones_like(x))) * x


def random_gen(num_planes=7, slm_type='ti'):
    """
    random hyperparameters for the dataset
    """
    frame_choices = [1, 2, 3, 3, 4, 4, 8, 8, 8, 8] if slm_type.lower() == 'ti' else [1]

    num_iters = random.choice(range(3000))
    phase_range = random.uniform(1.0, 6.28)
    target_range = random.uniform(0.5, 1.5)
    learning_rate = random.uniform(0.01, 0.03)
    plane_idx = random.choice(range(num_planes))

    return num_iters, phase_range, target_range, learning_rate, plane_idx
