"""
Various algorithms for RGBD/RGB supervision.
"""

import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import utils


def load_alg(alg_type):
    if alg_type.lower() in ('sgd', 'admm', 'gd', 'gradient-descent'):
        algorithm = gradient_descent

    return algorithm


def gradient_descent(init_phase, target_amp, target_mask=None, forward_prop=None, num_iters=1000, roi_res=None,
                     loss_fn=nn.MSELoss(), lr=0.01, out_path_idx='./results',
                     citl=False, camera_prop=None, writer=None, admm_opt=None,
                     *args, **kwargs):
    """
    Gradient-descent based method for phase optimization.

    :param init_phase:
    :param target_amp:
    :param target_mask:
    :param forward_prop:
    :param num_iters:
    :param roi_res:
    :param loss_fn:
    :param lr:
    :param out_path_idx:
    :param citl:
    :param camera_prop:
    :param writer:
    :param args:
    :param kwargs:
    :return:
    """

    assert forward_prop is not None
    dev = init_phase.device
    num_iters_admm_inner = 1 if admm_opt is None else admm_opt['num_iters_inner']

    slm_phase = init_phase.requires_grad_(True)  # phase at the slm plane
    optvars = [{'params': slm_phase}]
    optimizer = optim.Adam(optvars, lr=lr)

    loss_vals = []
    loss_vals_quantized = []
    best_loss = 10.
    best_iter = 0

    if target_mask is not None:
        target_amp = target_amp * target_mask
        nonzeros = target_mask > 0
    if roi_res is not None:
        target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)
        if target_mask is not None:
            target_mask = utils.crop_image(target_mask, roi_res, stacked_complex=False)
            nonzeros = target_mask > 0

    if admm_opt is not None:
        u = torch.zeros(1, 1, *roi_res).to(dev)
        z = torch.zeros(1, 1, *roi_res).to(dev)

    for t in range(num_iters):
        for t_inner in range(num_iters_admm_inner):
            optimizer.zero_grad()

            recon_field = forward_prop(slm_phase)
            recon_field = utils.crop_image(recon_field, roi_res, stacked_complex=False)
            recon_amp = recon_field.abs()

            if citl:  # surrogate gradients for CITL
                captured_amp = camera_prop(slm_phase)
                captured_amp = utils.crop_image(captured_amp, roi_res,
                                                stacked_complex=False)
                recon_amp = recon_amp + captured_amp - recon_amp.detach()

            if target_mask is not None:
                final_amp = torch.zeros_like(recon_amp)
                final_amp[nonzeros] += (recon_amp[nonzeros] * target_mask[nonzeros])
            else:
                final_amp = recon_amp

            with torch.no_grad():
                s = (final_amp * target_amp).mean() / \
                    (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

            loss_val = loss_fn(s * final_amp, target_amp)

            # second loss term if ADMM
            if admm_opt is not None:
                # augmented lagrangian
                recon_phase = recon_field.angle()
                loss_prior = loss_fn(utils.laplacian(recon_phase) * target_mask, (z - u) * target_mask)
                loss_val = loss_val + admm_opt['rho'] * loss_prior

            loss_val.backward()
            optimizer.step()

        ## ADMM steps
        if admm_opt is not None:
            with torch.no_grad():
                reg_norm = utils.laplacian(recon_phase).detach() * target_mask
                Ax = admm_opt['alpha'] * reg_norm + (1 - admm_opt['alpha']) * z  # over-relaxation
                z = utils.soft_thresholding(u + Ax, admm_opt['gamma'] / (rho + 1e-10))
                u = u + Ax - z

                # varying penalty (rho)
                if admm_opt['varying-penalty']:
                    if t == 0:
                        z_prev = z

                    r_k = ((reg_norm - z).detach() ** 2).mean()  # primal residual
                    s_k = ((rho * utils.laplacian(z_prev - z).detach()) ** 2).mean()  # dual residual

                    if r_k > admm_opt['mu'] * s_k:
                        rho = admm_opt['tau_incr'] * rho
                        u /= admm_opt['tau_incr']
                    elif s_k > admm_opt['mu'] * r_k:
                        rho /= admm_opt['tau_decr']
                        u *= admm_opt['tau_decr']
                    z_prev = z

        with torch.no_grad():
            if loss_val < best_loss:
                best_phase = slm_phase
                best_loss = loss_val
                best_amp = s * recon_amp
                best_iter = t + 1
    print(f' -- optimization is done, best loss: {best_loss}')

    return {'loss_vals': loss_vals,
            'loss_vals_q': loss_vals_quantized,
            'best_iter': best_iter,
            'best_loss': best_loss,
            'recon_amp': best_amp,
            'final_phase': best_phase}