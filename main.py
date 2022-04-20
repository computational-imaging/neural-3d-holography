"""
Neural 3D holography: Learning accurate wave propagation models for 3D holographic virtual and augmented reality displays

Suyeon Choi*, Manu Gopakumar*, Yifan Peng, Jonghyun Kim, Gordon Wetzstein

This is the main executive script used for the phase generation using SGD.
This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
-----

$ python main.py --lr=0.01 --num_iters=10000

"""
import algorithms as algs
import image_loader as loaders
import numpy as np

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import imageio
import configargparse
import prop_physical
import prop_model
import utils
import params
    

def main():
    # Command line argument processing / Parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False,
          is_config_file=True, help='Path to config file.')
    params.add_parameters(p, 'eval')
    opt = p.parse_args()
    params.set_configs(opt)
    dev = torch.device('cuda')

    run_id = params.run_id(opt)
    # path to save out optimized phases
    out_path = os.path.join(opt.out_path, run_id)
    print(f'  - out_path: {out_path}')

    # Tensorboard
    summaries_dir = os.path.join(out_path, 'summaries')
    utils.cond_mkdir(summaries_dir)
    writer = SummaryWriter(summaries_dir)

    # Propagations
    camera_prop = None
    if opt.citl:
        camera_prop = prop_physical.PhysicalProp(*(params.hw_params(opt))).to(dev)
    sim_prop = prop_model.model(opt)

    # Algorithm
    algorithm = algs.load_alg(opt.method)

    # Loader
    if ',' in opt.data_path:
        opt.data_path = opt.data_path.split(',')
    img_loader = loaders.TargetLoader(opt.data_path, opt.target, channel=opt.channel,
                                      image_res=opt.image_res, roi_res=opt.roi_res,
                                      crop_to_roi=False, shuffle=opt.random_gen,
                                      vertical_flips=opt.random_gen, horizontal_flips=opt.random_gen,
                                      physical_depth_planes=opt.physical_depth_planes,
                                      virtual_depth_planes=opt.virtual_depth_planes,
                                      scale_vd_range=False,
                                      mod_i=opt.mod_i, mod=opt.mod, options=opt)

    for i, target in enumerate(img_loader):
        target_amp, target_mask, target_idx = target
        target_amp = target_amp.to(dev).detach()
        if target_mask is not None:
            target_mask = target_mask.to(dev).detach()
        if len(target_amp.shape) < 4:
            target_amp = target_amp.unsqueeze(0)

        print(f'  - run phase optimization for {target_idx}th image ...')
        if opt.random_gen:  # random parameters for dataset generation
            opt.num_iters, opt.init_phase_range, \
            target_range, opt.lr, opt.eval_plane_idx = utils.random_gen(num_planes=opt.num_planes,
                                                                        slm_type=opt.slm_type)
            sim_prop = prop_model.model(opt)
            target_amp *= target_range

        # initial slm phase
        init_phase = (opt.init_phase_range * (-0.5 + 1.0 * torch.rand(1, 1, *opt.slm_res))).to(dev)

        # run algorithm
        results = algorithm(init_phase, target_amp, target_mask,
                            forward_prop=sim_prop, num_iters=opt.num_iters, roi_res=opt.roi_res,
                            loss_fn=opt.loss_fn, lr=opt.lr,
                            out_path_idx=f'{opt.out_path}_{target_idx}',
                            citl=opt.citl, camera_prop=camera_prop,
                            writer=writer,
                            )

        # optimized slm phase
        final_phase = results['final_phase']

        # encoding for SLM & save it out
        phase_out = utils.phasemap_8bit(final_phase)
        if opt.random_gen:
            phase_out_path = os.path.join(out_path, f'{target_idx}_{opt.num_iters}.png')
        else:
            phase_out_path = os.path.join(out_path, f'{target_idx}_{opt.eval_plane_idx}.png')
        imageio.imwrite(phase_out_path, phase_out)

if __name__ == "__main__":
    main()
