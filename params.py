"""
Default parameter settings for SLMs as well as laser/sensors

"""
import datetime
import math
import sys
import numpy as np
import utils
import torch.nn as nn
if sys.platform == 'win32':
    import serial

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

def str2bool(v):
    """ Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


class PMap(dict):
    # use it for parameters
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def add_parameters(p, mode='train'):
    p.add_argument('--channel', type=int, default=None, help='Red:0, green:1, blue:2')
    p.add_argument('--method', type=str, default='SGD', help='Type of algorithm, GS/SGD/DPAC/HOLONET/UNET')
    p.add_argument('--slm_type', type=str, default='leto', help='leto/pluto ...')
    p.add_argument('--sensor_type', type=str, default='2k', help='sensor type')
    p.add_argument('--laser_type', type=str, default='new', help='old, new_laser, sLED, ...')
    p.add_argument('--setup_type', type=str, default='sigasia2021_vr', help='VR or AR')
    p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model, ASM/NH/NH3D')
    p.add_argument('--out_path', type=str, default='./results',
                   help='Directory for output')
    p.add_argument('--citl', type=str2bool, default=False,
                   help='If True, run camera-in-the-loop')
    p.add_argument('--mod_i', type=int, default=None,
                   help='If not None, say K, pick every K target images from the target loader')
    p.add_argument('--mod', type=int, default=None,
                   help='If not None, say K, pick every K target images from the target loader')
    p.add_argument('--data_path', type=str, default='/mount/workspace/data/NH3D/',
                   help='Directory for input')
    p.add_argument('--exp', type=str, default='', help='Name of experiment')
    p.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    p.add_argument('--num_iters', type=int, default=1000, help='Number of iterations (GS, SGD)')
    p.add_argument('--prop_dist', type=float, default=None, help='propagation distance from SLM to midplane')
    p.add_argument('--F_aperture', type=float, default=1.0, help='Fourier filter size')
    p.add_argument('--eyepiece', type=float, default=0.12, help='eyepiece focal length')
    p.add_argument('--full_roi', type=str2bool, default=False,
                   help='If True, force ROI to SLM resolution')
    p.add_argument('--target', type=str, default='rgbd',
                   help='Type of target:' '{2d, rgb} or {2.5d, rgbd}')
    p.add_argument('--show_preview', type=str2bool, default=False,
                   help='If true, show the preview for homography calibration')
    p.add_argument('--random_gen', type=str2bool, default=False,
                   help='If true, randomize a few parameters for phase dataset generation')
    p.add_argument('--eval_plane_idx', type=int, default=None,
                   help='When evaluating 2d, choose the index of the plane')
    p.add_argument('--init_phase_range', type=float, default=1.0,
                   help='Phase sampling range for initializaation')

    if mode in ('train', 'eval'):
        p.add_argument('--num_epochs', type=int, default=350, help='')
        p.add_argument('--batch_size', type=int, default=1, help='')
        p.add_argument('--prop_model_path', type=str, default=None, help='Path to checkpoints')
        p.add_argument('--predefined_model', type=str, default=None, help='string for predefined model'
                                                                          'nh, nh3d')
        p.add_argument('--num_downs_slm', type=int, default=5, help='')
        p.add_argument('--num_feats_slm_min', type=int, default=32, help='')
        p.add_argument('--num_feats_slm_max', type=int, default=128, help='')
        p.add_argument('--num_downs_target', type=int, default=5, help='')
        p.add_argument('--num_feats_target_min', type=int, default=32, help='')
        p.add_argument('--num_feats_target_max', type=int, default=128, help='')
        p.add_argument('--slm_coord', type=str, default='rect', help='coordinates to represent a complex-valued field.'
                                                                 'rect(real+imag) or polar(amp+phase)')
        p.add_argument('--target_coord', type=str, default='rect', help='coordinates to represent a complex-valued field.'
                                                                 'rect(real+imag) or polar(amp+phase)')
        p.add_argument('--norm', type=str, default='instance', help='normalization layer')
        p.add_argument('--loss_func', type=str, default='l1', help='l1 or l2')
        p.add_argument('--energy_compensation', type=str2bool, default=True, help='adjust intensities '
                                                                                  'with avg intensity of training set')
        p.add_argument('--num_train_planes', type=int, default=7, help='number of planes fed to models')

    return p


def set_configs(opt):
    """
    set or replace parameters with pre-defined parameters with string inputs
    """

    # hardware setup
    optics_config(opt.setup_type, opt)  # prop_dist, etc ...
    laser_config(opt.laser_type, opt)  # Our Old FISBA Laser, New, SLED, LED, ...
    slm_config(opt.slm_type, opt)  # Holoeye
    sensor_config(opt.sensor_type, opt)  # our sensor

    # set predefined model parameters
    forward_model_config(opt.prop_model, opt)

    # wavelength, propagation distance (from SLM to midplane)
    if opt.channel is None:
        opt.chan_str = 'rgb'
        opt.prop_dist = opt.prop_dists_rgb
        opt.wavelength = opt.wavelengths
    else:
        opt.chan_str = ('red', 'green', 'blue')[opt.channel]
        if opt.prop_dist is None:
            opt.prop_dist = opt.prop_dists_rgb[opt.channel][opt.mid_idx]  # prop dist from SLM plane to target plane
            opt.prop_dist_green = opt.prop_dists_rgb[opt.channel][1]
        else:
            opt.prop_dist_green = opt.prop_dist
        opt.wavelength = opt.wavelengths[opt.channel]  # wavelength of each color

    # propagation distances from the wavefront recording plane
    opt.prop_dists_from_wrp = [p - opt.prop_dist for p in opt.prop_dists_rgb[opt.channel]]
    opt.physical_depth_planes = [p - opt.prop_dist_green for p in opt.prop_dists_physical]
    opt.virtual_depth_planes = utils.prop_dist_to_diopter(opt.physical_depth_planes,
                                                          opt.eyepiece,
                                                          opt.physical_depth_planes[0])
    opt.num_planes = len(opt.prop_dists_from_wrp)
    opt.all_plane_idxs = range(opt.num_planes)

    # force ROI to that of SLM
    if opt.full_roi:
        opt.roi_res = opt.slm_res

    ################
    # Model Training
    # compensate the brightness difference per plane (for model training)
    if opt.energy_compensation:
        opt.avg_energy_ratio = opt.avg_energy_ratio_rgb[opt.channel]
    else:
        opt.avg_energy_ratio = None

    # loss functions (for model training)
    opt.loss_train = None
    opt.loss_fn = None
    if opt.loss_func.lower() in ('l2', 'mse'):
        opt.loss_train = nn.functional.mse_loss
        opt.loss_fn = nn.functional.mse_loss
    elif opt.loss_func.lower() == 'l1':
        opt.loss_train = nn.functional.l1_loss
        opt.loss_fn = nn.functional.l1_loss

    # plane idxs (for model training)
    opt.plane_idxs = {}
    opt.plane_idxs['all'] = opt.all_plane_idxs
    opt.plane_idxs['train'] = opt.training_plane_idxs
    opt.plane_idxs['validation'] = opt.training_plane_idxs
    opt.plane_idxs['test'] = opt.training_plane_idxs
    opt.plane_idxs['heldout'] = opt.heldout_plane_idxs
    
    admm_opt = None
    if 'admm' in opt.method:
        admm_opt = {'num_iters_inner': 50,
                    'rho': 0.01,
                    'alpha': 1.0,
                    'gamma': 0.1,
                    'varying-penalty': True,
                    'mu': 10.0,
                    'tau_incr': 2.0,
                    'tau_decr': 2.0}
    
    return opt


def run_id(opt):
    id_str = f'{opt.exp}_{opt.chan_str}_{opt.prop_model}_{opt.lr}_{opt.num_iters}'
    id_str = f'{opt.chan_str}'
    return id_str

def run_id_training(opt):
    id_str = f'{opt.exp}_{opt.chan_str}-' \
             f'slm{opt.num_downs_slm}-{opt.num_feats_slm_min}-{opt.num_feats_slm_max}_' \
             f'tg{opt.num_downs_target}-{opt.num_feats_target_min}-{opt.num_feats_target_max}_' \
             f'{opt.slm_coord}{opt.target_coord}_{opt.loss_func}_{opt.num_train_planes}pls_' \
             f'bs{opt.batch_size}'
    cur_time = datetime.datetime.now().strftime("%d-%H%M")
    id_str = f'{cur_time}_{id_str}'

    return id_str


def hw_params(opt):
    """ Default setting for hardware. Please replace and adjust parameters for your own setup. """
    params_slm = PMap()
    params_slm.settle_time = 0.3
    params_slm.monitor_num = 2
    params_slm.slm_type = opt.slm_type

    params_camera = PMap()
    params_camera.img_size_native = (3000, 4096)  # 4k sensor native
    params_camera.ser = None #serial.Serial('COM5', 9600, timeout=0.5)

    params_calib = PMap()
    params_calib.show_preview = opt.show_preview
    params_calib.range_y = slice(0, params_camera.img_size_native[0])
    params_calib.range_x = slice(0, params_camera.img_size_native[1])
    params_calib.num_circles = (13, 22)
    params_calib.spacing_size = [int(roi / (num_circs - 1))
                                 for roi, num_circs in zip(opt.roi_res, params_calib.num_circles)]
    params_calib.pad_pixels = [int(slm - roi) // 2 for slm, roi in zip(opt.slm_res, opt.roi_res)]
    params_calib.quadratic = True
    params_calib.phase_path = f'./calibration/{opt.chan_str}/1_{opt.eval_plane_idx}.png'
    params_calib.img_size_native = params_camera.img_size_native

    return params_slm, params_camera, params_calib


def slm_config(slm_type, opt):
    """ Setting for specific SLM. """
    if slm_type.lower() in ('leto'):
        opt.feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
        opt.slm_res = (1080, 1920)  # resolution of SLM
        opt.image_res = opt.slm_res
    elif slm_type.lower() in ('pluto'):
        opt.feature_size = (8.0 * um, 8.0 * um)  # SLM pitch
        opt.slm_res = (1080, 1920)  # resolution of SLM
        opt.image_res = opt.slm_res


def laser_config(laser_type, opt):
    """ Setting for specific laser. """
    if 'new' in laser_type.lower():
        opt.wavelengths = (636.17 * nm, 518.48 * nm, 442.03 * nm)  # wavelength of each color
    elif 'ar' in laser_type.lower():
        opt.wavelengths = (532 * nm, 532 * nm, 532 * nm)
    else:
        opt.wavelengths = (636.4 * nm, 517.7 * nm, 440.8 * nm)


def sensor_config(sensor_type, opt):
    return opt


def optics_config(setup_type, opt):
    """ Setting for specific setup (prop dists, filter, training plane index ...) """
    if setup_type in ('sigasia2021_vr'):
        opt.laser_type = 'old'
        opt.slm_type = 'leto'
        opt.prop_dists_rgb = [[0.0, 1*mm, 2*mm, 3*mm, 4.4*mm, 5.7*mm, 7.0*mm, 8.1*mm],
                              [0.0, 1.2*mm, 2.0*mm, 3.4*mm, 4.4*mm, 5.7*mm, 7.2*mm, 8.2*mm],
                              [0.0, 1*mm, 2.1*mm, 3.2*mm, 4.3*mm, 5.5*mm, 7.2*mm, 8.2*mm]]

        opt.avg_energy_ratio_rgb = [[1.0000, 1.0407, 1.0870, 1.1216, 1.1568, 1.2091, 1.2589, 1.2924],
                                    [1.0000, 1.0409, 1.0869, 1.1226, 1.1540, 1.2107, 1.2602, 1.2958],
                                    [1.0000, 1.0409, 1.0869, 1.1226, 1.1540, 1.2107, 1.2602, 1.2958]]
        opt.prop_dists_physical = opt.prop_dists_rgb[1]
        opt.F_aperture = 0.5
        opt.roi_res = (960, 1680)  # regions of interest (to penalize for SGD)
        opt.training_plane_idxs = [0, 1, 3, 4, 5, 6, 7]
        opt.heldout_plane_idxs = [2]
        opt.mid_idx = 4  # intermediate plane as 1.5D
    elif setup_type in ('sigasia2021_ar'):
        opt.laser_type = 'ar_green_only'
        opt.slm_type = 'pluto'
        opt.prop_dists_rgb = [[9.9*mm, 10.3*mm, 11.8*mm, 13.3*mm],
                              [9.9*mm, 10.3*mm, 11.8*mm, 13.3*mm],
                              [9.9*mm, 10.3*mm, 11.8*mm, 13.3*mm]]
        opt.prop_dists_physical = opt.prop_dists_rgb[1]
        opt.F_aperture = 0.5
        opt.roi_res = (768, 1536)  # regions of interest (to penalize for SGD)
        opt.training_plane_idxs = [0, 1, 2]
        opt.heldout_plane_idxs = []


def forward_model_config(model_type, opt):
    # setting for specific model that is predefined.
    if model_type is not None:
        print(f'  - changing model parameters for {model_type}')
        if model_type.lower() in ('cnnpropcnn', 'nh3d'):
            opt.num_downs_slm = 8
            opt.num_feats_slm_min = 32
            opt.num_feats_slm_max = 512
            opt.num_downs_target = 5
            opt.num_feats_target_min = 8
            opt.num_feats_target_max = 128
        elif model_type.lower() == 'hil':
            opt.num_downs_slm = 0
            opt.num_feats_slm_min = 0
            opt.num_feats_slm_max = 0
            opt.num_downs_target = 8
            opt.num_feats_target_min = 32
            opt.num_feats_target_max = 512
            opt.target_coord = 'amp'
        elif model_type.lower() == 'cnnprop':
            opt.num_downs_slm = 8
            opt.num_feats_slm_min = 32
            opt.num_feats_slm_max = 512
            opt.num_downs_target = 0
            opt.num_feats_target_min = 0
            opt.num_feats_target_max = 0
        elif model_type.lower() == 'propcnn':
            opt.num_downs_slm = 0
            opt.num_feats_slm_min = 0
            opt.num_feats_slm_max = 0
            opt.num_downs_target = 8
            opt.num_feats_target_min = 32
            opt.num_feats_target_max = 512