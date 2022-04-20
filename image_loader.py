import os
import math
import skimage.io
from imageio import imread
from skimage.transform import resize
from torchvision.transforms.functional import resize as resize_tensor
import cv2
import random
import json
import numpy as np
import h5py
import torch
import utils

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(
        width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def get_matlab_filenames(dir, focuses=None):
    """Returns all files in the input directory dir that are images"""
    image_types = ('mat')
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = (os.path.splitext(f)[1] for f in files)
        if focuses is not None:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types and int(os.path.splitext(f)[0].split('_')[-1]) in focuses]
        else:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types]
        return images
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            images_in_folder = [os.path.join(folder, f)
                                for e, f in zip(exts, files)
                                if e[1:] in image_types]
            images = [*images, *images_in_folder]

        return images


def get_image_filenames(dir, focuses=None):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif', 'exr', 'dpt', 'hdf5')
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = (os.path.splitext(f)[1] for f in files)
        if focuses is not None:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types and int(os.path.splitext(f)[0].split('_')[-1]) in focuses]
        else:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types]
        return images
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            images_in_folder = [os.path.join(folder, f)
                                for e, f in zip(exts, files)
                                if e[1:] in image_types]
            images = [*images, *images_in_folder]

        return images


def resize_keep_aspect(image, target_res, pad=False, lf=False, pytorch=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False, lf=lf)

    # switch to numpy channel dim convention, resize, switch back
    if lf or pytorch:
        image = resize_tensor(image, target_res)
        return image
    else:
        image = np.transpose(image, axes=(1, 2, 0))
        image = resize(image, target_res, mode='reflect')
        return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res, pytorch=False):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=pytorch, stacked_complex=False),
                            target_res, pytorch=pytorch, stacked_complex=False)


def get_folder_names(folder):
    """Returns all files in the input directory dir that are images"""
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


class PairsLoader(torch.utils.data.IterableDataset):
    """Loads (phase, captured) tuples for forward model training

    Class initialization parameters
    -------------------------------

    :param data_path:
    :param plane_idxs:
    :param batch_size:
    :param image_res:
    :param shuffle:
    :param avg_energy_ratio:
    :param slm_type:


    """

    def __init__(self, data_path, plane_idxs=None, batch_size=1,
                 image_res=(800, 1280), shuffle=True,
                 avg_energy_ratio=None, slm_type='leto'):
        """

        """
        print(data_path)
        if isinstance(data_path, str):
            if not os.path.isdir(data_path):
                raise NotADirectoryError(f'Data folder: {data_path}')
            self.phase_path = os.path.join(data_path, 'phase')
            self.captured_path = os.path.join(data_path, 'captured')
        elif isinstance(data_path, list):
            self.phase_path = [os.path.join(path, 'phase') for path in data_path]
            self.captured_path = [os.path.join(path, 'captured') for path in data_path]

        self.all_plane_idxs = plane_idxs
        self.avg_energy_ratio = avg_energy_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.slm_type = slm_type.lower()
        self.im_names = get_image_filenames(self.phase_path)
        self.im_names.sort()

        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        self.order = list(self.order)

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __len__(self):
        return len(self.im_names)

    def __next__(self):
        if self.ind < len(self.order):
            phase_idx = self.order[self.ind]

            self.ind += 1
            return self.load_pair(phase_idx[0])
        else:
            raise StopIteration

    def load_pair(self, filenum):
        phase_path = self.im_names[filenum]
        captured_path = os.path.splitext(os.path.dirname(phase_path))[0]
        captured_path = os.path.splitext(os.path.dirname(captured_path))[0]
        captured_path = os.path.join(captured_path, 'captured')

        # load phase
        phase_im_enc = imread(phase_path)
        im = (1 - phase_im_enc / np.iinfo(np.uint8).max) * 2 * np.pi - np.pi
        phase_im = torch.tensor(im, dtype=torch.float32).unsqueeze(0)

        _, captured_filename = os.path.split(os.path.splitext(self.im_names[filenum])[0])
        idx = captured_filename.split('/')[-1]

        # load focal stack
        captured_amps = []
        for plane_idx in self.all_plane_idxs:
            captured_filename = os.path.join(captured_path, f'{idx}_{plane_idx}.png')
            captured_intensity = utils.im2float(skimage.io.imread(captured_filename))
            captured_intensity = torch.tensor(captured_intensity, dtype=torch.float32)
            if self.avg_energy_ratio is not None:
                captured_intensity /= self.avg_energy_ratio[plane_idx]  # energy compensation;
            captured_amp = torch.sqrt(captured_intensity)
            captured_amps.append(captured_amp)
        captured_amps = torch.stack(captured_amps, 0)

        return phase_im, captured_amps


class TargetLoader(torch.utils.data.IterableDataset):
    """Loads target amp/mask tuples for phase optimization

    Class initialization parameters
    -------------------------------
    :param data_path:
    :param target_type:
    :param channel:
    :param image_res:
    :param roi_res:
    :param crop_to_roi:
    :param shuffle:
    :param vertical_flips:
    :param horizontal_flips:
    :param virtual_depth_planes:
    :param scale_vd_range:

    """

    def __init__(self, data_path, target_type, channel=None,
                 image_res=(800, 1280), roi_res=(700, 1190),
                 crop_to_roi=False, shuffle=False,
                 vertical_flips=False, horizontal_flips=False,
                 physical_depth_planes=None,
                 virtual_depth_planes=None, scale_vd_range=True,
                 mod_i=None, mod=None, options=None):
        """ initialization """
        if isinstance(data_path, str) and not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')

        self.data_path = data_path
        self.target_type = target_type.lower()
        self.channel = channel
        self.roi_res = roi_res
        self.crop_to_roi = crop_to_roi
        self.image_res = image_res
        self.shuffle = shuffle
        self.physical_depth_planes = physical_depth_planes
        self.virtual_depth_planes = virtual_depth_planes
        self.vd_min = 0.01
        self.vd_max = max(self.virtual_depth_planes)
        self.scale_vd_range = scale_vd_range
        self.options = options

        self.augmentations = []
        if vertical_flips:
            self.augmentations.append(self.augment_vert)
        if horizontal_flips:
            self.augmentations.append(self.augment_horz)

        # store the possible states for enumerating augmentations
        self.augmentation_states = [fn() for fn in self.augmentations]

        if target_type in ('2d', 'rgb'):
            self.im_names = get_image_filenames(self.data_path)
            self.im_names.sort()
        elif target_type in ('2.5d', 'rgbd'):
            self.im_names = get_image_filenames(os.path.join(self.data_path, 'rgb'))
            self.depth_names = get_image_filenames(os.path.join(self.data_path, 'depth'))

            self.im_names.sort()
            self.depth_names.sort()

        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        for aug_type in self.augmentations:
            states = aug_type()  # empty call gets possible states
            # augment existing list with new entry to states tuple
            self.order = ((*prev_states, s)
                          for prev_states in self.order
                          for s in states)
        self.order = list(self.order)

        if mod_i is not None:
            new_order = []
            for m, o in enumerate(self.order):
                if m % mod == mod_i:
                    new_order.append(o)
            self.order = new_order

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __len__(self):
        return len(self.order)

    def __next__(self):
        if self.ind < len(self.order):
            img_idx = self.order[self.ind]

            self.ind += 1
            if self.target_type in ('2d', 'rgb'):
                return self.load_image(*img_idx)
            if self.target_type in ('2.5d', 'rgbd'):
                return self.load_image_mask(*img_idx)
        else:
            raise StopIteration

    def load_image(self, filenum, *augmentation_states):
        im = imread(self.im_names[filenum])

        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  # augment channels for gray images

        if self.channel is None:
            im = im[..., :3]  # remove alpha channel, if any
        else:
            # select channel while keeping dims
            im = im[..., self.channel, np.newaxis]

        im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

        # linearize intensity and convert to amplitude
        im = utils.srgb_gamma2lin(im)
        im = np.sqrt(im)  # to amplitude

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))

        # apply data augmentation
        for fn, state in zip(self.augmentations, augmentation_states):
            im = fn(im, state)

        # normalize resolution
        if self.crop_to_roi:
            im = pad_crop_to_res(im, self.roi_res)
        else:
            im = resize_keep_aspect(im, self.roi_res)
        im = pad_crop_to_res(im, self.image_res)
        
        path = os.path.splitext(self.im_names[filenum])[0]

        return (torch.from_numpy(im).float(),
                None,
                os.path.split(path)[1].split('_')[-1])

    def load_depth(self, filenum, *augmentation_states):
        depth_path = self.depth_names[filenum]
        if 'exr' in depth_path:
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        elif 'dpt' in depth_path:
            dist = depth_read(depth_path)
            depth = np.nan_to_num(dist, 100)  # NaN to inf
        elif 'hdf5' in depth_path:
            # Depth (in m)
            with h5py.File(depth_path, 'r') as f:
                dist = np.array(f['dataset'][:], dtype=np.float32)
                depth = np.nan_to_num(dist, 100)  # NaN to inf
        else:
            depth = imread(depth_path)

        depth = utils.im2float(depth, dtype=np.float64)  # convert to double, max 1

        if 'bbb' in depth_path:
            depth *= 6  # this gives us decent depth distribution with 120mm eyepiece setting.
        elif 'sintel' in depth_path or 'dpt' in depth_path:
            depth /= 2.5  # this gives us decent depth distribution with 120mm eyepiece setting.
        if len(depth.shape) > 2 and depth.shape[-1] > 1:
            depth = depth[..., 1]

        if not 'eth' in depth_path.lower():
            depth = 1 / (depth + 1e-20)  # meter to diopter conversion

        # apply data augmentation
        for fn, state in zip(self.augmentations, augmentation_states):
            depth = fn(depth, *state)

        depth = torch.from_numpy(depth.copy()).float().unsqueeze(0)
        # normalize resolution
        depth.unsqueeze_(0)
        if self.crop_to_roi:
            depth = pad_crop_to_res(depth, self.roi_res, pytorch=True)
        else:
            depth = resize_keep_aspect(depth, self.roi_res, pytorch=True)
        depth = pad_crop_to_res(depth, self.image_res, pytorch=True)

        # perform scaling in meters
        if self.scale_vd_range:
            depth = depth - depth.min()
            depth = (depth / depth.max()) * (self.vd_max - self.vd_min)
            depth = depth + self.vd_min

        # check nans
        if (depth.isnan().any()):
            print("Found Nans in target depth!")
            min_substitute = self.vd_min * torch.ones_like(depth)
            depth = torch.where(depth.isnan(), min_substitute, depth)

        path = os.path.splitext(self.depth_names[filenum])[0]

        return (depth.float(),
                None,
                os.path.split(path)[1].split('_')[-1])

    def load_image_mask(self, filenum, *augmentation_states):
        img_none_idx = self.load_image(filenum, *augmentation_states)
        depth_none_idx = self.load_depth(filenum, *augmentation_states)
        mask = utils.decompose_depthmap(depth_none_idx[0], self.virtual_depth_planes)
        return (img_none_idx[0].unsqueeze(0), mask, img_none_idx[-1])

    def augment_vert(self, image=None, flip=False):
        """ augment data with vertical flip """
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1, :]
        return image

    def augment_horz(self, image=None, flip=False):
        """ augment data with horizontal flip """
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1]
        return image

