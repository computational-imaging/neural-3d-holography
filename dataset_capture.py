"""
capture all the phases in a folder
201203

"""

import os
import time
import concurrent
import cv2
import skimage.io
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import imageio
import configargparse

import utils
import props
import params

def save_image(raw_data, file_path, demosaickrule, warper, dev):
    raw_data = raw_data - 64
    color_cv_image = cv2.cvtColor(raw_data, demosaickrule)  # it gives float64 from uint16
    captured_intensity = utils.im2float(color_cv_image)  # float64 to float32

    # Numpy to tensor
    captured_intensity = torch.tensor(captured_intensity, dtype=torch.float32,
                                      device=dev).permute(2, 0, 1).unsqueeze(0)
    captured_intensity = torch.sum(captured_intensity, dim=1, keepdim=True)

    warped_intensity = warper(captured_intensity)
    imageio.imwrite(file_path, (np.clip(warped_intensity.squeeze().cpu().detach().numpy(), 0.0, 1.0)
                     * np.iinfo(np.uint16).max).round().astype(np.uint16))

if __name__ == '__main__':
    # parse arguments
    # Command line argument processing / Parameters
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
    params.add_parameters(p, 'eval')
    opt = p.parse_args()
    params.set_configs(opt)
    dev = torch.device('cuda')

    # hardware setup
    params_slm, params_camera, params_calib = params.hw_params(opt)
    camera_prop = props.PhysicalProp(params_slm, params_camera, params_calib).to(dev)

    data_path = opt.data_path  # pass a path of your dataset folder like f'F:/dataset/green'

    if not opt.chan_str in data_path:
        raise ValueError('Double check the color!')

    ds = ['test', 'val', 'train']
    data_paths = [os.path.join(data_path, d) for d in ds]
    for root_path in data_paths:

        # load phases
        phase_path = f'{root_path}/phase'
        captured_path = f'{root_path}/captured'
        utils.cond_mkdir(captured_path)
        names = os.listdir(f'{phase_path}')

        # run multiple thread for fast capture
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for ii, full_name in enumerate(names):
                t0 = time.perf_counter()

                filename = f'{phase_path}/{full_name}'
                phase_img = skimage.io.imread(filename)
                raw_uint16_data = camera_prop.capture_uint16(phase_img)  # display & retrieve buffer

                out_full_path = os.path.join(captured_path, f'{full_name[:-4]}_{opt.plane_idx}.png')
                executor.submit(save_image,
                                raw_uint16_data,
                                out_full_path,
                                camera_prop.camera.demosaick_rule,
                                camera_prop.warper, dev)

                print(f'{out_full_path}: '
                      f'{1 / (time.perf_counter() - t0):.2f}Hz')

                if ii % 500 == 0 and ii > 0:
                    time.sleep(10.0)

camera_prop.disconnect()