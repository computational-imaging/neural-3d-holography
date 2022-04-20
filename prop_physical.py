"""
Propagation happening on the setup

"""

import torch
import torch.nn as nn
import utils
import time
import cv2
import numpy as np
import imageio

import sys
if sys.platform == 'win32':
    import slmpy
    import hw.camera_capture_module as cam
    import hw.calibration_module as calibration


class PhysicalProp(nn.Module):
    """ A module for physical propagation,
    forward pass displays gets SLM pattern as an input and display the pattern on the physical setup,
    and capture the diffraction image at the target plane,
    and then return warped image using pre-calibrated homography from instantiation.

    Class initialization parameters
    -------------------------------
    :param params_slm: a set of parameters for the SLM.
    :param params_camera: a set of parameters for the camera sensor.
    :param params_calib: a set of parameters for homography calibration.

    Usage
    -----
    Functions as a pytorch module:

    >>> camera_prop = PhysicalProp(...)
    >>> captured_amp = camera_prop(slm_phase)

    slm_phase: phase at the SLM plane, with dimensions [batch, 1, height, width]
    captured_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]

    """
    def __init__(self, params_slm, params_camera, params_calib=None):
        super(PhysicalProp, self).__init__()

        # 1. Connect Camera
        self.camera = cam.CameraCapture(params_camera)
        self.camera.connect(0)  # specify the camera to use, 0 for main cam, 1 for the second cam
        self.camera.start_capture()

        # 2. Connect SLM
        self.slm = slmpy.SLMdisplay(isImageLock=True, monitor=params_slm.monitor_num)
        self.params_slm = params_slm

        # 3. Calibrate hardware using homography
        if params_calib is not None:
            self.warper = calibration.Warper(params_calib)
            self.calibrate(params_calib.phase_path, params_calib.show_preview)
        else:
            self.warper = None

    def calibrate(self, phase_path, show_preview=False):
        """

        :param phase_path:
        :param show_preview:
        :return:
        """
        print('  -- Calibrating ...')
        phase_img = imageio.imread(phase_path)
        self.slm.updateArray(phase_img)
        time.sleep(self.params_slm.settle_time)
        captured_img = self.camera.grab_images_fast(5)  # capture 5-10 images for averaging
        calib_success = self.warper.calibrate(captured_img, show_preview)
        if calib_success:
            print('  -- Calibration succeeded!...')
        else:
            raise ValueError('  -- Calibration failed')

    def forward(self, slm_phase):
        """

        :param slm_phase:
        :return:
        """
        raw_intensity = self.capture_linear_intensity(slm_phase)  # grayscale raw16 intensity image
        warped_intensity = self.warper(raw_intensity)  # apply homography
        return warped_intensity.sqrt()  # return amplitude

    def capture_linear_intensity(self, slm_phase):
        """
        display a phase pattern on the SLM and capture a generated holographic image with the sensor.

        :param slm_phase:
        :return:
        """
        raw_uint16_data = self.capture_uint16(slm_phase)  # display & retrieve buffer
        captured_intensity = self.process_raw_data(raw_uint16_data)  # demosaick & sum up
        return captured_intensity

    def capture_uint16(self, slm_phase):
        """
        gets phase pattern(s) and display it on the SLM, and then send a signal to board (wait next clock from SLM).
        Right after hearing back from the SLM, it sends another signal to PC so that PC retreives the camera buffer.

        :param slm_phase:
        :return:
        """
        if torch.is_tensor(slm_phase):
            slm_phase_encoded = utils.phasemap_8bit(slm_phase)
        else:
            slm_phase_encoded = slm_phase
        self.slm.updateArray(slm_phase_encoded)

        if self.camera.params.ser is not None:
            self.camera.params.ser.write(f'D'.encode())

            # TODO: make the following in a separate function.
            # Wait until receiving signal from arduino
            incoming_byte = self.camera.params.ser.inWaiting()
            t0 = time.perf_counter()
            while True:
                received = self.camera.params.ser.read(incoming_byte).decode('UTF-8')
                if received != 'C':
                    incoming_byte = self.camera.params.ser.inWaiting()
                    if time.perf_counter() - t0 > 2.0:
                        break
                else:
                    break
        else:
            time.sleep(self.params_slm.settle_time)

        raw_data_from_buffer = self.camera.retrieve_buffer()
        return raw_data_from_buffer

    def process_raw_data(self, raw_data):
        """
        gets raw data from the camera buffer, and demosaick it

        :param raw_data:
        :return:
        """
        raw_data = raw_data - 64
        color_cv_image = cv2.cvtColor(raw_data, self.camera.demosaick_rule)  # it gives float64 from uint16 -- double check it
        captured_intensity = utils.im2float(color_cv_image)  # float64 to float32

        # Numpy to tensor
        captured_intensity = torch.tensor(captured_intensity, dtype=torch.float32,
                                          device=self.dev).permute(2, 0, 1).unsqueeze(0)
        captured_intensity = torch.sum(captured_intensity, dim=1, keepdim=True)
        return captured_intensity

    def disconnect(self):
        self.camera.stop_capture()
        self.camera.disconnect()
        self.slm.close()

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.warper is not None:
            slf.warper = slf.warper.to(*args, **kwargs)
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf