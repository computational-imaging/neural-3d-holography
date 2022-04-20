"""
This is the script containing the calibration module, basically calculating homography matrix.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
S. Choi, M. Gopakumar, Y. Peng, J. Kim, G. Wetzstein. Neural 3D holography: Learning accurate wave propagation models for 3D holographic virtual and augmented reality displays. ACM TOG (SIGGRAPH Asia), 2021.
"""

import PyCapture2
import cv2
import numpy as np
import time
import utils


def callback_captured(image):
    print(image.getData())


class CameraCapture:
    def __init__(self, params):
        self.bus = PyCapture2.BusManager()
        num_cams = self.bus.getNumOfCameras()
        if not num_cams:
            exit()
        # self.demosaick_rule = cv2.COLOR_BAYER_RG2BGR
        self.demosaick_rule = cv2.COLOR_BAYER_GR2RGB  # GBRG to RGB
        self.params = params

    def connect(self, i, trigger=False):
        uid = self.bus.getCameraFromIndex(i)
        self.camera_device = PyCapture2.Camera()
        self.camera_device.connect(uid)
        self.camera_device.setConfiguration(highPerformanceRetrieveBuffer=True)
        self.camera_device.setConfiguration(numBuffers=1000)
        config = self.camera_device.getConfiguration()
        self.toggle_embedded_timestamp(True)

        if trigger:
            trigger_mode = self.camera_device.getTriggerMode()
            trigger_mode.onOff = True
            trigger_mode.mode = 0
            trigger_mode.parameter = 0
            trigger_mode.source = 3  # Using software trigger
            self.camera_device.setTriggerMode(trigger_mode)
        else:
            trigger_mode = self.camera_device.getTriggerMode()
            trigger_mode.onOff = False
            trigger_mode.mode = 0
            trigger_mode.parameter = 0
            trigger_mode.source = 3  # Using software trigger
            self.camera_device.setTriggerMode(trigger_mode)

        trigger_mode = self.camera_device.getTriggerMode()
        if trigger_mode.onOff is True:
            print('    - setting trigger mode on')


    def disconnect(self):
        self.toggle_embedded_timestamp(False)
        self.camera_device.disconnect()

    def toggle_embedded_timestamp(self, enable_timestamp):
        embedded_info = self.camera_device.getEmbeddedImageInfo()
        if embedded_info.available.timestamp:
            self.camera_device.setEmbeddedImageInfo(timestamp=enable_timestamp)

    def grab_images(self, num_images_to_grab=1):
        """
        Retrieve the camera buffer and returns a list of grabbed images.

        :param num_images_to_grab: integer, default 1
        :return: a list of numpy 2d color images from the camera buffer.
        """
        self.camera_device.startCapture()
        img_list = []
        for i in range(num_images_to_grab):
            imgData = self.retrieve_buffer()
            offset = 64  # offset that inherently exist.retrieve_buffer
            imgData = imgData - offset

            color_cv_image = cv2.cvtColor(imgData, self.demosaick_rule)
            color_cv_image = utils.im2float(color_cv_image)
            img_list.append(color_cv_image.copy())

        self.camera_device.stopCapture()
        return img_list

    def grab_images_fast(self, num_images_to_grab=1):
        """
        Retrieve the camera buffer and returns a grabbed image

        :param num_images_to_grab: integer, default 1
        :return: a list of numpy 2d color images from the camera buffer.
        """
        imgData = self.retrieve_buffer()
        offset = 64  # offset that inherently exist.
        imgData = imgData - offset

        color_cv_image = cv2.cvtColor(imgData, self.demosaick_rule)
        color_cv_image = utils.im2float(color_cv_image)
        color_img = color_cv_image
        return color_img

    def retrieve_buffer(self):
        try:
            img = self.camera_device.retrieveBuffer()
        except PyCapture2.Fc2error as fc2Err:
            raise fc2Err

        imgData = img.getData()

        # when using raw8 from the PG sensor
        # cv_image = np.array(img.getData(), dtype="uint8").reshape((img.getRows(), img.getCols()))

        # when using raw16 from the PG sensor - concat 2 8bits in a row
        imgData.dtype = np.uint16
        imgData = imgData.reshape(img.getRows(), img.getCols())
        return imgData.copy()

    def start_capture(self):
        # these two were previously inside the grab_images func, and can be clarified outside the loop
        self.camera_device.startCapture()

    def stop_capture(self):
        self.camera_device.stopCapture()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        self._params = p