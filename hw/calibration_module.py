"""
This is the script containing the calibration module, basically calculating homography matrix.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
S. Choi, M. Gopakumar, Y. Peng, J. Kim, G. Wetzstein. Neural 3D holography: Learning accurate wave propagation models for 3D holographic virtual and augmented reality displays. ACM TOG (SIGGRAPH Asia), 2021.
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import cv2
import skimage.transform as transform
import time
import datetime
from scipy.io import savemat
from scipy.ndimage import map_coordinates
import torch
import torch.nn.functional as F
import torch.nn as nn

def id(x):
    return x

def circle_detect(captured_img, num_circles, spacing, pad_pixels=(0., 0.), show_preview=True, quadratic=False):
    """
    Detects the circle of a circle board pattern

    :param captured_img: captured image
    :param num_circles: a tuple of integers, (num_circle_x, num_circle_y)
    :param spacing: a tuple of integers, in pixels, (space between circles in x, space btw circs in y direction)
    :param show_preview: boolean, default True
    :param pad_pixels: coordinate of the left top corner of warped image.
                       Assuming pad this amount of pixels on the other side.
    :return: a tuple, (found_dots, H)
             found_dots: boolean, indicating success of calibration
             H: a 3x3 homography matrix (numpy)
    """

    # Binarization
    # org_copy = org.copy() # Otherwise, we write on the original image!
    img = (captured_img.copy() * 255).astype(np.uint8)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img[...,0].mean())
    # print(img[...,1].mean())
    # print(img[...,2].mean())

    #img = cv2.medianBlur(img, 31)
    img = cv2.medianBlur(img, 55)  # Red 71
    # img = cv2.medianBlur(img, 5)  #210104
    img_gray = img.copy()

    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 121, 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 0)

    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 117, 0)  # Red 127
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = 255 - img

    # Blob detection
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.filterByColor = True
    params.minThreshold = 150
    params.minThreshold = 121
    # params.minThreshold = 121

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 150
    # params.minArea = 80  # Red 120
    # params.minArea = 30  # 210104
    # params.maxArea = 100  # 210104

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.85
    params.minCircularity = 0.60

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87
    # params.minConvexity = 0.80

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    # Detecting keypoints
    # this is redundant for what comes next, but gives us access to the detected dots for debug
    keypoints = detector.detect(img)
    found_dots, centers = cv2.findCirclesGrid(img, (num_circles[1], num_circles[0]),
                                              blobDetector=detector, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

    # Drawing the keypoints
    cv2.drawChessboardCorners(captured_img, num_circles, centers, found_dots)
    img_gray = cv2.drawKeypoints(img_gray, keypoints, np.array([]), (0, 255, 0),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Find transformation
    H = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]], dtype=np.float32)
    if found_dots:
        # Generate reference points to compute the homography
        ref_pts = np.zeros((num_circles[0] * num_circles[1], 1, 2), np.float32)
        pos = 0
        for j in range(0, num_circles[0]):
            for i in range(0, num_circles[1]):
                ref_pts[pos, 0, :] = spacing * np.array([i, j]) + np.array([pad_pixels[1], pad_pixels[0]])

                pos += 1


        H, mask = cv2.findHomography(centers, ref_pts, cv2.RANSAC, 1)

        ref_pts = ref_pts.reshape(num_circles[0] * num_circles[1], 2)
        centers = np.flip(centers.reshape(num_circles[0] * num_circles[1], 2), 1)


        now = datetime.datetime.now()
        mdic = {"centers": centers, 'H': H}
        # savemat(f'F:/2021/centers_coords/centers_{now.strftime("%m%d_%H%M%S")}.mat', mdic)
        dsize = [int((num_circs - 1) * space + 2 * pad_pixs)
                     for num_circs, space, pad_pixs in zip(num_circles, spacing, pad_pixels)    ]
        if quadratic:
            H = transform.estimate_transform('polynomial', ref_pts, centers)
            coords = transform.warp_coords(H, dsize, dtype=np.float32)  # for pytorch
        else:
            tf = transform.estimate_transform('projective', ref_pts, centers)
            coords = transform.warp_coords(tf, (800, 1280), dtype=np.float32)  # for pytorch

        if show_preview:
            dsize = [int((num_circs - 1) * space + 2 * pad_pixs)
                     for num_circs, space, pad_pixs in zip(num_circles, spacing, pad_pixels)]
            if quadratic:
                captured_img_warp = transform.warp(captured_img, H, output_shape=(dsize[0], dsize[1]))
            else:
                captured_img_warp = cv2.warpPerspective(captured_img, H, (dsize[1], dsize[0]))


    if show_preview:
        fig = plt.figure()

        ax = fig.add_subplot(223)
        ax.imshow(img_gray, cmap='gray')

        ax2 = fig.add_subplot(221)
        ax2.imshow(img, cmap='gray')

        ax3 = fig.add_subplot(222)
        ax3.imshow(captured_img, cmap='gray')

        if found_dots:
            ax4 = fig.add_subplot(224)
            ax4.imshow(captured_img_warp, cmap='gray')

        plt.show()

    return found_dots, H, coords


class Warper(nn.Module):
    def __init__(self, params_calib):
        super(Warper, self).__init__()
        self.num_circles = params_calib.num_circles
        self.spacing_size = params_calib.spacing_size
        self.pad_pixels = params_calib.pad_pixels
        self.quadratic = params_calib.quadratic
        self.img_size_native = params_calib.img_size_native  # get this from image
        self.h_transform = np.array([[1., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 1.]])
        self.range_x = params_calib.range_x  # slice
        self.range_y = params_calib.range_y  # slice


    def calibrate(self, img, show_preview=True):
        img_masked = np.zeros_like(img)
        img_masked[self.range_y, self.range_x, ...] = img[self.range_y, self.range_x, ...]

        found_corners, self.h_transform, self.coords = circle_detect(img_masked, self.num_circles,
                                                                     self.spacing_size, self.pad_pixels, show_preview,
                                                                     quadratic=self.quadratic)
        if not self.coords is None:
            self.coords_tensor = torch.tensor(np.transpose(self.coords, (1, 2, 0)),
                                              dtype=torch.float32).unsqueeze(0)

            # normalize it into [-1, 1]
            # self.coords_tensor[..., 0] = self.coords_tensor[..., 0] / (self.img_size_native[1]//2) - 1
            # self.coords_tensor[..., 1] = self.coords_tensor[..., 1] / (self.img_size_native[0]//2) - 1
            self.coords_tensor[..., 0] = 2*self.coords_tensor[..., 0] / (self.img_size_native[1]-1) - 1
            self.coords_tensor[..., 1] = 2*self.coords_tensor[..., 1] / (self.img_size_native[0]-1) - 1

        return found_corners

    def __call__(self, input_img, img_size=None):
        """
        This forward pass returns the warped image.

        :param input_img: A numpy grayscale image shape of [H, W].
        :param img_size: output size, default None.
        :return: output_img: warped image with pre-calculated homography and destination size.
        """

        if img_size is None:
            img_size = [int((num_circs - 1) * space + 2 * pad_pixs)
                        for num_circs, space, pad_pixs in zip(self.num_circles, self.spacing_size, self.pad_pixels)]

        if torch.is_tensor(input_img):
            # output_img = F.grid_sample(input_img, self.coords_tensor, align_corners=False)
            output_img = F.grid_sample(input_img, self.coords_tensor, align_corners=True)
        else:
            if self.quadratic:
                output_img = transform.warp(input_img, self.h_transform, output_shape=(img_size[0], img_size[1]))
            else:
                output_img = cv2.warpPerspective(input_img, self.h_transform, (img_size[0], img_size[1]))

        return output_img

    @property
    def h_transform(self):
        return self._h_transform

    @h_transform.setter
    def h_transform(self, new_h):
        self._h_transform = new_h

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.coords_tensor is not None:
            slf.coords_tensor = slf.coords_tensor.to(*args, **kwargs)
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf