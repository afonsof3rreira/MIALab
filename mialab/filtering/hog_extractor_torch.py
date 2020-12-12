import os
import stat
import sys
import time
import SimpleITK as sitk
import numpy as np
from pymia.filtering.filter import FilterParams

from exercise.exercise_simpleitk import load_image
import scipy.ndimage.filters as fltrs
import matplotlib.pyplot as plt
import pymia.filtering.filter as fltr
import imageio
import shutil
import multiprocessing as mp
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Useful link: https://www.learnopencv.com/histogram-of-oriented-gradients/
# --> article for more info: https://www.hindawi.com/journals/bmri/2017/3956363/


class SimpleHOGModule(nn.Module):

    def __init__(self, image: sitk.Image, theta_bins=8, phi_bins=8, block_size=15, stride: int = 1):
        """Initializes a new instance of the Hog_3D_extractor class.

        Args:
            theta_bins (int): the theta dimension of the histogram.
            phi_bins (int): the phi dimension of the histogram.
            block_size (int): the size of a 3D block (block dimensions = block_size x block_size x block_size)
            from which to extract each HOG feature set.
            stride (int): the displacement applied when running the block through the image.
        """
        super().__init__()

        # Set the general properties
        # ==========================
        self.image = image
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins
        self.block_size = block_size
        self.stride = stride
        self.output_convolved_images = []
        #   these lens are the ranges of angle values that each histogram bin can receive
        self.theta_bin_len = (2 * np.math.pi) / self.theta_bins
        self.phi_bin_len = np.math.pi / self.phi_bins
        assert is_odd(block_size), "block size must be odd"
        self.running_time = 0

        # Calculates the number of blocks that fit in the image in each dimension
        # ==========================
        img_arr_shape = sitk.GetArrayFromImage(image).shape

        # maximal coordinates that can be reached (inclusive)
        or_z = img_arr_shape[0] - 1
        or_y = img_arr_shape[1] - 1
        or_x = img_arr_shape[2] - 1

        # starting coordinates
        z, y, x = 0, 0, 0

        # centered starting coordinates in the padded image = origin of the original image
        cent_z, cent_y, cent_x = (self.block_size // 2), (self.block_size // 2), (self.block_size // 2)

        while cent_z <= or_z + (self.block_size // 2) - self.stride:
            cent_z += self.stride
            z += 1
        self.nr_block_z = z + 1

        while cent_y <= or_y + (self.block_size // 2) - self.stride:
            cent_y += self.stride
            y += 1
        self.nr_block_y = y + 1

        while cent_x <= or_x + (self.block_size // 2) - self.stride:
            cent_x += self.stride
            x += 1
        self.nr_block_x = x + 1

        # Setting the filter torch arrays that approximate the partial first-order derivatives for z, y, x
        # ==========================

        s_z = torch.FloatTensor([[[0, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]]])

        s_y = torch.FloatTensor([[[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 [[0, -1, 0],
                                  [0, 0, 0],
                                  [0, 1, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]])

        s_x = torch.FloatTensor([[[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [-1, 0, 1],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]])

        # Add an additional dimension to the tensors
        s_z = torch.unsqueeze(s_z, dim=0)
        s_y = torch.unsqueeze(s_y, dim=0)
        s_x = torch.unsqueeze(s_x, dim=0)

        # Concatenate the tensors to one tensor and unsqueeze it
        # such that the shape is of form [ch_out, ch_in, kT, kH, kW]
        self.mat = torch.cat((s_z, s_y, s_x), dim=0)
        self.mat = torch.unsqueeze(self.mat, dim=1)
        if torch.cuda.is_available():
            self.mat = self.mat.cuda()
        self.register_buffer("weight", self.mat)

    def forward(self, x: torch.Tensor):

        # Cast the input to the correct data type
        x = x.type(torch.FloatTensor)
        if torch.cuda.is_available():
            x = x.cuda()

        # Render the tensor shape appropriate
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
        # print(x.size())
        # print(x.shape[5])
        # print(x.dtype)
        # Check if the input tensor x has the correct number of dimensions
        # The input tensor must be of shape [batch, ch_in, iT, iH, iW, feature set]
        if x.ndim != 5:
            raise ValueError(f'The input tensor needs to be 5 dimensional, but has {x.ndim} dimensions!')
        if x.shape[1] != 1:
            raise ValueError(f'The number of input channels is not correct ({x.shape[1]} instead of 1)!')

        # Convolve the input tensor with the weights and remove the first dimension
        out = F.conv3d(x, self.weight, None, self.stride, padding=1)
        out = torch.squeeze(out, dim=0)

        # Retrieving convolved images
        img_conv_z = out[0]
        img_conv_y = out[1]
        img_conv_x = out[2]

        # creating the output tensor
        new_size = list(x.size())
        new_size.append(self.theta_bins * self.phi_bins)
        out_arr = torch.zeros(new_size, dtype=x.dtype)

        #   cord_z, cord_y, cord_x = are the starting-window-coordinates of the padded image
        # and correspond to the central window coordinates on the original image
        with torch.no_grad():
            for z in range(self.nr_block_z):
                cord_z = int(self.stride * z)

                for y in range(self.nr_block_y):
                    cord_y = int(self.stride * y)

                    for x in range(self.nr_block_x):
                        cord_x = int(self.stride * x)

                        # Calculates the HOG-features for a 3D block in the image.

                        # z, y, x are the starting coordinates of each block on the overall image
                        #   histogram matrix
                        h_bin_set = torch.zeros((self.theta_bins, self.phi_bins))
                        eps = sys.float_info.epsilon

                        for zz in range(cord_z, cord_z + self.block_size):
                            for yy in range(cord_y, cord_y + self.block_size):
                                for xx in range(cord_x, cord_x + self.block_size):

                                    #   magnitude
                                    r = \
                                        np.sqrt(img_conv_x[zz, yy, xx] ** 2 + img_conv_y[zz, yy, xx] ** 2 + img_conv_z[
                                            zz, yy, xx] ** 2)
                                    #   theta
                                    theta = \
                                        np.math.atan(img_conv_y[zz, yy, xx] / (img_conv_x[zz, yy, xx] + eps))
                                    #   phi
                                    phi = \
                                        np.math.acos(img_conv_z[zz, yy, xx] / (r + eps))
                                    #   updating histogram matrix

                                    # --------------
                                    # def bin_assignment(self, r, theta, phi, h_bin_set):

                                    theta_split, phi_split = True, True
                                    low_t_bin_ratio, high_t_bin_ratio, low_p_bin_ratio, high_p_bin_ratio = None, None, None, None

                                    # ----- theta -----
                                    theta = angle_normalizer(theta, 0, 2 * np.pi)
                                    theta_raw_index = theta / self.theta_bin_len
                                    low_t_bin_index = int(np.floor(theta_raw_index))

                                    # in case theta index > max index, go back to origin
                                    if low_t_bin_index == self.theta_bins:
                                        low_t_bin_index = 0

                                    #   defining adjacent bin indices for magnitude assignment in case of splitting
                                    if np.modf(theta_raw_index)[0] <= 0.5:
                                        low_t_bin_ratio = np.modf(theta_raw_index)[0]
                                        high_t_bin_ratio = 1 - np.modf(theta_raw_index)[0]
                                    elif np.modf(theta_raw_index)[0] > 0.5:
                                        low_t_bin_ratio = 1 - np.modf(theta_raw_index)[0]
                                        high_t_bin_ratio = np.modf(theta_raw_index)[0]
                                    else:
                                        theta_split = False

                                    # ----- phi -----
                                    phi = angle_normalizer(phi, 0, np.pi)
                                    phi_raw_index = phi / self.phi_bin_len
                                    low_p_bin_index = int(np.floor(phi_raw_index))

                                    # in case phi index > max index, go back to origin
                                    if low_p_bin_index == self.phi_bins:
                                        low_p_bin_index = 0

                                    #   defining adjacent bin indices for magnitude assignment in case of splitting
                                    if 0 < np.modf(theta_raw_index)[0] <= 0.5:
                                        low_p_bin_ratio = np.modf(theta_raw_index)[0]
                                        high_p_bin_ratio = 1 - np.modf(theta_raw_index)[0]
                                    elif 1 > np.modf(theta_raw_index)[0] > 0.5:
                                        low_p_bin_ratio = 1 - np.modf(theta_raw_index)[0]
                                        high_p_bin_ratio = np.modf(theta_raw_index)[0]
                                    else:
                                        phi_split = False

                                    # ----- 4 possible 2D histogram splitting cases -----

                                    # 1) when both phi and theta fit exactly in 1 bin
                                    if not theta_split and not phi_split:
                                        h_bin_set[low_t_bin_index][low_p_bin_index] += r

                                    # 2) when only theta is split in 2
                                    elif theta_split and not phi_split:

                                        # in case theta is split between last and origin bin
                                        if low_t_bin_index == self.theta_bins - 1:
                                            high_t_bin_index = 0
                                        else:
                                            high_t_bin_index = low_t_bin_index + 1

                                        h_bin_set[low_t_bin_index][low_p_bin_index] += r * low_t_bin_ratio
                                        h_bin_set[high_t_bin_index][low_p_bin_index] += r * high_t_bin_ratio

                                    # 3) when only phi is split in 2
                                    elif phi_split and not theta_split:

                                        # in case phi is split between last and origin bin
                                        if low_p_bin_index == self.phi_bins - 1:
                                            high_p_bin_index = 0
                                        else:
                                            high_p_bin_index = low_p_bin_index + 1

                                        h_bin_set[low_t_bin_index][low_p_bin_index] += r * low_p_bin_ratio
                                        h_bin_set[low_t_bin_index][high_p_bin_index] += r * high_p_bin_ratio

                                    # 4) when both phi and theta are split in a 2x2 histogram "block"
                                    else:
                                        # in case theta is split between last and origin bin
                                        if low_t_bin_index == self.theta_bins - 1:
                                            high_t_bin_index = 0
                                        else:
                                            high_t_bin_index = low_t_bin_index + 1

                                        # in case phi is split between last and origin bin
                                        if low_p_bin_index == self.phi_bins - 1:
                                            high_p_bin_index = 0
                                        else:
                                            high_p_bin_index = low_p_bin_index + 1

                                        # theta-axis wise splitting
                                        h_bin_set[low_t_bin_index][
                                            low_p_bin_index] += r * low_t_bin_ratio * low_p_bin_ratio
                                        h_bin_set[high_t_bin_index][
                                            low_p_bin_index] += r * high_p_bin_ratio * low_p_bin_ratio
                                        # phi-axis wise splitting
                                        h_bin_set[low_t_bin_index][
                                            high_p_bin_index] += r * low_t_bin_ratio * high_p_bin_ratio
                                        h_bin_set[high_t_bin_index][
                                            high_p_bin_index] += r * high_p_bin_ratio * high_p_bin_ratio

                        #   returning histogram as a feature set
                        out_arr[0, 0, z, y, x] = torch.flatten(h_bin_set)
            #             print(out_arr[0, 0, z, y, x])
                        print('finished 1 x point')
            # print(out.size())
            out = torch.squeeze(out_arr, dim=0)
            return out

    def Get_bin_sizes(self):
        return [self.theta_bins, self.phi_bins]


class HOGExtractorGPU(fltr.Filter):

    def __init__(self, image: sitk.Image):
        super().__init__()
        self.hog_module = SimpleHOGModule(image)

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        # Cast the image to a Pytorch tensor
        image_arr = torch.from_numpy(sitk.GetArrayFromImage(image))

        # print(image_arr.size())

        # Compute the 3D-HOG features using Pytorch
        features = self.hog_module(image_arr)

        # Detach the features from the computational graph, write the memory to the RAM and
        # cast the features to be a np.ndarray
        features_np = features.detach().cpu().numpy()
        features_np = np.squeeze(features_np)

        img_out = sitk.GetImageFromArray(features_np)
        img_out.CopyInformation(image)
        return img_out


def angle_normalizer(angle, lower_bound, upper_bound):
    """Wraps an angle in radians to a specific range of values. Example:
    angle_normalizer(4*pi, 0, 2*pi) -> 2*pi

    Args:
        angle (float): the angle in radians to be wrapped
        lower_bound (float): the starting range value, inclusive
        upper_bound (float): the ending range value, inclusive

    Returns:
        float: the normalized angle in radians
    """

    newAngle = angle
    while newAngle < lower_bound:
        newAngle += 2 * np.pi
    while newAngle > upper_bound:
        newAngle -= 2 * np.pi
    return newAngle


def is_odd(nr):
    """Confirms that a number is odd
    Args:
        nr (float): the number
    Returns:
        bool: True if the number is odd, False otherwise
    """
    if nr % 2 == 0:
        return False
    else:
        return True

# def info_txt_writer(path: str, filename: str, value):
#     with open(os.path.join(path, filename + '.txt'), 'w') as outfile:
#         outfile.write('-' * 37 + '\n')
#         #   writing running time
#         rounded_val = round(value, 2)
#         outfile.write('-' * 10 + ' Running time = ' + str(rounded_val) + ' second(s) ' + '-' * 10)
#         outfile.write('\n' + '-' * 37)


path1 = 'C:/Users/afons/PycharmProjects/MIAlab project/data/train/116524/T1native.nii.gz'
image1 = load_image(path1, False)
# image1_np = sitk.GetArrayFromImage(image1)
# image1 = sitk.GetImageFromArray(image1_np[:179, :, :])
print(image1.GetSize())
# new dimensions x, y, z = (181, 217, 179)
hog_extractor = HOGExtractorGPU(image1)
hog_extractor.execute(image1)


# image1_arr_slice_a = a[98]
# image1_arr_slice_b = b[98]
# image1_arr_slice_c = c[98]
# image1_slice_a = sitk.GetImageFromArray(image1_arr_slice_a)
# image1_slice_b = sitk.GetImageFromArray(image1_arr_slice_b)
# image1_slice_c = sitk.GetImageFromArray(image1_arr_slice_c)
#
# saving_path = 'C:/Users/afons/OneDrive - Universidade de Lisboa/Erasmus/studies/MIAlab/project/Results_midterm/MIALab_tests/brain_images_torch/after_conv'
# # sitk.WriteImage(image1_slice_a, os.path.join(saving_path, 'a.png'))
# # sitk.WriteImage(image1_slice_b, os.path.join(saving_path, 'b.png'))
# # sitk.WriteImage(image1_slice_c, os.path.join(saving_path, 'c.png'))
# sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(image1_slice_a), sitk.sitkUInt8), os.path.join(saving_path, 'a_slice_zeros.png'))
# sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(image1_slice_b), sitk.sitkUInt8), os.path.join(saving_path, 'b_slice_zeros.png'))
# sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(image1_slice_c), sitk.sitkUInt8), os.path.join(saving_path, 'c_slice_zeros.png'))


# if __name__ == '__main__':
#
#     # loading the image
#     path1 = 'C:/Users/afons/PycharmProjects/MIAlab project/data/train/116524/T1native.nii.gz'
#     image1 = load_image(path1, False)
#
#     # ----- testing the algorithm -----
#     block_size = 15
#     block_displacement = 20

# confirming equal results when multiprocess = T or F

# file_name_m = 'brain_slice_m.nii'
# file_name_nm = 'brain_slice_nm.nii'
# dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mia-result')
# path_m = os.path.join(dir_path, file_name_m)
# path_nm = os.path.join(dir_path, file_name_nm)
#
# image_m = load_image(path_m, False)
# print('la')
# image_nm = load_image(path_nm, False)
# print('la')
#
#
# A = sitk.GetArrayFromImage(image_m)
# print('la')
#
# B = sitk.GetArrayFromImage(image_nm)
# print('la')
#
# print((A==B).all())

# -----------------------comment / uncomment------------------

# hog_example = HOG_extractor(block_size=block_size, block_displacement=block_displacement)
# hog_image = hog_example.execute(image1, multiprocessing=False)
# print('HOG extraction finished')
#
# # creating the folder string name where to write the extracted HOG image
# date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# test_type = '__size_' + str(block_size) + '_disp_' + str(block_displacement)
# folder_name = date + test_type
#
# dir_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mia-result/HOG_testing'),
#                         folder_name)
#
# # this deletes the dir if exists
# if os.path.exists(dir_path):
#     shutil.rmtree(dir_path)
# os.makedirs(dir_path)
#
# full_path = os.path.join(dir_path, 'brain_slice.nii')
# sitk.WriteImage(hog_image, full_path)
# print('HOG image saved')
# info_txt_writer(dir_path, 'running_time', hog_example.GetRunningInfo())

# ------------------------comment / uncomment-----------------
#
# getting slice 100
# image1_arr = sitk.GetArrayFromImage(image1)
# image1_arr_22 = image1_arr[105]
# image1_22 = sitk.GetImageFromArray(image1_arr_22)
#
# # loading brain slice image
# image = sitk.ReadImage(os.path.join(new_dir_path, 'brain_slice_105_.nii'))
# image_arr = sitk.GetArrayFromImage(image)
# print(image_arr.shape)
#
# # saving every feature image
# file_names = []
# for feature_i in range(image_arr.shape[2]):
#     saving_img_arr = image_arr[:, :, feature_i]
#     saving_img = sitk.GetImageFromArray(saving_img_arr)
#     file_name = 'feature{0}.png'.format(feature_i)
#     saving_path = os.path.join(new_dir_path, file_name)
#     sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(saving_img), sitk.sitkUInt8), saving_path)
#     file_names.append(file_name)
#
# # creating a gif
# images = []
# for filename in file_names:
#     images.append(imageio.imread(os.path.join(new_dir_path, filename)))
# imageio.mimsave(os.path.join(new_dir_path, 'video.gif'), images)
