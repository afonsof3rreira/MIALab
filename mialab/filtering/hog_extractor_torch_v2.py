import math
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

    def __init__(self, image: sitk.Image, theta_bins=8, phi_bins=8, block_size=15, stride: int = 1,
                 max_phi_angle=math.pi):
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
        self.max_phi_angle = max_phi_angle
        self.output_convolved_images = []
        #   these lens are the ranges of angle values that each histogram bin can receive
        self.theta_bin_len = (2 * np.math.pi) / self.theta_bins
        self.phi_bin_len = np.math.pi / self.phi_bins
        assert is_odd(block_size), "block size must be odd"
        self.running_time = 0

        # Calculates the number of blocks that fit in the image in each dimension
        # ==========================
        img_arr_shape = sitk.GetArrayFromImage(image).shape
        # print('shape = ' + str(img_arr_shape))
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
        # print(self.nr_block_z)
        # print(self.nr_block_y)
        # print(self.nr_block_x)

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
        offset = self.block_size // 2
        self.pooler = nn.AvgPool3d(block_size, stride=stride, padding=offset, ceil_mode=False, count_include_pad=False)


    def forward(self, x: torch.Tensor):

        # Cast the input to the correct data type
        x = x.type(torch.FloatTensor)
        if torch.cuda.is_available():
            x = x.cuda()

        # Render the tensor shape appropriate
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)

        # Check if the input tensor x has the correct number of dimensions
        # The input tensor must be of shape [batch, ch_in, iT, iH, iW, feature set]
        if x.ndim != 5:
            raise ValueError(f'The input tensor needs to be 5 dimensional, but has {x.ndim} dimensions!')
        if x.shape[1] != 1:
            raise ValueError(f'The number of input channels is not correct ({x.shape[1]} instead of 1)!')

        # Convolve the input tensor with the weights and remove the first dimension
        # print(x.size())
        offset = self.block_size // 2 + 1
        # print(offset)
        padding = [offset, offset, offset]
        out = F.conv3d(x, self.weight, None, self.stride, padding=padding)
        out = torch.squeeze(out, dim=0)  # - torch.Size([3, 195, 231, 195])

        # convoluted arrays in z, y x = out[0], out[1], out[2]

        # print('padded size = ' + str(out.size()))

        # creating the output tensor

        #   cord_z, cord_y, cord_x = are the starting-window-coordinates of the padded image
        # and correspond to the central window coordinates on the original image
        # ------------------------------------
        eps = sys.float_info.epsilon
        with torch.no_grad():
            # x torch => torch.Size([3, 181, 217, 181])

            #   magnitude
            mag = out.norm(p="fro", dim=0)  # correct !!!
            # print('mag size = ' + str(mag.size()))
            # print('z blocks = ' + str(self.nr_block_z))

            #   theta
            theta = torch.atan2(out[1], out[2])

            # phi
            phi = torch.acos(torch.div(out[0], mag + eps))
            # size become like torch.Size([1, 1, 195, 231, 195])
            mag = torch.unsqueeze(torch.unsqueeze(mag, dim=0), dim=0)
            theta = torch.unsqueeze(torch.unsqueeze(theta, dim=0), dim=0)
            phi = torch.unsqueeze(torch.unsqueeze(phi, dim=0), dim=0)

            # print('=' * 10)
            # print(mag.size())
            # print(theta.size())
            # print(phi.size())
            # print('=' * 10)

            # Binning Mag with linear interpolation
            theta_raw_ind = (theta / self.max_phi_angle * self.phi_bins)  # torch.Size([195, 231, 195])
            theta_frac_ind = torch.frac(theta_raw_ind)

            phi_raw_ind = (phi / self.max_phi_angle * self.phi_bins)  # torch.Size([195, 231, 195])
            phi_frac_ind = torch.frac(phi_raw_ind)

            # --------------------------
            # creating a torch containing lower and upper indices for theta and phi (4 dimensions)
            # torch will be like this [theta lower ind, theta upper ind, phi lower ind, phi upper ind]

            #  # out -> torch.Size([3, 195, 231, 195])
            conv, d, h, w = out.size()
            int_indices = torch.zeros(4, d, h, w, dtype=torch.int64, device=x.device)  # torch.Size([4, 195, 231, 195])
            int_indices = torch.unsqueeze(int_indices, 0)  # torch.Size([1, 4, 195, 231, 195])

            # theta indices
            #   lower theta indices (0)
            int_indices[0, 0, :, :, :] = theta_raw_ind.floor().long() % self.theta_bins
            #   upper theta indices (1)
            int_indices[0, 1, :, :, :] = theta_raw_ind.ceil().long() % self.theta_bins

            # phi indices
            #   lower phi indices   (2)
            int_indices[0, 2, :, :, :] = phi_raw_ind.floor().long() % self.phi_bins
            #   upper phi indices   (3)
            int_indices[0, 3, :, :, :] = phi_raw_ind.ceil().long() % self.phi_bins

            # int_indices = torch.unsqueeze(int_indices, dim=0)

            # convert int indices to int64

            # -------------------------- creating a torch containing the "fractional parts" (%) of lower and upper
            # indices for theta and phi (4 dimensions)
            # torch will be like this [% theta, 1 - % theta, % phi, 1 - % phi]

            frac_parts = torch.zeros(int_indices.size(), device=x.device)

            # theta fractions
            #   lower theta         (0)
            frac_parts[:, 0, :, :, :] = torch.abs(theta_frac_ind)
            #   upper theta         (1)
            frac_parts[:, 1, :, :, :] = torch.abs(1 - theta_frac_ind)

            # phi fractions
            #   lower phi           (2)
            frac_parts[:, 2, :, :, :] = torch.abs(phi_frac_ind)
            #   upper phi           (3)
            frac_parts[:, 3, :, :, :] = torch.abs(1 - phi_frac_ind)

            # -------------------------- creating a torch containing the "composed fractional parts" (%) of lower and
            # upper indices for theta and phi (4 dimensions) torch will be like this:
            # [(% theta) x (% phi),   (% theta) x (1 - % phi),  (1 - % theta) x (% phi),  (1 - % theta) x (1 - % phi)]

            composed_frac_parts = torch.zeros(int_indices.size(), device=x.device)

            # theta
            #  (% theta) x (% phi)          (0)
            composed_frac_parts[:, 0, :, :, :] = torch.mul(frac_parts[0, 0, :, :, :], frac_parts[0, 2, :, :, :])
            #  (% theta) x (1 - % phi)      (1)
            composed_frac_parts[:, 1, :, :, :] = torch.mul(frac_parts[0, 0, :, :, :], frac_parts[0, 3, :, :, :])

            # phi indices
            #  (1 - % theta) x (% phi)      (2)
            composed_frac_parts[:, 2, :, :, :] = torch.mul(frac_parts[0, 1, :, :, :], frac_parts[0, 2, :, :, :])
            #  (1 - % theta) x (1 - % phi)  (3)
            composed_frac_parts[:, 3, :, :, :] = torch.mul(frac_parts[0, 1, :, :, :], frac_parts[0, 3, :, :, :])

            # ---------- scattering composed fractions to the right angle (theta or phi)
            # creating tensors containing theta or phi bins (1)
            n, c, d, h, w = x.size()

            theta_lower_ind_fracs_f_0 = torch.zeros((1, self.theta_bins,
                                                     d + 2 * offset - 2, h + 2 * offset - 2, w + 2 * offset - 2),
                                                    dtype=torch.float, device=x.device)

            theta_lower_ind_fracs_f_1 = torch.zeros(theta_lower_ind_fracs_f_0.size(), device=x.device)

            theta_upper_ind_fracs_f_2 = torch.zeros(theta_lower_ind_fracs_f_0.size(), device=x.device)

            theta_upper_ind_fracs_f_3 = torch.zeros(theta_lower_ind_fracs_f_0.size(), device=x.device)

            # phi_upper_ind = torch.zeros(phi_upper_ind.size())
            # theta_lower_indices # torch.Size([1, theta_bins (1), 195, 231, 195])

            # compesed fracs to be organized

            low_p_ordered_by_low_t = torch.zeros((1, self.theta_bins,
                                                  d + 2 * offset - 2, h + 2 * offset - 2, w + 2 * offset - 2),
                                                 dtype=torch.int64, device=x.device)

            upper_p_ordered_by_low_t = torch.zeros(low_p_ordered_by_low_t.size(), dtype=torch.int64, device=x.device)

            lower_p_ordered_by_upper_t = torch.zeros(low_p_ordered_by_low_t.size(), dtype=torch.int64, device=x.device)

            upper_p_ordered_by_upper_t = torch.zeros(low_p_ordered_by_low_t.size(), dtype=torch.int64, device=x.device)

            # ---------- dims for scattering
            # out torch.Size([1, 12, 217, 181])
            # ind torch.Size([1, 1, 217, 181])
            # val torch.Size([1, 1, 217, 181])
            # print('=*10')

            # here we got a set of fractions scattered through the different lower theta indices
            int_indices = torch.unsqueeze(int_indices, dim=0)
            # print('=*10')
            # print(torch.mul(composed_frac_parts[:, 0, :, :, :], mag).size())
            # print(int_indices[:, :, 0, :, :, :].size())
            # print(theta_lower_ind_fracs_f_0.size())

            theta_lower_ind_fracs_f_0.scatter_(1, int_indices[:, :, 0, :, :, :],
                                               torch.mul(composed_frac_parts[:, 0, :, :, :], mag))

            theta_lower_ind_fracs_f_1.scatter_(1, int_indices[:, :, 0, :, :, :],
                                               torch.mul(composed_frac_parts[:, 1, :, :, :], mag))

            theta_upper_ind_fracs_f_2.scatter_(1, int_indices[:, :, 1, :, :, :],
                                               torch.mul(composed_frac_parts[:, 2, :, :, :], mag))

            theta_upper_ind_fracs_f_3.scatter_(1, int_indices[:, :, 1, :, :, :],
                                               torch.mul(composed_frac_parts[:, 3, :, :, :], mag))

            # print('=' * 10)
            # print(int_indices[:, :, 2, :, :, :].size())
            # print(int_indices[:, :, 0, :, :, :].size())
            # print(low_p_ordered_by_low_t.size())

            #                                lower theta indices (0) #   lower phi indices   (2)
            low_p_ordered_by_low_t.scatter_(1, int_indices[:, :, 0, :, :, :], int_indices[:, :, 2, :, :, :])
            low_p_ordered_by_low_t = torch.unsqueeze(low_p_ordered_by_low_t, dim=0)
            low_p_ordered_by_low_t = torch.transpose(low_p_ordered_by_low_t, 1, 2)

            #                                lower theta indices (0) #   upper phi indices   (3)
            upper_p_ordered_by_low_t.scatter_(1, int_indices[:, :, 0, :, :, :], int_indices[:, :, 3, :, :, :])
            upper_p_ordered_by_low_t = torch.unsqueeze(upper_p_ordered_by_low_t, dim=0)
            upper_p_ordered_by_low_t = torch.transpose(upper_p_ordered_by_low_t, 1, 2)

            #                                upper theta indices (1) #   lower phi indices   (2)
            lower_p_ordered_by_upper_t.scatter_(1, int_indices[:, :, 1, :, :, :], int_indices[:, :, 2, :, :, :])
            lower_p_ordered_by_upper_t = torch.unsqueeze(lower_p_ordered_by_upper_t, dim=0)
            lower_p_ordered_by_upper_t = torch.transpose(lower_p_ordered_by_upper_t, 1, 2)

            #                                upper theta indices (1) #   upper phi indices   (3)
            upper_p_ordered_by_upper_t.scatter_(1, int_indices[:, :, 1, :, :, :], int_indices[:, :, 3, :, :, :])
            upper_p_ordered_by_upper_t = torch.unsqueeze(upper_p_ordered_by_upper_t, dim=0)
            upper_p_ordered_by_upper_t = torch.transpose(upper_p_ordered_by_upper_t, 1, 2)

            theta_lower_ind_fracs_f_0 = torch.unsqueeze(theta_lower_ind_fracs_f_0, dim=0)
            theta_lower_ind_fracs_f_0 = torch.transpose(theta_lower_ind_fracs_f_0, 1, 2)

            theta_lower_ind_fracs_f_1 = torch.unsqueeze(theta_lower_ind_fracs_f_1, dim=0)
            theta_lower_ind_fracs_f_1 = torch.transpose(theta_lower_ind_fracs_f_1, 1, 2)

            theta_upper_ind_fracs_f_2 = torch.unsqueeze(theta_upper_ind_fracs_f_2, dim=0)
            theta_upper_ind_fracs_f_2 = torch.transpose(theta_upper_ind_fracs_f_2, 1, 2)

            theta_upper_ind_fracs_f_3 = torch.unsqueeze(theta_upper_ind_fracs_f_3, dim=0)
            theta_upper_ind_fracs_f_3 = torch.transpose(theta_upper_ind_fracs_f_3, 1, 2)

            # print(low_p_ordered_by_low_t.size())

            # bin assignment
            n, c, d, h, w = x.shape
            out_plus_bins = torch.zeros(  # torch.Size([1, 8, 8, 195, 231, 195])
                (n, self.theta_bins, self.phi_bins, d + 2 * offset - 2, h + 2 * offset - 2, w + 2 * offset - 2),
                dtype=torch.float, device=x.device)

            # print('=' * 10)
            # print(out_plus_bins.size())
            # print(low_p_ordered_by_low_t.size())
            # print(theta_lower_ind_fracs_f_1.size())

            # assigning low t x low p
            out_plus_bins.scatter_(2, low_p_ordered_by_low_t, theta_lower_ind_fracs_f_0)
            out_plus_bins.scatter_add_(2, upper_p_ordered_by_low_t, theta_lower_ind_fracs_f_1)
            out_plus_bins.scatter_add_(2, lower_p_ordered_by_upper_t, theta_upper_ind_fracs_f_2)
            out_plus_bins.scatter_add_(2, upper_p_ordered_by_upper_t, theta_upper_ind_fracs_f_3)

            # print('\n' * 3)
            # print('phi not working')
            # print(torch.max(phi_raw_ind))
            # print(torch.min(phi_raw_ind))

            # if torch.all(torch.eq(phi, torch.zeros(phi.size())) == True):
            #     print('\n'*3)
            #     print('phi not working')

            # print(theta_frac_ind.size())

            # checking the zero problem # out_plus_bins = torch.Size([1, 8, 8, 195, 231, 195])
            # for feature1 in range(8):
            #     for feature2 in range(8):
            #         print('checking feature1 and 2 = ' + str(feature1) + ' ' + str(feature2))
            #         sub_features = out_plus_bins[0, feature1, feature2, :, :, :]
            #         if torch.all((sub_features == 0)):
            #             print('\n' + 'found whole zeros at ... feature1 and 2 = ' + str(feature1) + ' ' + str(feature2))

            # print('=' * 10 + '\n' + '=' * 10)
            # print(out_plus_bins.size())

            print(out_plus_bins.size())
            # torch.Size([1, 8, 8, 195, 231, 195]) -> torch.Size([1, 64, 195, 231, 195])
            out_plus_bins = torch.reshape(out_plus_bins, (n, self.theta_bins * self.phi_bins,
                                                          d + 2 * offset - 2, h + 2 * offset - 2, w + 2 * offset - 2))
            print(out_plus_bins.size())
            return self.pooler(out_plus_bins)   # torch.Size([1, 64, 195, 231, 195])

    def Get_block_size(self):
        return self.block_size


class HOGExtractorGPU(fltr.Filter):

    def __init__(self, image: sitk.Image):
        super().__init__()
        self.hog_module = SimpleHOGModule(image)

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        # Cast the image to a Pytorch tensor
        image_arr = torch.from_numpy(sitk.GetArrayFromImage(image))

        # print(image_arr.size())

        # Compute the 3D-HOG features using Pytorch
        features = self.hog_module(image_arr)   # torch.Size([1, 64, 195, 231, 195])
        # print('=' * 20)
        # print(image.GetSize())
        # print(type(image.GetSize()))
        print(features.size())
        # Detach the features from the computational graph, write the memory to the RAM and
        # cast the features to be a np.ndarray
        features_np = features.detach().cpu().numpy()
        features_np = np.squeeze(features_np)   # torch.Size([64, 195, 231, 195])
        features_np = np.transpose(features_np, (1, 2, 3, 0))   # torch.Size([195, 231, 195, 64])

        image_size = image.GetSize()
        # [7:180 + 8, 7:216 + 8, 7:180 + 8]
        offset = self.hog_module.Get_block_size() // 2

        features_np = features_np[offset:image_size[2] + offset,
                                  offset:image_size[1] + offset,
                                  offset:image_size[0] + offset]

        # print(features_np.shape)
        # checking the zero problem
        # for feature in range(64):
        #     print('checking feature = ' + str(feature))
        #     sub_features = features_np[:, :, :, feature]
        #     if np.all((sub_features == 0)):
        #         print('\n' + 'found whole zeros at ... feature = ' + str(feature))

        # print(features_np.shape)
        img_out = sitk.GetImageFromArray(features_np)
        # print(img_out.GetSize())
        # print(img_out.GetNumberOfComponentsPerPixel())
        img_out.CopyInformation(image)
        return img_out


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


# val = torch.tensor(6.34 * 10)
# val1 = val.floor().long() % 10
# print(val1)

# source_vec = torch.tensor(
#     [[3.3, 4.0, 5.4, 6, 6.98888, 4], [6.2, 5.0, 3.0, 5, 2, 3], [0.0, 8.0, 4.0, 4, 3, 2], [6.0, 5.0, 3.0, 5, 2, 3],
#      [6.0, 5.0, 3.0, 5, 2, 3], [6.0, 5.0, 3.3, 5.9, 2, 3]])
#
# source_vec = torch.unsqueeze(source_vec, 0)
#
#
# print(source_vec)
# # source_vec[0, :, :] = 3
# print(source_vec.size())
#
# new_vec = torch.zeros(3, 3)
# new_vec = torch.unsqueeze(torch.unsqueeze(new_vec, 0), 0)
# new_vec = new_vec.repeat_interleave(4, dim=0)
# new_vec = new_vec.repeat_interleave(4, dim=1)
# print('\n')
# # print(new_vec.size())
# # print(new_vec)
#
# block_size = 3
#
# # tensor([[3., 4., 5., 6., 6., 4.],
# #         [6., 5., 3., 5., 2., 3.],
# #         [0., 8., 4., 4., 3., 2.],
# #         [6., 5., 3., 5., 2., 3.],
# #         [6., 5., 3., 5., 2., 3.],
# #         [6., 5., 3., 5., 2., 3.]])
#
# scatter_vec = torch.tensor([[[3., 4., 5., 6., 6., 4.],
#         [6., 5., 3., 5., 2., 3.],
#         [0., 8., 4., 0, 3., 2.],
#         [6., 5., 3., 5., 2., 3.],
#         [6., 5., 3., 5., 2., 3.],
#         [6., 5., 3., 5., 2., 3.]]])
# # source_vec = torch.tensor([2.3, 9.7, 9.8, 8.1948])
#
# source_vec[torch.frac(source_vec) < 0.5] = 0
# print('la')
# print(source_vec)
#
# # new_vec = torch.zeros(torch.unsqueeze(torch.unsqueeze(scatter_vec, dim=0), dim=0).size())
# # new_vec = new_vec.repeat_interleave(3, dim=0)
# # new_vec = new_vec.repeat_interleave(3, dim=1)
#
# print('...new vec size')
# print(scatter_vec)


# for y in range(source_vec.size()[0] - block_size + 1):
#     for x in range(source_vec.size()[1] - block_size + 1):
#         # print(y)
#         # print(x)
#         # print(source_vec[y:y + block_size, x:x + block_size])
#         # print(new_vec[y, x, :, :].size())
#         # print('source')
#         # print(source_vec[y: y + block_size, x:x + block_size].size())
#         new_vec[y, x, :, :] = torch.frac(source_vec[y:y + block_size, x:x + block_size])
#
#     print('new y')
#     # print('la')
# print(new_vec)


# print(type(source_vec.size()))
# # unsqueezed_vec = torch.unsqueeze(source_vec, dim=0)
# # print(unsqueezed_vec.size())
# new_vec = torch.zeros(torch.unsqueeze(torch.unsqueeze(source_vec, dim=0), dim=0).size())
# new_vec = new_vec.repeat_interleave(3, dim=0)
# new_vec = new_vec.repeat_interleave(3, dim=1)
# print('\n')
# print('...source vec size')
# print(source_vec.size())
#
# print('\n')
# print('...new vec size')
# print(new_vec.size())
# print('\n')
#
# print('....new vec')
# print(new_vec)
# # index_vec = torch.tensor()
# index_vec = torch.zeros(3,3)
# index_vec.int()
# print(index_vec.type())
# new_vec.scatter_(0, index_vec, source_vec)
# print('final')
# print(new_vec)

# input_tensor = torch.from_numpy(np.arange(1, 16)).float().view(3, 5)
# print(input_tensor.size())
# index_tensor = torch.tensor([4, 0, 1]).unsqueeze(1)
# print(index_tensor.size())

# ------------ running and saving feature .nii.gz image -----------------
path1 = 'C:/Users/afons/PycharmProjects/MIAlab project/data/train/116524/T1native.nii.gz'
image1 = sitk.ReadImage(path1, sitk.sitkFloat32)
# image1 = load_image(path1, False)
image1_np = sitk.GetArrayFromImage(image1)
image1 = sitk.GetImageFromArray(image1_np[:179, :, :])
print(image1.GetSize())
# new dimensions x, y, z = (181, 217, 179)
hog_extractor = HOGExtractorGPU(image1)
image_out = hog_extractor.execute(image1)
# --------------------------------------------------------------------
file_name = 'image_hog_final_3d_avg.nii.gz'
sitk.WriteImage(sitk.RescaleIntensity(image_out), file_name)
# ----------------------------------------------------------------------
# ---------------------
# path1 = 'C:/Users/afons/PycharmProjects\MIAlab project\mialab/filtering\image_hog_final.nii.gz'
# # image1 = load_image(path1, False)
# image1 = sitk.ReadImage(path1)
# # print(image1.GetNumberOfComponentsPerPixel())
# image1_np = sitk.GetArrayFromImage(image1)
# print(image1_np[0,0,0])
# print(image1_np.shape)
# for i in range(image1_np.shape[3]):
#     file_name = 'C:/Users/afons\PycharmProjects\MIAlab project\mialab/filtering\hog_v2\image_hog_feature_{}.nii.gz'.format(i)
#     image = image1_np[:, :, :, i]
#     image_sitk = sitk.GetImageFromArray(image)
#     sitk.WriteImage(image_sitk, file_name)

# # -----------------------
# path1 = 'C:/Users/afons/PycharmProjects\MIAlab project\mialab/filtering\image_hog_final.nii.gz'
# # # # image1 = load_image(path1, False)
# image1 = sitk.ReadImage(path1, sitk.sitkVectorUInt8)    #(181, 217, 181, 64)
# print(image1.GetNumberOfComponentsPerPixel())
#
# image1_np = sitk.GetArrayFromImage(image1)
# ## image1_np = image1_np[105, :, :]
# print(image1_np.shape)
# image1 = sitk.GetImageFromArray(image1_np)
# # sitk.WriteImage(sitk.Cast(image1, sitk.sitkUInt8), 'feat.jpeg')
#
# # print(image1.GetNumberOfComponentsPerPixel())
# image1_np = sitk.GetArrayFromImage(image1)
# print(image1_np.shape)  # (181, 217, 181, 64)
# # print(image1_np[0, 0, 0])
# # print(image1_np.shape)
#
# # saving every feature image
# import cv2
# # file_names = []
# new_dir_path = 'C:/Users/afons\PycharmProjects\MIAlab project\mialab/filtering\hog_slices_y_v2'
# #
# # im = cv2.imread('/hog_slices/image_hog_feature_{}.png'.format(i), cv2.IMREAD_GRAYSCALE)
# #
# i = 0
# while 1:
#     im = image1_np[:, :, i]
#     print(im.shape)
#     print(im.min())
#     print(im.max())
#     im = (im - im.min()) / (im.max() - im.min() + sys.float_info.epsilon)
#
#     print('... ' + 'i = ' + str(i))
#     cv2.imshow('i = ' + str(i), im)
#     cv2.waitKey()
#     i += 1
#     if i == 63:
#         i = 0
#
# # ---------------------------------------
# file_names = []
#
# for i in range(image1_np.shape[3]):
#     saving_img_arr = image1_np[:, 110, :, i]
#     saving_img = sitk.GetImageFromArray(saving_img_arr)
#     file_name = 'image_hog_feature_{}.png'.format(i)
#     saving_path = os.path.join(new_dir_path, file_name)
#     sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(saving_img), sitk.sitkUInt8), saving_path)
#     file_names.append(file_name)
#
#
#
# # creating a gif
# images = []
# for filename in file_names:
#     images.append(imageio.imread(os.path.join(new_dir_path, filename)))
# imageio.mimsave(os.path.join(new_dir_path, 'video.gif'), images)
