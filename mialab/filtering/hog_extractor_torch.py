import math
import sys
import SimpleITK as sitk
import numpy as np
from pymia.filtering.filter import FilterParams
import pymia.filtering.filter as fltr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

''' (Not used in the MIALab presentation/article)
This was our final implementation of the HOG feature extractor. Despite being implemented on pytorch, it was not adapted 
to the GPU usage because, meanwhile, we opted not to include it in our research experiments. Moreover, the HOG images
obtained for a few tested MRIs seem to be correct.
'''


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
        offset = self.block_size // 2 + 1
        padding = [offset, offset, offset]
        out = F.conv3d(x, self.weight, None, self.stride, padding=padding)
        out = torch.squeeze(out, dim=0)  # - torch.Size([3, 195, 231, 195])

        # convoluted arrays in z, y x = out[0], out[1], out[2]

        #   cord_z, cord_y, cord_x = are the starting-window-coordinates of the padded image
        # and correspond to the central window coordinates on the original image

        # ------------------------------------
        eps = sys.float_info.epsilon
        with torch.no_grad():

            #   magnitude
            mag = out.norm(p="fro", dim=0)

            #   theta
            theta = torch.atan2(out[1], out[2])

            # phi
            phi = torch.acos(torch.div(out[0], mag + eps))
            mag = torch.unsqueeze(torch.unsqueeze(mag, dim=0), dim=0)
            theta = torch.unsqueeze(torch.unsqueeze(theta, dim=0), dim=0)
            phi = torch.unsqueeze(torch.unsqueeze(phi, dim=0), dim=0)

            # Binning Mag with linear interpolation
            theta_raw_ind = (theta / self.max_phi_angle * self.phi_bins)
            theta_frac_ind = torch.frac(theta_raw_ind)

            phi_raw_ind = (phi / self.max_phi_angle * self.phi_bins)
            phi_frac_ind = torch.frac(phi_raw_ind)

            # --------------------------
            # creating a torch containing lower and upper indices for theta and phi (4 dimensions)
            # torch will be like this [theta lower ind, theta upper ind, phi lower ind, phi upper ind]

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

            # here we got a set of fractions scattered through the different lower theta indices
            int_indices = torch.unsqueeze(int_indices, dim=0)

            theta_lower_ind_fracs_f_0.scatter_(1, int_indices[:, :, 0, :, :, :],
                                               torch.mul(composed_frac_parts[:, 0, :, :, :], mag))

            theta_lower_ind_fracs_f_1.scatter_(1, int_indices[:, :, 0, :, :, :],
                                               torch.mul(composed_frac_parts[:, 1, :, :, :], mag))

            theta_upper_ind_fracs_f_2.scatter_(1, int_indices[:, :, 1, :, :, :],
                                               torch.mul(composed_frac_parts[:, 2, :, :, :], mag))

            theta_upper_ind_fracs_f_3.scatter_(1, int_indices[:, :, 1, :, :, :],
                                               torch.mul(composed_frac_parts[:, 3, :, :, :], mag))

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

            # freeing up unused tensors from memory
            n, c, d, h, w = x.shape
            del out
            del x
            del int_indices
            del composed_frac_parts
            del mag
            del phi
            del theta
            torch.cuda.empty_cache()

            t = torch.cuda.get_device_properties(0).total_memory
            c = torch.cuda.memory_cached(0)
            a = torch.cuda.memory_allocated(0)
            f = c - a  # free inside cache

            # bin assignment
            out_plus_bins = torch.zeros(  # torch.Size([1, 8, 8, 195, 231, 195])
                (n, self.theta_bins, self.phi_bins, d + 2 * offset - 2, h + 2 * offset - 2, w + 2 * offset - 2),
                dtype=torch.float, device=theta_upper_ind_fracs_f_3.device)

            # assigning low t x low p
            out_plus_bins.scatter_(2, low_p_ordered_by_low_t, theta_lower_ind_fracs_f_0)
            out_plus_bins.scatter_add_(2, upper_p_ordered_by_low_t, theta_lower_ind_fracs_f_1)
            out_plus_bins.scatter_add_(2, lower_p_ordered_by_upper_t, theta_upper_ind_fracs_f_2)
            out_plus_bins.scatter_add_(2, upper_p_ordered_by_upper_t, theta_upper_ind_fracs_f_3)

            out_plus_bins = torch.reshape(out_plus_bins, (n, self.theta_bins * self.phi_bins,
                                                          d + 2 * offset - 2, h + 2 * offset - 2, w + 2 * offset - 2))

            return self.pooler(out_plus_bins)

    def Get_block_size(self):
        return self.block_size


class HOGExtractorGPU(fltr.Filter):

    def __init__(self, image: sitk.Image):
        super().__init__()
        self.hog_module = SimpleHOGModule(image)

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        # Cast the image to a Pytorch tensor
        image_arr = torch.from_numpy(sitk.GetArrayFromImage(image))

        # Compute the 3D-HOG features using Pytorch
        features = self.hog_module(image_arr)

        # Detach the features from the computational graph, write the memory to the RAM and
        # cast the features to be a np.ndarray
        features_np = features.detach().cpu().numpy()

        del features
        torch.cuda.empty_cache()

        features_np = np.squeeze(features_np)
        features_np = np.transpose(features_np, (1, 2, 3, 0))

        image_size = image.GetSize()
        offset = self.hog_module.Get_block_size() // 2

        features_np = features_np[offset:image_size[2] + offset,
                      offset:image_size[1] + offset,
                      offset:image_size[0] + offset]

        img_out = sitk.GetImageFromArray(features_np)
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


'''
testing the HOG feature extractor
'''
# ------------ running and saving feature .nii.gz image -----------------
# path1 = 'C:/Users/afons/PycharmProjects/MIAlab project/data/train/116524/T1native.nii.gz'
# image1 = sitk.ReadImage(path1, sitk.sitkFloat32)
# # image1 = load_image(path1, False)
# image1_np = sitk.GetArrayFromImage(image1)
# image1 = sitk.GetImageFromArray(image1_np[:179, :, :])
# print(image1.GetSize())
# # new dimensions x, y, z = (181, 217, 179)
# hog_extractor = HOGExtractorGPU(image1)
# image_out = hog_extractor.execute(image1)

# --------------------------------------------------------------------
# file_name = 'image_hog_final_3d_avg.nii.gz'
# sitk.WriteImage(sitk.RescaleIntensity(image_out), file_name)
# ----------------------------------------------------------------------

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
# # creating a gif
# images = []
# for filename in file_names:
#     images.append(imageio.imread(os.path.join(new_dir_path, filename)))
# imageio.mimsave(os.path.join(new_dir_path, 'video.gif'), images)
