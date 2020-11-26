import os
import sys
import time
import shutil
import SimpleITK as sitk
import numpy as np
from exercise.exercise_simpleitk import load_image
import scipy.ndimage.filters as fltrs
import matplotlib.pyplot as plt
import pymia.filtering.filter as fltr
import imageio


class HOG_extractor(fltr.Filter):
    """Represents a 3D HOG (Histogram of Oriented Gradients extractor) filter, which works on a neighborhood."""

    def __init__(self, theta_bins=8, phi_bins=8, block_size=15, block_displacement=15):

        """Initializes a new instance of the Hog_3D_extractor class."""

        super().__init__()
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins
        self.block_size = block_size
        self.feature_vec = np.array([-1, 0, 1])
        self.block_displacement = block_displacement
        self.output_convolved_images = []
        #   these lens are the ranges of angle values that each histogram bin can receive
        self.theta_bin_len = (2 * np.math.pi) / self.theta_bins
        self.phi_bin_len = np.math.pi / self.phi_bins
        assert is_odd(block_size), "block size should be odd"

    def arr_convolver(self, img_arr, conv_mode='reflect'):
        # padding the image to include the starting and ending boundaries of the array
        # convolution axes: z, y, x = 0, 1, 2
        offset = self.block_size // 2
        pad_start = ((offset, 0), (offset, 0), (offset, 0))
        pad_end = ((0, offset), (0, offset), (0, offset))
        for i in range(3):
            image_conv = fltrs.convolve1d(img_arr, weights=self.feature_vec, axis=i, mode=conv_mode)
            image_conv_start = np.pad(image_conv, pad_start, 'symmetric')
            image_conv_end = np.pad(image_conv_start, pad_end, 'symmetric')
            self.output_convolved_images.append(image_conv_end)

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Calculates the HOG-features for a given image.

        Args:
            image (sitk.Image): the image from which to apply HOG

        Returns:
            np.array: A vector containing the HOG-features
            :param image:
            :param params:
        """
        img_arr = sitk.GetArrayFromImage(image)
        print('original image size = ' + str(img_arr.shape))

        #   creating the output image
        img_out = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, self.theta_bins * self.phi_bins)
        img_out_arr = sitk.GetArrayFromImage(img_out)

        #   calculating image 3D gradient
        self.arr_convolver(img_arr)

        # calculating nr of blocks for histogram extraction
        blocks_zyx = nr_of_blocks(img_arr.shape, self.block_size, self.block_displacement)

        # 2D image test (to be removed)
        img_test_1 = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, self.theta_bins * self.phi_bins)
        img_test_1_arr = sitk.GetArrayFromImage(img_test_1)

        #   cord_z, cord_y, cord_x = are the starting-window-coordinates of the padded image
        # and correspond to the central window coordinates on the original image
        print('.' * 5 + ' processing brain slice z = ' + str(int(7 * self.block_displacement)))
        print('      (' + str(blocks_zyx[1]) + ' blocks to process)')
        slice_val = 105
        for z in range(slice_val, slice_val + 1):  # range(blocks_zyx[0):
            cord_z = int(self.block_displacement * z)

            for y in range(blocks_zyx[1]):
                cord_y = int(self.block_displacement * y)
                start = time.perf_counter()

                for x in range(blocks_zyx[2]):
                    cord_x = int(self.block_displacement * x)

                    img_out_arr[cord_z, cord_y, cord_x] = self.neighborhood_hog_extractor(cord_z, cord_y, cord_x)

                    # 2D image test (to be removed), z ~ 100  slice
                    img_test_1_arr[cord_z, cord_y, cord_x] = img_out_arr[cord_z, cord_y, cord_x]
                    # print(img_test_1_arr[cord_z, cord_y, cord_x])

                end = time.perf_counter()
                print('Finished block ' + str(y) + f' in {round(end - start, 2)} second(s)')
            break

        # 2D image test (to be removed), z ~ 100  slice
        slice_z = int(slice_val * self.block_displacement)
        img_slice = sitk.GetImageFromArray(img_test_1_arr[int(slice_val * self.block_displacement)])
        main_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mia-result/HOG_testing')
        dir_path = os.path.join(main_path, 'size_' + str(self.block_size) + '_disp_' + str(self.block_displacement))
        full_path = os.path.join(dir_path, 'brain_slice_' + str(slice_z) + '_.nii')
        sitk.WriteImage(img_slice, full_path)
        print('HOG extraction finished')

        img_out = sitk.GetImageFromArray(img_out_arr)
        img_out.CopyInformation(image)

        return image

    def neighborhood_hog_extractor(self, z, y, x):
        #   retrieving convolved images
        img_conv_z = self.output_convolved_images[0]
        img_conv_y = self.output_convolved_images[1]
        img_conv_x = self.output_convolved_images[2]
        # z, y, x are the starting coordinates of each block on the overall image
        #   histogram matrix
        h_bin_set = np.zeros((self.theta_bins, self.phi_bins))
        # print('z = ' + str(z) + ',' + 'y = ' + str(y) + ',' + 'x = ' + str(x))
        # print(x + block_size)
        eps = sys.float_info.epsilon
        for zz in range(z, z + self.block_size):
            for yy in range(y, y + self.block_size):
                for xx in range(x, x + self.block_size):
                    # print(xx)
                    #   magnitude
                    r = \
                        np.sqrt(img_conv_x[zz, yy, xx] ** 2 + img_conv_y[zz, yy, xx] ** 2 + img_conv_z[zz, yy, xx] ** 2)
                    #   theta
                    theta = \
                        np.math.atan(img_conv_y[zz, yy, xx] / (img_conv_x[zz, yy, xx] + eps))
                    #   phi
                    phi = \
                        np.math.acos(img_conv_z[zz, yy, xx] / (r + eps))
                    #   updating histogram matrix
                    h_bin_set = self.bin_assignment(r, theta, phi, h_bin_set)

        #   returning histogram as a feature set
        return np.matrix.flatten(h_bin_set)

    def bin_assignment(self, r, theta, phi, h_bin_set):

        theta_split = True
        phi_split = True
        low_t_bin_ratio = None
        high_t_bin_ratio = None
        low_p_bin_ratio = None
        high_p_bin_ratio = None

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

        # print(' angles = ' + str(theta) + ' ' + str(phi))
        # print(' angles = ' + str(theta) + ' ' + str(phi))
        # print(' values = ' + str(low_t_bin_index) + ' ' + str(low_p_bin_index))

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
            h_bin_set[low_t_bin_index][low_p_bin_index] += r * low_t_bin_ratio * low_p_bin_ratio
            h_bin_set[high_t_bin_index][low_p_bin_index] += r * high_p_bin_ratio * low_p_bin_ratio
            # phi-axis wise splitting
            h_bin_set[low_t_bin_index][high_p_bin_index] += r * low_t_bin_ratio * high_p_bin_ratio
            h_bin_set[high_t_bin_index][high_p_bin_index] += r * high_p_bin_ratio * high_p_bin_ratio

        return h_bin_set


def nr_of_blocks(img_arr_shape, block_size, block_displacement):
    #   Calculating nr of blocks from which to extract image histograms
    # maximal coordinates that can be reached (inclusive)
    or_z = img_arr_shape[0] - 1
    or_y = img_arr_shape[1] - 1
    or_x = img_arr_shape[2] - 1
    # starting coordinates
    z, y, x = 0, 0, 0
    # centered starting coordinates in the padded image = origin of the original image
    cent_z, cent_y, cent_x = (block_size // 2), (block_size // 2), (block_size // 2)

    while cent_z <= or_z + (block_size // 2) - block_displacement:
        cent_z += block_displacement
        z += 1
    nr_block_z = z + 1

    while cent_y <= or_y + (block_size // 2) - block_displacement:
        cent_y += block_displacement
        y += 1
    nr_block_y = y + 1

    while cent_x <= or_x + (block_size // 2) - block_displacement:
        cent_x += block_displacement
        x += 1
    nr_block_x = x + 1

    return [nr_block_z, nr_block_y, nr_block_x]


def angle_normalizer(angle, lower_bound, upper_bound):
    """Wraps an angle in radians to a specific range of values. Example:
    angle_normalizer(4*pi, 0, 2*pi) -> 2*pi

    :param angle: the angle in radians to be wrapped
    :param lower_bound: starting range value, inclusive
    :param upper_bound: ending range value, inclusive
    :return: the normalized angle in angles
    """

    newAngle = angle
    while newAngle < lower_bound:
        newAngle += 2 * np.pi
    while newAngle > upper_bound:
        newAngle -= 2 * np.pi
    return newAngle


def is_odd(nr):
    if nr % 2 == 0:
        return False
    else:
        return True


# loading the image
path1 = 'C:/Users/afons/PycharmProjects/MIAlab project/data/train/116524/T1native.nii.gz'
image1 = load_image(path1, False)

# ----- testing the algorithm -----
block_size = 15
block_displacement = 1

# creating a sub-directory for the parameters used
main_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mia-result/HOG_testing')
new_dir_path = os.path.join(main_path, 'size_' + str(block_size) + '_disp_' + str(block_displacement))

# -----------------------comment / uncomment------------------
# # this deletes the dir if exists
# if os.path.exists(new_dir_path):
#     shutil.rmtree(new_dir_path)
# os.makedirs(new_dir_path)
#
# hog_example = HOG_extractor(block_size=block_size, block_displacement=block_displacement)
# hog_example.execute(image1)
# ------------------------comment / uncomment-----------------
#
# getting slice 100
image1_arr = sitk.GetArrayFromImage(image1)
image1_arr_22 = image1_arr[105]
image1_22 = sitk.GetImageFromArray(image1_arr_22)

# loading brain slice image
image = sitk.ReadImage(os.path.join(new_dir_path, 'brain_slice_105_.nii'))
image_arr = sitk.GetArrayFromImage(image)
print(image_arr.shape)

# saving every feature image
file_names = []
for feature_i in range(image_arr.shape[2]):
    saving_img_arr = image_arr[:, :, feature_i]
    saving_img = sitk.GetImageFromArray(saving_img_arr)
    file_name = 'feature{0}.png'.format(feature_i)
    saving_path = os.path.join(new_dir_path, file_name)
    sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(saving_img), sitk.sitkUInt8), saving_path)
    file_names.append(file_name)

# creating a gif
images = []
for filename in file_names:
    images.append(imageio.imread(os.path.join(new_dir_path, filename)))
imageio.mimsave(os.path.join(new_dir_path, 'video.gif'), images)
