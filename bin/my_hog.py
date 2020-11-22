import sys
import time

import SimpleITK as sitk
import numpy as np
from exercise.exercise_simpleitk import load_image
import scipy.ndimage.filters as fltrs
import matplotlib.pyplot as plt
import math

path1 = 'C:/Users/afons/PycharmProjects/MIAlab project/data/train/116524/T1native.nii.gz'
image1 = load_image(path1, False)


def hog_features(image: sitk.Image, theta_bins=8, phi_bins=8, block_size=16, feature_vec=(-1, 0, 1),
                 block_displacement=16):
    """Calculates the HOG-features

    Args:
        values (np.array): THe values to calculate the HOG-features from.

    Returns:
        np.array: A vector containing the HOG-features
    """
    img_arr = sitk.GetArrayFromImage(image)
    or_z, or_y, or_x = img_arr.shape
    print(img_arr.shape)

    #   padding image in case block size doesn't fit
    z_offset, y_offset, x_offset = 0, 0, 0
    for i in range(len(img_arr.shape)):
        if (img_arr.shape[i] % block_size) != 0 and i == 0:
            z_offset = block_size - (img_arr.shape[i] % block_size)
        elif (img_arr.shape[i] % block_size) != 0 and i == 1:
            y_offset = block_size - (img_arr.shape[i] % block_size)
        elif (img_arr.shape[i] % block_size) != 0 and i == 2:
            x_offset = block_size - (img_arr.shape[i] % block_size)

    pad = ((0, z_offset), (0, y_offset), (0, x_offset))
    img_arr_padded = np.pad(img_arr, pad, 'symmetric')
    z_dim, y_dim, x_dim = img_arr_padded.shape
    print(' padded array size = ' + str(img_arr_padded.shape))
    #   creating the output image
    img_out = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, theta_bins * phi_bins)
    img_out_arr = sitk.GetArrayFromImage(img_out)
    # print(img_out.GetNumberOfComponentsPerPixel())

    #   calculating image 3D gradient
    # convolution axis: 0, 1, 2 = z, y, x
    img_conv_z = fltrs.convolve1d(img_arr_padded, weights=feature_vec, axis=0, mode='reflect')
    img_conv_y = fltrs.convolve1d(img_arr_padded, weights=feature_vec, axis=1, mode='reflect')
    img_conv_x = fltrs.convolve1d(img_arr_padded, weights=feature_vec, axis=2, mode='reflect')

    nr_block_z = z_dim // block_displacement  # all ints
    nr_block_y = y_dim // block_displacement
    nr_block_x = x_dim // block_displacement

    print('-' * 5)
    print(nr_block_z)
    print(nr_block_x)
    print(nr_block_y)
    print('-' * 5)

    # 2D image test (to be removed)
    img_test_1 = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, 1)
    img_test_1_arr = sitk.GetArrayFromImage(img_test_1)
    print(img_test_1_arr.shape)

    for z in range(nr_block_z - 1):
        print(' z = ' + str(z))
        for y in range(nr_block_y - 1):
            start = time.perf_counter()
            for x in range(nr_block_x - 1):
                #   starting image coordinates for histogram extraction
                cord_z = int(block_displacement * z)
                cord_y = int(block_displacement * y)
                cord_x = int(block_displacement * x)
                # print('z = ' + str(cord_z) + ',' + 'y = ' + str(cord_y) + ',' + 'x = ' + str(cord_x))
                img_out_arr[cord_z, cord_y, cord_x] \
                    = block_hog_extractor(img_conv_z, img_conv_y, img_conv_x, block_size, theta_bins, phi_bins,
                                          cord_z, cord_y, cord_x)

                # 2D image test (to be removed), z = 0 slice
                img_test_1_arr[cord_z, cord_y, cord_x] = img_out_arr[cord_z, cord_y, cord_x][0]
                # print(img_test_1_arr[cord_z, cord_y, cord_x])

            end = time.perf_counter()
            print(f'Finished in {round(end - start, 2)} second(s)')
        break
    # 2D image test (to be removed), z = 0 slice
    plt.figure()
    plt.imshow(img_test_1_arr[0])
    plt.show()

    img_out = sitk.GetImageFromArray(img_out_arr)
    img_out.CopyInformation(image)

    return image


def block_hog_extractor(img_conv_z, img_conv_y, img_conv_x, block_size, theta_bins, phi_bins, z, y, x):
    # z, y, x are the starting coordinates of each block on the overall image
    # img_out = np.zeros(img_conv_x.shape + (3,))  # img_out has dimensions (z,y,x,3)
    h_bin_set = np.zeros((theta_bins, phi_bins))
    # print('z = ' + str(z) + ',' + 'y = ' + str(y) + ',' + 'x = ' + str(x))
    # print(x + block_size)
    eps = sys.float_info.epsilon
    for zz in range(z, z + block_size - 1):     # these indices are right
        for yy in range(y, y + block_size - 1):
            for xx in range(x, x + block_size - 1):
                # print(xx)
                #   magnitude
                r = \
                    np.sqrt(img_conv_x[zz, yy, xx] ** 2 + img_conv_y[zz, yy, xx] ** 2 + img_conv_z[zz, yy, xx] ** 2)
                # img_out[zz, yy, xx, 0] = r
                #   theta
                theta = \
                    np.math.atan(img_conv_y[zz, yy, xx] / (img_conv_x[zz, yy, xx] + eps))
                # img_out[zz, yy, xx, 1] = theta
                #   phi
                # if theta == 0:
                #     theta = eps
                phi = \
                    np.math.acos(img_conv_z[zz, yy, xx] / (r + eps))
                # if phi == 0:
                #     phi = eps
                # img_out[zz, yy, xx, 2] = phi
                h_bin_set = bin_assignment(theta_bins, phi_bins, r, theta, phi, h_bin_set)
    h_bin_set = np.matrix.flatten(h_bin_set)
    return h_bin_set


def bin_assignment(theta_bins, phi_bins, r, theta, phi, h_bin_set):
    # defining theta and phi bins' range
    theta_bin_len = (2 * np.math.pi) / theta_bins  # bc the last bin cannot be 2*pi, the same as 0
    phi_bin_len = np.math.pi / phi_bins

    # bin assignment
    theta_split = True
    phi_split = True
    # theta
    theta = angle_normalizer(theta, 0, 2 * np.pi)
    theta_raw_index = theta / theta_bin_len
    low_t_bin_index = int(np.floor(theta_raw_index))

    # in case theta index > max index, go back to origin
    if low_t_bin_index == theta_bins:
        low_t_bin_index = 0

    if np.modf(theta_raw_index)[0] <= 0.5:
        low_t_bin_ratio = np.modf(theta_raw_index)[0]
        high_t_bin_ratio = 1 - np.modf(theta_raw_index)[0]
    elif np.modf(theta_raw_index)[0] > 0.5:
        low_t_bin_ratio = 1 - np.modf(theta_raw_index)[0]
        high_t_bin_ratio = np.modf(theta_raw_index)[0]
    else:
        theta_split = False

    # phi
    phi = angle_normalizer(phi, 0, np.pi)
    phi_raw_index = phi / phi_bin_len
    low_p_bin_index = int(np.floor(phi_raw_index))

    # in case phi index > max index, go back to origin
    if low_p_bin_index == phi_bins:
        low_p_bin_index = 0

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
    # 4 possible 2D histogram splitting cases
    # when both phi and theta are fit in 1 exact bin
    if not theta_split and not phi_split:

        # if low_t_bin_index == theta_bins and low_p_bin_index == phi_bins:
        #     low_t_bin_index = 0
        #     low_p_bin_index = 0
        h_bin_set[low_t_bin_index][low_p_bin_index] += r

    # when only theta is split in 2
    elif theta_split and not phi_split:

        # in case theta is split between last and origin bin
        if low_t_bin_index == theta_bins - 1:
            high_t_bin_index = 0
        else:
            high_t_bin_index = low_t_bin_index + 1

        h_bin_set[low_t_bin_index][low_p_bin_index] += r * low_t_bin_ratio
        h_bin_set[high_t_bin_index][low_p_bin_index] += r * high_t_bin_ratio

    # when only phi is split in 2
    elif phi_split and not theta_split:

        # in case phi is split between last and origin bin
        if low_p_bin_index == phi_bins - 1:
            high_p_bin_index = 0
        else:
            high_p_bin_index = low_p_bin_index + 1

        h_bin_set[low_t_bin_index][low_p_bin_index] += r * low_p_bin_ratio
        h_bin_set[low_t_bin_index][high_p_bin_index] += r * high_p_bin_ratio

    # when both phi and theta are split in a 2x2 histogram "block"
    else:
        # in case theta is split between last and origin bin
        if low_t_bin_index == theta_bins - 1:
            high_t_bin_index = 0
        else:
            high_t_bin_index = low_t_bin_index + 1

        # in case phi is split between last and origin bin
        if low_p_bin_index == phi_bins - 1:
            high_p_bin_index = 0
        else:
            high_p_bin_index = low_p_bin_index + 1

        # theta wise splitting
        h_bin_set[low_t_bin_index][low_p_bin_index] += r * low_t_bin_ratio * low_p_bin_ratio
        h_bin_set[high_t_bin_index][low_p_bin_index] += r * high_p_bin_ratio * low_p_bin_ratio
        # phi wise splitting
        h_bin_set[low_t_bin_index][high_p_bin_index] += r * low_t_bin_ratio * high_p_bin_ratio
        h_bin_set[high_t_bin_index][high_p_bin_index] += r * high_p_bin_ratio * high_p_bin_ratio

    return h_bin_set


number = 15.5
newn = np.floor(number)
newn = np.int64(newn)


# print(type(newn))


def angle_normalizer(angle, lower_bound, upper_bound):
    #   for angles in radians
    newAngle = angle
    while newAngle < lower_bound:
        newAngle += 2 * np.pi
    while newAngle > upper_bound:
        newAngle -= 2 * np.pi
    return newAngle


# number += 10
# n = np.modf(number)
# print(n[0])  # 0.5
# print(np.modf(number)[0])
# print(n[1])  # 15.0
# tuple1 = (1, 2, 3)
# tuple2 = (4,)
# tuplex = sum((tuple1, tuple2), ())
# print(tuplex)

hog_features(image1, block_displacement=8, theta_bins=7, phi_bins=7)

# print(101 % 10)
