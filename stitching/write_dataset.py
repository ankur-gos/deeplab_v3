#!/usr/bin/env python3
"""
"""


import imageio
import tensorflow as tf
import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# TODO: Factor out magic numbers here
def get_patches(image_np):
    dim_pad = []
    step_size = 45
    for dim in image_np.shape[:-1]:
        if dim < step_size * 2:
            dim_pad.append((0, (step_size * 2) - dim))
        else:
            dim_pad.append((0, dim % step_size))
    dim_pad.append((0, 0))
    image_np = np.pad(image_np, pad_width=dim_pad, mode='constant', constant_values=255)
    marker_j, marker_i = step_size, step_size
    marker_j1, marker_i1 = step_size + 37, step_size + 37
    slices = []
    slices.append(image_np[:53, :53])
    while marker_j1 + 8 <= image_np.shape[1]:
        slices.append(image_np[marker_j-8:marker_j1+8, :53])
        marker_j += 37
        marker_j1 += 37
    while marker_i + 8 <= image_np.shape[0]:
        marker_j = step_size
        marker_j1 = step_size + 37
        slices.append(image_np[0:53, marker_i-8:marker_i1+8])
        while marker_j + 8 <= image_np.shape[1]:
            patch = image_np[marker_j-8:marker_j1+8, marker_i-8:marker_i1+8]
            slices.append(patch)
            marker_j += step_size
            marker_j1 += step_size
        marker_i += 37
        marker_i1 += 37

    return slices, image_np


# TODO: Factor our magic numbers here
# This is some ugly code
def stitch_image(merge_buffer, image_name):
    pred_images = []
    input_images = []
    height = None
    width = None
    for buffer in merge_buffer:
        for buffer_item, _ in buffer:
            pred_image, input_image, input_name, image_height, image_width = buffer_item
            if input_name != image_name:
                break
            height = image_height
            width = image_width
            pred_images.append(pred_image)
            input_images.append(input_image)
    i = 0
    j = 0
    # Initialize to a non class nd array
    final_predicted_image = np.full((4, height, width), -1)
    for image in pred_images:
        marker_i = 45 + (i - 1) * 37
        marker_j = 45 + (j - 1) * 37
        marker_i1 = marker_i + 37
        marker_j1 = marker_j + 37
        if i == 0 and j == 0:
            final_predicted_image[0, :53, :53] = image
            j += 1
            continue
        if i == 0:
            top_image = image[:16, :]
            bottom_image = image[16:, :]
            final_predicted_image[1, marker_j-8:marker_j+8, :53] = top_image
            final_predicted_image[0, marker_j+8:marker_j1+8, :53] = bottom_image
            if marker_j1 == final_predicted_image.shape[1]:
                j = 0
                i += 1
            else:
                j += 1
            continue
        if j == 0:
            top_left_image = image[:-16, :16]
            bottom_left_image = image[-16:, :16]
            right_image = image[:, 16:]
            final_predicted_image[1, :marker_j1-8, marker_i-8:marker_i+8] = top_left_image
            final_predicted_image[2, marker_j1-8:marker_j1+8, marker_i-8:marker_i+8] = bottom_left_image
            final_predicted_image[0, :marker_j1+8, marker_i+8:marker_i1+8] = right_image
            j += 1
            continue
        if marker_j == final_predicted_image.shape[1] - 45:
            top_left_image = image[:16, :16]
            bottom_left_image = image[16:, :16]
            top_right_image = image[:16, 16:]
            bottom_right_image = image[16:, 16:]
            final_predicted_image[3, marker_j-8:marker_j+8, marker_i-8:marker_i+8] = top_left_image
            final_predicted_image[1, marker_j+8:, marker_i-8:marker_i+8] = bottom_left_image
            final_predicted_image[1, marker_j-8:marker_j+8, marker_i+8:marker_i1+8] = top_right_image
            final_predicted_image[0, marker_j+8:, marker_i+8:marker_i1+8] = bottom_right_image
            j = 0
            i += 1
            continue
        top_left_image = image[:16, :16]
        middle_left_image = image[16:-16, :16]
        bottom_left_image = image[-16:, :16]
        top_right_image = image[:16, 16:]
        bottom_right_image = image[16:, 16:]
        final_predicted_image[3, marker_j-8:marker_j+8, marker_i-8:marker_i+8] = top_left_image
        final_predicted_image[1, marker_j+8:marker_j1-8, marker_i-8:marker_i+8] = middle_left_image
        final_predicted_image[2, marker_j1-8:marker_j1+8, marker_i-8:marker_i+8] = bottom_left_image
        final_predicted_image[1, marker_j-8:marker_j+8, marker_i+8:marker_i1+8] = top_right_image
        final_predicted_image[0, marker_j+8:, marker_i+8:marker_i1+8] = bottom_right_image

    sample_final_layer = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            # Generate a sequence from the final_predicted_image
            seq = np.random.choice(final_predicted_image[:, i, j], 4, replace=False)
            # Choose the first number that's not -1
            for val in seq:
                if val != -1:
                    sample_final_layer[i, j] = val

    return sample_final_layer


def write_dataset(image_dir):
    """
    Write the dataset to a tfrecords file
    :param image_dir: image directory
    :return: Filename for the tfrecords file
    """
    tfrecords_filename = 'process.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for png in os.listdir(image_dir):
        image_np = imageio.imread(os.path.join(image_dir, png))
        patches, image_np_pad = get_patches(image_np)
        for patch in patches:
            image_raw = patch.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_np_pad.shape[0]),
                'width': _int64_feature(image_np_pad.shape[1]),
                'image_raw': _bytes_feature(image_raw),
                'name': _bytes_feature(png)
            }))
            writer.write(example.SerializeToString())
    return tfrecords_filename


