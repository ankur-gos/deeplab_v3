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
    i = 1
    x_lower, x_higher = 0, 53
    j = 1
    y_lower, y_higher = 0, 53
    slices = []
    while i <= image_np.shape[0]:
        while step_size * j <= image_np.shape[1]:
            patch = image_np[x_lower:x_higher, y_lower:y_higher, :]
            print(patch.shape)
            slices.append(patch)
            j += 1
            y_higher = j * step_size
            y_lower = y_higher - 53
        j = 1
        y_lower, y_higher = 0, 53
        i += 1
        x_higher = i * step_size
        x_lower = x_higher - 53
    return slices, image_np


# TODO: Factor our magic numbers here
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
    final_predicted_image = None
    i = 0
    j = 0
    # Initialize to a non class nd array
    final_predicted_image = np.full((4, height, width), -1)
    layer_index = {}
    for image in pred_images:
        if i == 0 and j == 0:
            final_predicted_image[0, :53, :53] = image
            j += 1
            continue
        if i == 0:
            top_image = image[:16, :]
            bottom_image = image[16:, :]
            final_predicted_image[1, marker_j-8:marker_j+8, :53] = top_image
            final_predicted_image[0, marker_j+8:, :53] = bottom_image
            if (j + 1) * 45 == final_predicted_image.shape[1]:
                j = 0
                i += 1
            else:
                j += 1
            continue
        if j == 0:
            top_left_image = image[:37, :16]
            bottom_left_image = image[37:, :16]
            right_image = image[:, 16:]
            final_predicted_image[1, :]


        marker_i = i * 45
        marker_j = j * 45

        # The current position to add to is i,j
        current_pos = (i, j)
        # Divide the matrix
        left = final_predicted_image[0:37, :]
        right = final_predicted_image[37:, :]
        left_n = image[0:16, :]
        right_n = image[16:, :]
        stacked_left = np.stack([left, left])
        stacked_middle = np.stack([right, left_n], axis=0)
        stacked_right = np.stack([right_n, right_n])
        # TODO Continue this




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


