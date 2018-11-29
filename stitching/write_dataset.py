#!/usr/bin/env python3
"""
"""


import imageio
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import ipdb


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# TODO: Factor out magic numbers here
def get_patches(image_np):
    dim_pad = []
    step_size = 45
    for ind, dim in enumerate(image_np.shape[:-1]):
        if dim < step_size * 2:
            dim_pad.append((0, (step_size * 2) - dim))
        else:
            dim_pad.append((0, step_size - (dim % step_size)))
    dim_pad.append((0, 0))
    image_np = np.pad(image_np, pad_width=dim_pad, mode='constant', constant_values=255)
    marker_j, marker_i = step_size, step_size
    marker_j1, marker_i1 = step_size + 37, step_size + 37
    slices = []
    slices.append(image_np[:53, :53])
    slices.append(image_np[37:90, 37:90])
    slices.append(image_np[37:90, :53])
    slices.append(image_np[:53, 37:90])
    marker_i, marker_j = 90, 90
    marker_i1, marker_j1 = 135, 135
    while marker_j1 <= image_np.shape[0]:
        slices.append(image_np[marker_j-8:marker_j1, :53])
        marker_j += 45
        marker_j1 += 45
    marker_j, marker_j1 = 90, 135
    while marker_i1 <= image_np.shape[1]:
        slices.append(image_np[:53, marker_i-8:marker_i1])
        marker_i += 45
        marker_i1 += 45
    marker_i = 90 
    marker_i1 = 135
    while marker_i1 <= image_np.shape[1]:
        marker_j = 90
        marker_j1 = 135
        while marker_j1 <= image_np.shape[0]:
            patch = image_np[marker_j-8:marker_j1, marker_i-8:marker_i1]
            slices.append(patch)
            marker_j += step_size
            marker_j1 += step_size
        marker_i += step_size
        marker_i1 += step_size

    return slices, image_np


# TODO: Factor our magic numbers here
# This is some ugly code
def stitch_image(merge_buffer, image_name):
    pred_images = []
    input_images = []
    height = None
    width = None
    for buffer in merge_buffer:
        for buffer_item in buffer[0]:
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
        marker_i = 45 * i - 8
        marker_j = 45 * j - 8
        marker_i1 = 45 * (i + 1) 
        marker_j1 = 45 * (j + 1) 
        if i == 0 and j == 0:
            final_predicted_image[0, :53, :53] = image
            j += 1
            continue
        if i == 0 and j == 1:
            top_image = image[:16, :]
            bottom_image = image[16:, :]
            final_predicted_image[1, 37:53, :53] = top_image
            final_predicted_image[0, 53:90, :53] = bottom_image
            j += 1
            continue
        if i == 0:
            top_image = image[:8, :]
            bottom_image = image[8:, :]
            final_predicted_image[1, marker_j:marker_j+8, :53] = top_image
            final_predicted_image[0, marker_j+8:marker_j1, :53] = bottom_image
            if marker_j1 == final_predicted_image.shape[1]:
                j = 0
                i += 1
            else:
                j += 1
            continue
        if i == 1 and j == 0:
            top_left_image = image[:-16, :16]
            bottom_left_image = image[-16:, :16]
            right_image = image[:, 16:]
            final_predicted_image[1, :37, 37:53] = top_left_image
            final_predicted_image[2, 37:53, 37:53] = bottom_left_image
            final_predicted_image[0, :53, 53:90] = right_image
            j += 1
            continue
        if i == 1 and j == 1:
            top_left_image = image[:16, :16]
            middle_left_image = image[16:-8, :16]
            bottom_left_image = image[-8:, :16]
            top_right_image = image[:16, 16:]
            bottom_right_image = image[16:, 16:]
            final_predicted_image[3, 37:53, 37:53] = top_left_image
            final_predicted_image[1, 53:82, 37:53] = middle_left_image
            final_predicted_image[2, 82:90, 37:53] = bottom_left_image
            final_predicted_image[1, 37:53, 53:90] = top_right_image
            final_predicted_image[0, 53:90, 53:90] = bottom_right_image
            j += 1
            continue
        if i == 1 and marker_j1 == final_predicted_image.shape[1]:
            top_left_image = image[:8, :16]
            bottom_left_image = image[8:, :16]
            top_right_image = image[:8, 16:]
            bottom_right_image = image[8:, 16:]
            final_predicted_image[3, marker_j:marker_j+8, 37:53] = top_left_image
            final_predicted_image[1, marker_j+8:, 37:53] = bottom_left_image
            final_predicted_image[1, marker_j:marker_j+8, 53:90] = top_right_image
            final_predicted_image[0, marker_j+8:, 53:90] = bottom_right_image
            i += 1
            j = 0
            continue
        if j == 0:
            top_left_image = image[:-16, :8]
            bottom_left_image = image[-16:, :8]
            right_image = image[:, 8:]
            final_predicted_image[1, :37, marker_i:marker_i+8] = top_left_image
            final_predicted_image[2, 37:53, marker_i:marker_i+8] = bottom_left_image
            final_predicted_image[0, :53, marker_i+8:marker_i1] = right_image
            j += 1
            continue
        if j == 1:
            top_left_image = image[:16, :8]
            middle_left_image = image[16:-8, :8]
            bottom_left_image = image[-8:, :8]
            top_right_image = image[:16, 8:]
            bottom_right_image = image[16:, 8:]
            final_predicted_image[3, marker_j:marker_j+16, marker_i:marker_i+8] = top_left_image
            final_predicted_image[1, marker_j+16:marker_j1-8, marker_i:marker_i+8] = middle_left_image
            final_predicted_image[2, marker_j1-8:marker_j1, marker_i:marker_i+8] = bottom_left_image
            final_predicted_image[1, marker_j:marker_j+16, marker_i+8:marker_i1] = top_right_image
            final_predicted_image[0, marker_j+16:marker_j1, marker_i+8:marker_i1] = bottom_right_image
            j += 1
            continue
        if marker_j1 == final_predicted_image.shape[1]:
            top_left_image = image[:8, :8]
            bottom_left_image = image[8:, :8]
            top_right_image = image[:8, 8:]
            bottom_right_image = image[8:, 8:]
            final_predicted_image[3, marker_j:marker_j+8, marker_i:marker_i+8] = top_left_image
            final_predicted_image[1, marker_j+8:, marker_i:marker_i+8] = bottom_left_image
            final_predicted_image[1, marker_j:marker_j+8, marker_i+8:marker_i1] = top_right_image
            final_predicted_image[0, marker_j+8:, marker_i+8:marker_i1] = bottom_right_image
            j = 0
            i += 1
            continue
        top_left_image = image[:8, :8]
        middle_left_image = image[8:-8, :8]
        bottom_left_image = image[-8:, :8]
        top_right_image = image[:8, 8:]
        bottom_right_image = image[8:, 8:]
        if final_predicted_image[0, 0, marker_i:marker_i+8].shape[0] == 0:
            ipdb.set_trace()
        final_predicted_image[3, marker_j:marker_j+8, marker_i:marker_i+8] = top_left_image
        final_predicted_image[1, marker_j+8:marker_j1-8, marker_i:marker_i+8] = middle_left_image
        final_predicted_image[2, marker_j1-8:marker_j1, marker_i:marker_i+8] = bottom_left_image
        final_predicted_image[1, marker_j:marker_j+8, marker_i+8:marker_i1] = top_right_image
        final_predicted_image[0, marker_j+8:marker_j1, marker_i+8:marker_i1] = bottom_right_image
        j += 1

    sample_final_layer = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            # Generate a sequence from the final_predicted_image
            seq = np.random.choice(final_predicted_image[:, i, j], 4, replace=False)
            # Choose the first number that's not -1
            for val in seq:
                if val != -1:
                    sample_final_layer[i, j] = val

    plt.imshow(sample_final_layer)
    plt.savefig('stitch_results/{}'.format(image_name.decode()))



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
                'name': _bytes_feature(png.encode('utf-8'))
            }))
            writer.write(example.SerializeToString())
    return tfrecords_filename


