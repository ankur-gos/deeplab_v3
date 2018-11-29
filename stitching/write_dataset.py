#!/usr/bin/env python3
"""
"""


import imageio
import tensorflow as tf
from skimage.util.shape import view_as_blocks
import os
import numpy as np
import matplotlib.pyplot as plt
import ipdb


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_blocks(image_np):
    dim_pad = []
    # Make sure the image can be split evenly
    for ind, dim in enumerate(image_np.shape[:-1]):
        dim_pad.append((0, 513 - (dim % 513)))
    dim_pad.append((0, 0))
    image_np = np.pad(image_np, pad_width=dim_pad, mode='constant', constant_values=255)
    blocks = view_as_blocks(image_np, block_shape=(513, 513, 3))
    blocks = blocks.squeeze()
    block_list = []
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block_list.append(blocks[i, j])
    return block_list, image_np


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

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def concat_image(merge_buffer, image_name):
    pred_images = []
    input_images = []
    height = None
    width = None
    o_h = None
    o_w = None
    new_buffer = []
    f = 0
    for ind, buffer in enumerate(merge_buffer):
        for ind2, buffer_item in enumerate(buffer[0]):
            pred_image, input_image, input_name, image_height, image_width, o_height, o_width = buffer_item
            if input_name != image_name:
                if ind2 != 0:
                    if image_name not in buffer[1]:
                        ipdb.set_trace()
                    del buffer[1][image_name]
                new_buffer.append((buffer[0][ind2:], buffer[1]))
                if ind + 1 != len(merge_buffer):
                    new_buffer = new_buffer + merge_buffer[ind+1:]
                f = 1
                break
            height = image_height
            width = image_width
            o_h = o_height
            o_w = o_width
            pred_images.append(pred_image)
            input_images.append(input_image)
        if f == 1:
            break
    if height is None or width is None:
        ipdb.set_trace()
    final_predicted_image = np.full((height, width), -1)
    marker_i = 0
    marker_j = 0
    for image in pred_images:
        if marker_j == width:
            marker_j = 0
            marker_i += 513
        final_predicted_image[marker_i:marker_i+513, marker_j:marker_j+513] = image
        marker_j += 513


    final_input_image = np.full((height, width, 3), -1)
    marker_i = 0
    marker_j = 0
    for image in input_images:
        if marker_j == width:
            marker_j = 0
            marker_i += 513
        final_input_image[marker_i:marker_i+513, marker_j:marker_j+513] = image
        marker_j += 513

    final_predicted_image = final_predicted_image[:o_h, :o_w]
    final_input_image = final_input_image[:o_h, :o_w]

    plt.figure(figsize=(8.5, 11))
    plt.imshow(final_input_image, cmap=discrete_cmap(4, 'cubehelix'))
    plt.colorbar()
    plt.savefig('stitch_results/input_{}'.format(image_name.decode()), dpi=300)
    
    plt.imshow(final_predicted_image, cmap=discrete_cmap(4, 'cubehelix'))
    plt.colorbar()
    plt.savefig('stitch_results/predict_{}'.format(image_name.decode()), dpi=300)
    plt.close()
    return new_buffer, final_predicted_image, final_input_image
    
        
def draw_bounding_boxes(bounding_boxes, input_image, image_name):
    # Really unnecessary copy but I'm lazy and don't want to think
    predicted_boxes = input_image.copy()
    colors = [np.array([128, 0, 0]), np.array([170, 110, 40]), np.array([230, 190, 255]),
              np.array([70, 240, 240]), np.array([200, 50, 200])]
    for cl in bounding_boxes:
        if cl == 0:
            continue
        bxs = bounding_boxes[cl]
        for coords in bxs:
            tl_x, tl_y = coords[0]
            br_x, br_y = coords[1]
            coord_color = colors[cl]
            predicted_boxes[tl_x:br_x + 1, tl_y-2:tl_y+2] = coord_color
            predicted_boxes[tl_x:br_x + 1, br_y-2:br_y+2] = coord_color
            predicted_boxes[tl_x-2:tl_x+2, tl_y:br_y+1] = coord_color
            predicted_boxes[br_x-2:br_x+2, tl_y:br_y+1] = coord_color
    plt.figure(figsize=(8.5, 11))
    plt.imshow(predicted_boxes.astype(np.uint8), aspect='auto', cmap=discrete_cmap(4, 'cubehelix'))
    plt.colorbar()
    plt.savefig('stitch_results/bb_{}'.format(image_name.decode()), dpi=300)
    plt.close()




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
        patches, image_np_pad = get_blocks(image_np)
        for patch in patches:
            image_raw = patch.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'original_height': _int64_feature(image_np.shape[0]),
                'original_width': _int64_feature(image_np.shape[1]),
                'height': _int64_feature(image_np_pad.shape[0]),
                'width': _int64_feature(image_np_pad.shape[1]),
                'image_raw': _bytes_feature(image_raw),
                'name': _bytes_feature(png.encode('utf-8'))
            }))
            writer.write(example.SerializeToString())
    return tfrecords_filename


