#!/usr/bin/env python3
"""
Run a trained model on a dataset
"""

import tensorflow as tf
print("TF version:", tf.__version__)
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import imageio
import matplotlib.pyplot as plt
import network
slim = tf.contrib.slim
import os
import argparse
import json
from preprocessing.read_data import tf_record_parser, scale_image_with_crop_padding
from preprocessing import training
from metrics import *
from bounding_boxes.bound import get_bounds
from dataset.write_dataset import write_dataset


parser = argparse.ArgumentParser()
envarg = parser.add_argument_group('Eval params')
envarg.add_argument("--model_id", default='best', type=str, help="Model id name to be loaded.")
envarg.add_argument("--datadir", default='dataset/data/', type=str, help="Directory to images")
input_args = parser.parse_args()
log_folder = './tboard_logs'

with open(log_folder + '/' + input_args.model_id + '/train/data.json', 'r') as fp:
    args = json.load(fp)


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    _getattr__ = dict.get
    _setattr__ = dict.__setitem__
    _delattr__ = dict.__delitem__


args = Dotdict(args)
class_dict = {0: 'background', 1: 'figure', 2: 'table', 3: 'equation', 4: 'text'}
class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255
LOG_FOLDER = './tboard_logs'
TEST_DATASET_DIR="./dataset/tfrecords"
TEST_FILE = 'test.tfrecords'

processf = write_dataset(args.datadir)


test_filenames = []
test_dataset = tf.data.TFRecordDataset(processf)


def parse_record(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
        "height": tf.FixedLenFeature((), tf.int64),
        "width": tf.FixedLenFeature((), tf.int64),
        "name": tf.FixedLenFeature((), tf.string, default_value="")
    }

    features = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    name = tf.cast(features['name'], tf.string)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    # reshape input and annotation images
    image = tf.reshape(image, (height, width, 3), name="image_reshape")
    return tf.to_float(image), (height, width), name


test_dataset = test_dataset.map(parse_record)  # Parse the record into tensors.
test_dataset = test_dataset.batch(args.batch_size)
iterator = test_dataset.make_one_shot_iterator()
batch_images_tf, batch_shapes_tf, batch_names_tf = iterator.get_next()
logits_tf = network.deeplab_v3(batch_images_tf, args, is_training=False, reuse=False)
predictions_tf = tf.argmax(logits_tf, axis=3)
probabilities_tf = tf.nn.softmax(logits_tf)
train_folder = os.path.join(log_folder, input_args.model_id, "train")
saver = tf.train.Saver()


def merge_image(merge_buffer):

    pass


with tf.Session() as sess:
    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, os.path.join(train_folder, 'model.ckpt'))
    merge_buffer = []
    while True:
        try:
            batch_images_np, batch_predictions_np, batch_shapes_np, batch_names_np = \
                sess.run([batch_images_tf, predictions_tf,  batch_shapes_tf, batch_names_tf])
            buff_names, buff_item = set(), []
            for i in range(batch_predictions_np.shape[0]):
                pred_image = batch_predictions_np[i]
                input_image = batch_images_np[i]
                heights, widths = batch_shapes_np
                pred_image = np.reshape(pred_image, (53, 53))
                input_image = np.reshape(input_image, (53, 53, 3))
                image_name = batch_names_np[i]
                if image_name not in buff_names:
                    buff_names[image_name] = 1
                buff_item.append((pred_image, input_image, image_name, heights[i], widths[i]))
            merge_buffer.append((buff_item, buff_names))
            if len(merge_buffer) > 1:
                # Get the buffer name of the first item
                check_name = merge_buffer[0][1].keys()[-1]
                # Check the last item
                if check_name != merge_buffer[-1][1].keys()[-1]:
                    merge_image(merge_buffer, check_name)
                    merge_buffer = [merge_buffer[-1]]

        except tf.errors.OutOfRangeError:
            # Clear the merge buffer
            if len(merge_buffer) > 0:
                merge_image(merge_buffer, merge_buffer[0][1].keys()[-1])
            break

