import tensorflow as tf
print("TF version:", tf.__version__)
import numpy as np
import matplotlib
import ipdb
matplotlib.use('TkAgg')
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


plt.interactive(False)

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Eval params')
envarg.add_argument("--model_id", default='best', type=str, help="Model id name to be loaded.")
input_args = parser.parse_args()

# best: 16645
model_name = str(input_args.model_id)

# uncomment and set the GPU id if applicable.
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

log_folder = './tboard_logs'

if not os.path.exists(os.path.join(log_folder, model_name, "test")):
    os.makedirs(os.path.join(log_folder, model_name, "test"))

with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)

# 0=background
# 1=figure
# 2=table
# 3=equation
# 4=text

class_dict = {0: 'background', 1: 'figure', 2: 'table', 3: 'equation', 4: 'text'}

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

LOG_FOLDER = './tboard_logs'
TEST_DATASET_DIR="./dataset/tfrecords"
TEST_FILE = 'test.tfrecords'

test_filenames = [os.path.join(TEST_DATASET_DIR,TEST_FILE)]
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)  # Parse the record into tensors.
test_dataset = test_dataset.map(lambda image, annotation, image_shape: scale_image_with_crop_padding(image, annotation, image_shape, args.crop_size))
test_dataset = test_dataset.shuffle(buffer_size=100)
test_dataset = test_dataset.batch(args.batch_size)

iterator = test_dataset.make_one_shot_iterator()
batch_images_tf, batch_labels_tf, batch_shapes_tf = iterator.get_next()

logits_tf = network.deeplab_v3(batch_images_tf, args, is_training=False, reuse=False)

valid_labels_batch_tf, valid_logits_batch_tf = training.get_valid_logits_and_labels(
    annotation_batch_tensor=batch_labels_tf,
    logits_batch_tensor=logits_tf,
    class_labels=class_labels)

cross_entropies_tf = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_logits_batch_tf,
                                                                labels=valid_labels_batch_tf)

cross_entropy_mean_tf = tf.reduce_mean(cross_entropies_tf)
tf.summary.scalar('cross_entropy', cross_entropy_mean_tf)

predictions_tf = tf.argmax(logits_tf, axis=3)
probabilities_tf = tf.nn.softmax(logits_tf)

merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

test_folder = os.path.join(log_folder, model_name, "test")
train_folder = os.path.join(log_folder, model_name, "train")

with tf.Session() as sess:

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    saver.restore(sess, os.path.join(train_folder, "model.ckpt"))
    print("Model", model_name, "restored.")

    mean_IoU = []
    mean_pixel_acc = []
    mean_freq_weighted_IU = []
    mean_acc = []
    mean_class_IU = {}

    j = 0
    while True:
        try:
            batch_images_np, batch_predictions_np, batch_labels_np, batch_shapes_np, summary_string = \
                sess.run([batch_images_tf, predictions_tf, batch_labels_tf, batch_shapes_tf, merged_summary_op])

            heights, widths = batch_shapes_np

            # loop through the images in the batch and extract the valid areas from the tensors
            for i in range(batch_predictions_np.shape[0]):

                label_image = batch_labels_np[i]
                pred_image = batch_predictions_np[i]
                input_image = batch_images_np[i]

                indices = np.where(label_image != 255)
                label_image = label_image[indices]
                pred_image = pred_image[indices]
                input_image = input_image[indices]

                if label_image.shape[0] == 263169:
                    label_image = np.reshape(label_image, (513,513))
                    pred_image = np.reshape(pred_image, (513,513))
                    input_image = np.reshape(input_image, (513,513,3))
                else:
                    label_image = np.reshape(label_image, (heights[i], widths[i]))
                    pred_image = np.reshape(pred_image, (heights[i], widths[i]))
                    input_image = np.reshape(input_image, (heights[i], widths[i], 3))

                # Get bounding boxes of things
                bounding_boxes_pred = get_bounds(pred_image)
                bounding_boxes_label = get_bounds(label_image, pixel_dist=1)
                label_boxes = input_image.copy()
                predicted_boxes = input_image.copy()
                # TODO: Factor this out and make it general
                colors = [np.array([128, 0, 0]), np.array([170, 110, 40]), np.array([230, 190, 255]),
                          np.array([70, 240, 240]), np.array([200, 50, 200])]
                for cl in bounding_boxes_pred:
                    if cl == 0:
                        continue
                    bxs = bounding_boxes_pred[cl]
                    for coords in bxs:
                        tl_x, tl_y = coords[0]
                        br_x, br_y = coords[1]
                        coord_color = colors[cl]
                        predicted_boxes[tl_x:br_x + 1, tl_y-2:tl_y+2] = coord_color
                        predicted_boxes[tl_x:br_x + 1, br_y-2:br_y+2] = coord_color
                        predicted_boxes[tl_x-2:tl_x+2, tl_y:br_y+1] = coord_color
                        predicted_boxes[br_x-2:br_x+2, tl_y:br_y+1] = coord_color

                for cl in bounding_boxes_label:
                    if cl == 0:
                        continue
                    bxs = bounding_boxes_label[cl]
                    for coords in bxs:
                        tl_x, tl_y = coords[0]
                        br_x, br_y = coords[1]
                        coord_color = colors[cl]
                        label_boxes[tl_x:br_x + 1, tl_y-2:tl_y+2] = coord_color
                        label_boxes[tl_x:br_x + 1, br_y-2:br_y+2] = coord_color
                        label_boxes[tl_x-2:tl_x+2, tl_y:br_y+1] = coord_color  
                        label_boxes[br_x-2:br_x+2, tl_y:br_y+1] = coord_color  

                pix_acc = pixel_accuracy(pred_image, label_image)
                m_acc = mean_accuracy(pred_image, label_image)
                IoU = mean_IU(pred_image, label_image)
                freq_weighted_IU = frequency_weighted_IU(pred_image, label_image)
                class_IU = mean_IU_classes(pred_image, label_image)
                for cl in class_IU:
                    cl_IU = class_IU[cl]
                    if cl not in mean_class_IU:
                        mean_class_IU[cl] = [cl_IU]
                    else:
                        mean_class_IU[cl].append(cl_IU)

                mean_pixel_acc.append(pix_acc)
                mean_acc.append(m_acc)
                mean_IoU.append(IoU)
                mean_freq_weighted_IU.append(freq_weighted_IU)

                f, ax = plt.subplots(2, 3, figsize=(20, 20))

                ax[0, 0].imshow(input_image.astype(np.uint8), aspect='auto')
                ax[0, 1].imshow(label_image, aspect='auto')
                ax[0, 2].imshow(pred_image, aspect='auto')
                ax[1, 0].imshow(label_boxes.astype(np.uint8), aspect='auto')
                ax[1, 1].imshow(predicted_boxes.astype(np.uint8), aspect='auto')
                f.delaxes(ax[1, 2])
                plt.savefig('result_image/bX_rX_{}_{}.png'.format(j, i))
                plt.close()
            j += 1

        except tf.errors.OutOfRangeError:
            break

    print("Mean pixel accuracy:", np.mean(mean_pixel_acc))
    print("Mean accuraccy:", np.mean(mean_acc))
    print("Mean IoU:", np.mean(mean_IoU))
    print("Mean frequency weighted IU:", np.mean(mean_freq_weighted_IU))
    for cl in mean_class_IU:
        vals = mean_class_IU[cl]
        label = class_dict[cl]
        print("{} Mean IoU: {}".format(label, np.mean(vals)))
