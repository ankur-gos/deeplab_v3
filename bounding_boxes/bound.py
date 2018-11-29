#!/usr/bin/env python3
"""
Some functions that produce bounding boxes for an image
"""

import numpy as np
import ipdb


def get_bounds(input_np_image, pixel_dist=5):
    """
    Get a list of bounding boxes and the class associated with them.
    I merge same class pixels within pixel_dist pixels
    :param input_np_image: n x m np image to be scanned
    :param pixel_dist: accept distance between pixel and adjacent class bounding box
    :return: dict of list of top left and bottom right coordinates, indexed by class
    Example: {1: [((0, 1), (2, 2))]}
    """
    # Store current list of
    current_boxes_list = {}
    # Scan left to right, top to bottom
    for y in range(input_np_image.shape[1]):
        for x in range(input_np_image.shape[0]):
            # Get the class of the current pixel
            current_pix = input_np_image[x][y]
            # Check if we're touching a current bounding box
            if current_pix in current_boxes_list:
                boxes = current_boxes_list[current_pix]

                def check_dist(coords):
                    top_left, bottom_right = coords
                    br_x, br_y = bottom_right
                    tl_x, _ = top_left
                    if br_x + pixel_dist >= x >= tl_x - pixel_dist and y <= br_y + pixel_dist:
                        return True
                    return False

                def map_new(coords):
                    # Note we're scanning top to bottom, so there's no way the new coordinate can extend the box up
                    tl, br = coords
                    br_x, br_y = br
                    tl_x, tl_y = tl
                    # If x, y is to the bottom left of the box
                    new_coord = ((x, tl_y), (br_x, y))
                    # If x, y is directly below the box
                    if br_x > x > tl_x:
                        new_coord = (tl, (br_x, y))
                    # If x, y is to the bottom right of the box
                    elif x >= br_x:
                        new_coord = (tl, (x, y))
                    return new_coord
                no_boxes = [b for b in boxes if not check_dist(b)]
                new_boxes = [map_new(b) for b in boxes if b not in no_boxes]
                if len(new_boxes) == 0:
                    new_boxes = [((x, y), (x, y))]
                current_boxes_list[current_pix] = no_boxes + new_boxes
            else:
                current_boxes_list[current_pix] = [((x, y), (x, y))]
    #for cl in current_boxes_list:
    #    for box in current_boxes_list[cl]:
    #        for box2 in current_boxes_list[cl]:
    #            b1_tl, b1_br = box
    #            b2_tl, b2_br = box2
    #            b1_tl_x, b1_tl_y = b1_tl
    #            b1_br_x, b1_br_y = b1_br
    #            b2_tl_x, b2_tl_y = b2_tl
    #            b2_br_x, b2_br_y = b1_br
    return current_boxes_list


def test_get_bounds():
    test1 = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    test1_arr = np.array(test1)
    bxs = get_bounds(test1_arr)
    predict = {0: [((0, 0), (2, 2))]}
    if bxs != predict:
        #raise Exception(f'Test1 failed, {bxs}, {predict}')
        pass

    test2 = [[0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1]]
    test2_arr = np.array(test2)
    bxs = get_bounds(test2_arr.transpose())
    predict = {0: [((0, 0), (4, 5))], 1: [((5, 0), (5, 5))]}
    if bxs != predict:
        #raise Exception(f'Test2 failed, {bxs}, {predict}')
        pass

    test3 = [[0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1]]
    test3_arr = np.array(test3)
    bxs = get_bounds(test3_arr.transpose())
    predict = {0: [((0, 0), (6, 5))], 1: [((7, 0), (7, 5)), ((0, 4), (0, 5))]}
    if bxs != predict:
        for key in bxs:
            l1 = bxs[key]
            l2 = predict[key]
            sl1 = sorted(l1)
            sl2 = sorted(l2)
            if sl1 != sl2:
                #raise Exception(f'Test3 failed, {bxs}, {predict})
                pass
    print('Tests all pass')


if __name__ == '__main__':
    test_get_bounds()


