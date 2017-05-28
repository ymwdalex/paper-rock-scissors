"""
Created on May 28th, Zhe Sun

Generate training and validation images
Image in blue capture box will be saved as training and validation images

Usage:
python generate_image.py -f data -t p
python generate_image.py -f data -t r
python generate_image.py -f data -t s
"""

import sys
import os
import glob
import time
import cv2
import numpy as np
import argparse

from utils import *


def prepare_folder(capture_folder, hand_type):
    """
    Create folders if they do not exist
    Return number of existing images in the folder
    """

    # create capture folders
    if not os.path.exists(capture_folder):
        os.makedirs(capture_folder)
        print "Create capture folder: {}".format(capture_folder)

    if not os.path.exists('{}/train'.format(capture_folder)):
        os.makedirs('{}/train'.format(capture_folder))
        print "Create capture folder: {}".format('{}/train'.format(capture_folder))

    if not os.path.exists('{}/validation'.format(capture_folder)):
        os.makedirs('{}/validation'.format(capture_folder))
        print "Create capture folder: {}".format('{}/validation'.format(capture_folder))

    # create folder for training images
    path_train_folder = '{}/train/{}'.format(capture_folder, hand_type)
    if not os.path.exists(path_train_folder):
        os.makedirs(path_train_folder)
        print "Create Train folder: {}".format(path_train_folder)

    img_train_idx = len(glob.glob('{}/*.png'.format(path_train_folder)))
    print "Train folder has {} images".format(img_train_idx)

    # create folder for validation images
    path_validation_folder = '{}/validation/{}'.format(capture_folder, hand_type)
    if not os.path.exists(path_validation_folder):
        os.makedirs(path_validation_folder)
        print "Create Validation folder: {}".format(path_validation_folder)

    img_validate_idx = len(glob.glob('{}/*.png'.format(path_validation_folder)))
    print "Validation folder has {} images".format(img_validate_idx)

    return path_train_folder, img_train_idx, path_validation_folder, img_validate_idx


def capture(path_train_folder, img_train_idx, path_validation_folder, img_validate_idx, width, height):
    """
    Use opencv2 to capture PRS images
    """

    # initial save_flag
    # When save_flag is True, start save images
    # Key 's' to switch save_flag
    save_flag = False

    # validation_flag will decide if a image is saved as train or validation
    validation_flag = 0

    # start camera
    cap = cv2.VideoCapture(0)

    while(1):

        # Capture frames from the camera
        _, frame = cap.read()

        # flip frame to align the movement
        frame=cv2.flip(frame,1)

        # backup the frame to draw capture area
        show_frame = frame

        # draw capture region
        cv2.rectangle(show_frame,CAP_REGION_LEFTTOP,CAP_REGION_RIGHTBOTTOM, CAP_COLOR, 2)
        cv2.imshow('Dilation',show_frame)

        # save images
        if save_flag:
            # every 3 images, save 2 in train and 1 in validation
            if validation_flag % 3 != 0:
                path = '{}/{}.png'.format(path_train_folder, img_train_idx)
                img_train_idx += 1
                print 'Save training images {}/{}.png'.format(path_train_folder, img_train_idx)
            else:
                path = '{}/{}.png'.format(path_validation_folder, img_validate_idx)
                img_validate_idx += 1
                print 'Save validation images {}/{}.png'.format(path_validation_folder, img_validate_idx)

            # get hand images
            hand_img = frame[CAP_REGION_Y:CAP_REGION_BOTTOM, CAP_REGION_X:CAP_REGION_RIGHT, :]
            hand_img_resize = resize_from_array(hand_img, width, height)

            cv2.imwrite(path, hand_img_resize)
            validation_flag += 1

        # waiting for keyboard input
        k = cv2.waitKey(500) & 0xFF
        if k == 27: #close the output video by pressing 'ESC'
            break
        elif k == ord('s'):
            save_flag = not save_flag
            print "Swith save flag"

    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture image parameters")
    parser.add_argument("-f",
                        dest="folder_name",
                        action="store",
                        default="data",
                        help='folder that save capture images'
                       )
    parser.add_argument("-t",
                        dest="hand_type",
                        action="store",
                        required=True,
                        choices=['p', 'r', 's'],
                        help='Hand type, required arguments, must be one of p/r/s'
                       )
    parser.add_argument("-width",
                        dest="width",
                        action="store",
                        default=64,
                        type=int,
                        help='Width of saved images'
                       )
    parser.add_argument("-height",
                        dest="height",
                        action="store",
                        default=64,
                        type=int,
                        help='Height of saved images'
                       )


    arguments = parser.parse_args()

    path_train_folder, img_train_idx, path_validation_folder, img_validate_idx = prepare_folder(arguments.folder_name, arguments.hand_type)
    capture(path_train_folder,
            img_train_idx,
            path_validation_folder,
            img_validate_idx,
            arguments.width,
            arguments.height)
