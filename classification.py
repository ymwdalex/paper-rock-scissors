"""
Created on May 28th, Zhe Sun

Classify the images

Usage:
python classification.py -type simple
"""

import sys
import os
import glob
import time
import cv2
import numpy as np
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras import applications

from utils import *

import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)


HAND_TYPE_STRING_LIST = ['paper', 'rock', 'scissors']


def classify(model_type):
    """
    Use opencv2 to classify PRS images
    """

    # load pretrained model
    model = load_model(MODEL_DICT[model_type]['model_file'])
    width = MODEL_DICT[model_type]['width']
    height = MODEL_DICT[model_type]['height']

    if model_type == 'vgg':
        model_vgg = applications.VGG16(include_top=False, weights='imagenet')

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

        # get hand images
        hand_img = frame[CAP_REGION_Y:CAP_REGION_BOTTOM, CAP_REGION_X:CAP_REGION_RIGHT, :]
        hand_img_resize = resize_from_array(hand_img, width, height)

        # Add an extra dimension, and scale the image pixel value
        data = np.expand_dims(hand_img_resize, axis=0)
        data = data / 255.0 # scale, very important!

        # make classification
        if model_type == 'vgg':
            input_data = model_vgg.predict(data, batch_size=1, verbose=0)
        else:
            input_data = data

        classification = model.predict(input_data, batch_size=1, verbose=0)
        result_string = HAND_TYPE_STRING_LIST[np.argmax(classification)]

        r.set('foo', result_string)

        cv2.putText(show_frame,
                    result_string,
                    (PREDICT_TEXT_REGION_X, PREDICT_TEXT_REGION_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, 255)

        cv2.imshow('Dilation',show_frame)


        # waiting for keyboard input
        k = cv2.waitKey(300) & 0xFF
        if k == 27: #close the output video by pressing 'ESC'
            break

    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture image parameters")
    parser.add_argument("-type",
                        dest="model_type",
                        action="store",
                        default="simple",
                        choices = MODEL_DICT.keys(),
                        help='model type'
                       )
    arguments = parser.parse_args()

    # classification
    classify(arguments.model_type)
