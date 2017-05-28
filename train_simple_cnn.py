"""
Created on May 28th, Zhe Sun

Train a simple CNN model

Usage:
python train_simple_cnn.py
"""

import numpy as np
from PIL import Image
import os
from glob import glob
from matplotlib import pyplot
from os import walk
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from utils import MODEL_DICT

def get_nr_images_in_folder(folder):
    png_list = filter(lambda f: f.split('.')[-1] == 'png',
                      (f for _, _, file_list in os.walk(folder) for f in file_list))
    return len(png_list)


def train(image_dir, model_file, img_width, img_height):
    train_data_dir = '{}/train'.format(image_dir)
    validation_data_dir = '{}/validation'.format(image_dir)

    nb_train_samples = get_nr_images_in_folder(train_data_dir)
    nb_validation_samples = get_nr_images_in_folder(validation_data_dir)

    epochs = 100
    batch_size = 16

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, # rescale is very important here!
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)


    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save(model_file)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train simple CNN to classify paper/rock/scissors")
    # parser.add_argument("-f",
    #                     dest="folder_name",
    #                     action="store",
    #                     default="data",
    #                     help='folder that save PRS images'
    #                    )
    # parser.add_argument("-o",
    #                     dest="model_name",
    #                     action="store",
    #                     default='simple_cnn.h5',
    #                     help='Model file'
    #                    )
    # parser.add_argument("-width",
    #                     dest="width",
    #                     action="store",
    #                     default=64,
    #                     type=int,
    #                     help='Width of saved images'
    #                    )
    # parser.add_argument("-height",
    #                     dest="height",
    #                     action="store",
    #                     default=64,
    #                     type=int,
    #                     help='Height of saved images'
    #                    )
    # arguments = parser.parse_args()


    # Use predefine parameters in utils.py
    train(MODEL_DICT['simple']['data_file'],
          MODEL_DICT['simple']['model_file'],
          MODEL_DICT['simple']['width'],
          MODEL_DICT['simple']['height']
        )
