"""
Created on May 28th, Zhe Sun

Use pretrain VGG model, only train fully connect layer

Usage:
python train_bottleneck_vgg.py
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
from keras import applications

from utils import MODEL_DICT

def get_nr_images_in_folder(folder):
    png_list = filter(lambda f: f.split('.')[-1] == 'png',
                      (f for _, _, file_list in os.walk(folder) for f in file_list))
    return len(png_list)

def get_nr_images_separately(folder):
    type_list = ['p', 'r', 's']
    folder_list = ['{}/{}'.format(folder, t) for t in type_list]

    get_png_nr = lambda folder: len(filter(lambda f: f.split('.')[-1] == 'png',
                                    (f for _, _, file_list in os.walk(folder) for f in file_list)))

    return [get_png_nr(folder) for folder in folder_list]


def train(image_dir, model_file, img_width, img_height):
    train_data_dir = '{}/train'.format(image_dir)
    validation_data_dir = '{}/validation'.format(image_dir)

    nb_train_samples = get_nr_images_in_folder(train_data_dir)
    nb_validation_samples = get_nr_images_in_folder(validation_data_dir)

    epochs = 100
    batch_size = 16

    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('model/bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('model/bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


    train_data = np.load(open('model/bottleneck_features_train.npy'))
    nr_train_png_list = get_nr_images_separately(train_data_dir)
    train_labels = np.array(
        [[1,0,0]]*nr_train_png_list[0] + [[0,1,0]] * nr_train_png_list[1] + [[0,0,1]]*nr_train_png_list[2])

    validation_data = np.load(open('model/bottleneck_features_validation.npy'))
    nr_validatioin_png_list = get_nr_images_separately(validation_data_dir)
    validation_labels = np.array(
        [[1,0,0]]*nr_validatioin_png_list[0] + [[0,1,0]]*nr_validatioin_png_list[1] + [[0,0,1]]*nr_validatioin_png_list[2])

    # train the fully connect model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save(model_file)


if __name__ == "__main__":
    # Use predefine parameters in utils.py
    train(MODEL_DICT['vgg']['data_file'],
          MODEL_DICT['vgg']['model_file'],
          MODEL_DICT['vgg']['width'],
          MODEL_DICT['vgg']['height']
        )
