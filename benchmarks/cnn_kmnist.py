#!/usr/bin/env python
# -*- coding: utf-8 -*-

# cnn_kmnist.py
#----------------
# Train a small CNN to identify 10 Japanese characters in classical script
# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from pprint import pprint

# For compatibility with RTX 2070
from tensorflow import keras
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

import argparse
import numpy as np
import os
from utils import load_train_data, load_test_data, load, KmnistCallback
import wandb
from wandb.keras import WandbCallback
import random


# default configuration / hyperparameter values
# you can modify these below or via command line
DATA_HOME = "./dataset" 
BATCH_SIZE = 2**random.randint(5, 10)  # 32, 64, 128, 256, 512, 1024
EPOCHS = 100
FILTERS_INIT = 2**random.randint(4, 6)  # 16, 32, 64
DROPOUT = random.uniform(0, 1)
FC_SIZE = 2 ** random.randint(7, 11)  # 128, 256, 512, 1024, 2056
BLOCKS = random.randint(3, 5)         # resolution too low to have many max pools
CONV_PER_BLOCK = random.randint(1, 4)
NUM_CLASSES = 10
LEARNING_RATE = 10 ** random.uniform(-4, -2)
#NUM_CLASSES_K49 = 49
SUBTRACT_PIXEL_MEAN = random.choice([True, False])

# input image dimensions
img_rows, img_cols = 28, 28
# ground truth labels for the 10 classes of Kuzushiji-MNIST Japanese characters 
LABELS_10 =["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"] 
LABELS_49 = ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち",
"つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め"
"も","や","ゆ","よ","ら","り","る","れ","ろ","わ","ゐ","ゑ","を","ん","ゝ"]

def train_cnn(args):
  # initialize wandb logging to your project
  wandb.init()
  config = {
    "model_type" : "cnn",
    "batch_size" : BATCH_SIZE,
    "num_classes" : NUM_CLASSES,
    "epochs" : EPOCHS,
    "filters_init": FILTERS_INIT,
    "dropout": DROPOUT,
    "blocks": BLOCKS,
    "conv_per_block": CONV_PER_BLOCK,
    "fc_size": FC_SIZE,
    "learning_rate": LEARNING_RATE,
    "subtract_pixel_mean": SUBTRACT_PIXEL_MEAN
  }
  wandb.config.update(config)

  pprint(config)

  # Load the data form the relative path provided
  x_train, y_train = load_train_data(args.data_home)
  x_test, y_test = load_test_data(args.data_home)

  # reshape to channels last
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  # If subtract pixel mean is enabled
  if SUBTRACT_PIXEL_MEAN:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
 
  # Data augmentation
  datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1)
  datagen.fit(x_train)
 
  N_TRAIN = len(x_train)
  N_TEST = len(x_test)
  wandb.config.update({"n_train" : N_TRAIN, "n_test" : N_TEST})
  print('{} train samples, {} test samples'.format(N_TRAIN, N_TEST))

  # Convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

  # Build model (inspired from https://keras.io/examples/cifar10_resnet/)
  model = tf.keras.Sequential()
  filters = FILTERS_INIT
  model.add(layers.Conv2D(filters, kernel_size=(3, 3),
            input_shape=input_shape, padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('relu'))
  model.add(layers.Dropout(DROPOUT / 2))
  for _ in range(CONV_PER_BLOCK - 1):
    model.add(layers.Conv2D(filters, kernel_size=(3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(DROPOUT / 2))
  for _ in range(BLOCKS - 1):
    filters *= 2
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    for _ in range(CONV_PER_BLOCK):
      model.add(layers.Conv2D(filters, kernel_size=(3, 3), padding="same"))
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))
      model.add(layers.Dropout(DROPOUT / 2))
  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dropout(DROPOUT))
  model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

  model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])

  model.summary()

  model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                      epochs=EPOCHS,
                      verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=[KmnistCallback(), EarlyStopping(patience=5), ReduceLROnPlateau(),
                                 WandbCallback(data_type="image", labels=LABELS_10)])

  train_score = model.evaluate(x_train, y_train, verbose=0)
  test_score = model.evaluate(x_test, y_test, verbose=0)
  print('Train loss:', train_score[0])
  print('Train accuracy:', train_score[1])
  print('Test loss:', test_score[0])
  print('Test accuracy:', test_score[1])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "--data_home",
    type=str,
    default=DATA_HOME,
    help="Relative path to training/test data")
  args = parser.parse_args()

  train_cnn(args)

