import csv
import cv2
import numpy as np

from keras.model import Sequential, Model
from keras.layers import Flatten, Dense, Lambda        # Lambda for preprocessing
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D                     # Crop the images

import keras.models import Model
import matplotlib.pyplot as plt

def plot_training_validation_loss(history_object):
    """
    Plot the training and validation loss from model.fit or model.fit_generator
    using matplotlib

    :param history_object: object returned by model.fit
    :return: None
    """
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def augment_image(image):
    """
    Return an augmented copy of the original image. augmentation can be brightness,shift horizontal or vertical

    :param image: Image from the driving log
    :return: augmented_image: Transformed image
    Note: Keras has Imagedatagenerator which can augment images in batches, so this might be unnecessary.
    """

    pass

def augmented_dataset(images, measurements):
    """
    Take the original images and measurements and return an augmented dataset
    TODO: Make this a generator

    :param images: Original images from the driving log
    :param measurements: steering wheel measurement for that image
    :return: augmented_images, augmented_measurements
    """

def flipped_image(image):
    """
    Return flipped image

    :param image: Original image
    :return: flipped image
    """
    return np.fliplr(image)


def get_measurements_from_log(csv_file):
    """
    Get the input dataset and the measurements from the driving log
    and return the image and measurements
    When processing each line we randomly flip the measurement.
    Check if it is fine adding only the flipped images
    Also note that we add images from all three measurements
    A constant 0.2 is used to derive the steering wheel measurements for left and right
    This parameter needs to be tuned.

    :param csv_file:
    :return:
    """
    pass

def model(X_train, Y_train):
    """
    Steps to fit the model for augmented images and measurements, and save it at model.h5

    :param X_train: Images from the driving log
    :param Y_train: Steering wheel measurements
    :return: None

    """

    pass





