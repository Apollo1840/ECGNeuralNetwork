from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Dropout, MaxPooling2D
from graphics.train_val_tensorboard import TrainValTensorBoard
from keras import Sequential
from dataset import dataset
from utilities.labels import LABELS
from random import randint
from keras import regularizers
from graphics import confusion_matrix as cm
import numpy as np
import random
import cv2
import os
import imutils
from dataset.data_augmentation import augmentated_filenames2

_dataset_dir = './data/beats_img'
_model = './trained_models/cnn_baseline.h5'

_train_files = 71207
_validation_files = 36413
_rotate_range = 180
_size = (64, 64)
_batch_size = 32
_filters = (4, 4)
_epochs = 3
_n_classes = 8
_regularizers = 0.0001

_probability_to_change = 0.30
_seed = 7


def create_model():
    """
        Create model
        :return:

    """

    model = Sequential()

    model.add(Conv2D(64, _filters, input_shape=(_size[0], _size[1], 1), padding='same',
                     kernel_regularizer=regularizers.l1_l2(_regularizers, _regularizers), activation='relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(64, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(
        Conv2D(128, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(
        Conv2D(128, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(
        Conv2D(256, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(
        Conv2D(256, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(_regularizers), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def load_cnn_model():
    """
        Load and return model
        :return:
    """
    model = load_model(_model)
    return model


def encode_label(file):
    """
        Encode the class label
        :param file:
        :return:
    """
    label = [0 for _ in range(_n_classes)]
    label[int(LABELS.index(file[:3]))] = 1
    return label


def steps(files, batch_size):
    """
        Calculate the number steps necessary to process each files
        :param files:
        :param batch_size:
        :return: the numbers of files divided to batch
    """
    return len(files) / batch_size


def load_dataset(files, directory,
                 batch_size, size,
                 random_crop,
                 random_rotate,
                 flip):
    """
        Load dataset in minibatch
        :param files: List[str], each file is xxx.png
        :param directory: str
        :param batch_size: int
        :param size: int
        :param random_crop: Bool, no effect in this case
        :return:
    """

    L = len(files)

    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X, Y = images2array(files[batch_start:limit], size, directory, random_rotate, flip)
            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size


def images2array(files,
                 size,
                 directory,
                 random_rotate,
                 flip,
                 encode_labels=True):
    """
        Convert an image to array and encode its label

        :param files: str. eg. xxx.png
        :param size:
        :param directory:
        :return: image converted and its label
    """

    images = []
    labels = []
    for file in files:
        file_name = os.path.join(directory, file[:3], file)
        img = image_to_array(file_name, size, random_rotate, flip)

        if encode_labels:
            label = encode_label(file)
        else:
            label = LABELS.index([file[:3]])

        images.append(img)
        labels.append(label)

    X = np.array(images)
    Y = np.array(labels)

    return X, Y


def image_to_array(filename, size, random_rotate=True, flip=True):
    """

    :param filename:
    :param size:
    :param random_rotate:
    :param flip:
    :return: np.array
    """

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if random_rotate:
        if random.uniform(0, 1) < _probability_to_change:
            img = imutils.rotate(img, randint(-_rotate_range, _rotate_range))
    if flip:
        if random.uniform(0, 1) < _probability_to_change:
            img = cv2.flip(img, randint(-1, 1))
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)

    img = img.astype('float64')
    img /= 255
    img = np.reshape(img, [size[0], size[1], 1])
    return img


def training(train=None, validation=None, augmentation=True):
    """
        Training the model
        :return:
    """
    model = create_model()
    if train is None and validation is None:
        train, validation, test = dataset.load_files(_dataset_dir)

    if augmentation:
        train = augmentated_filenames2(train)

    callbacks_list = [ModelCheckpoint(_model, monitor='val_loss', save_best_only=True),
                      TrainValTensorBoard(write_graph=False)]

    data_gen_train = load_dataset(train, _dataset_dir,
                                  _batch_size, _size,
                                  random_crop=augmentation,
                                  random_rotate=augmentation,
                                  flip=augmentation)

    data_gen_valid = load_dataset(validation, _dataset_dir,
                                  _batch_size, _size,
                                  random_crop=augmentation,
                                  random_rotate=augmentation,
                                  flip=augmentation)

    model.fit_generator(data_gen_train,
                        steps_per_epoch=steps(train, _batch_size),
                        epochs=_epochs,
                        validation_data=data_gen_valid,
                        validation_steps=steps(validation, _batch_size),
                        callbacks=callbacks_list)

    return model


def predict_model(model=None, filenames=None):
    """
        Predict model
        :return:
    """
    x, _ = images2array(filenames, _size, _dataset_dir, True, True, True)
    y = model.predict(np.reshape(x, (len(filenames), _size[0], _size[1], 1)))
    return y
