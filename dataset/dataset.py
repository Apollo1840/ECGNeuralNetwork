import tqdm
import random
random.seed(2020)

from os.path import isfile, join
from os import listdir
from utilities.labels import LABELS, LABELS_MAPPING
import os
import wfdb
import numpy as np
from .data_transform import signal_to_im2

_range_to_ignore = 20
_directory = '../Data/mitbih/'
_dataset_dir = '../Data/dataset_filtered/'
_dataset_ann_dir = '../Data/dataset_ann/'
_split_percentage = .70
_split_validation_percentage = 0.70
_split_test_percentage = 0.50
_width = 2503
_height = 3361


def create_img_from_dir(size=(128, 128), augmentation=True, smoothing=True):
    """
       For each beat for each patient creates img apply some filters
       :param size: tuple of int, the img size
       :param augmentation: Bool, create for each image another nine for each side
       :param smoothing: Bool
    """

    prepare_data_pathes()

    records = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]
    random.shuffle(records)

    train = records[: int(len(records) * _split_percentage)]
    # test = records[int(len(records) * _split_percentage):]

    for record in records:
        if record in train:
            filename_convention = '{}train/{}/{}_{}{}{}0.png'
        else:
            filename_convention = '{}validation/{}/{}_{}{}{}0.png'

        sig, _ = wfdb.rdsamp(_directory + record)
        ann = wfdb.rdann(_directory + record, extension='atr')

        # loop through all beats
        for i in tqdm.tqdm(range(1, len(ann.sample) - 1)):

            if ann.symbol[i] not in LABELS_MAPPING:
                continue

            # get beat
            ''' Get the Q-peak intervall '''
            id_lead = 0
            start = ann.sample[i - 1] + _range_to_ignore
            end = ann.sample[i + 1] - _range_to_ignore
            beat = [sig[j][id_lead] for j in range(start, end)]

            if smoothing:
                beat = piecewise_aggregate_approximation(beat, paa_dim=100)

            # get filename by label
            label = LABELS_MAPPING[ann.symbol[i]]
            filename = filename_convention.format(_dataset_dir, label, label, record[-3:], start, end)

            ''' Convert in gray scale and resize img '''
            signal_to_im2(beat, filename, resize=size, use_cropping=augmentation)


def create_txt_from_dir(smoothing=True):

    prepare_data_pathes()

    records = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]
    random.shuffle(records)

    train = records[: int(len(records) * _split_percentage)]
    # test = records[int(len(records) * _split_percentage):]

    for record in records:
        if record in train:
            filename_convention = '{}train/{}/{}_{}{}{}.txt'
        else:
            filename_convention = '{}validation/{}/{}_{}{}{}.txt'

        sig, _ = wfdb.rdsamp(_directory + record)
        ann = wfdb.rdann(_directory + record, extension='atr')

        # loop through all beats
        for i in tqdm.tqdm(range(1, len(ann.sample) - 1)):

            if ann.symbol[i] not in LABELS_MAPPING:
                continue

            # get beat
            ''' Get the Q-peak intervall '''
            id_lead = 0
            start = ann.sample[i - 1] + _range_to_ignore
            end = ann.sample[i + 1] - _range_to_ignore
            beat = [sig[j][id_lead] for j in range(start, end)]

            if smoothing:
                beat = piecewise_aggregate_approximation(beat, paa_dim=100)

            # get filename by label
            label = LABELS_MAPPING[ann.symbol[i]]
            filename = filename_convention.format(_dataset_ann_dir, label, label, record[-3:], start, end)

            signal_to_txt(beat, filename)


def prepare_data_pathes():
    for label in LABELS:
        dir = '{}train/{}'.format(_dataset_dir, label)
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir = '{}validation/{}'.format(_dataset_dir, label)
        if not os.path.exists(dir):
            os.makedirs(dir)


def piecewise_aggregate_approximation(vector, paa_dim: int):
    '''
        Transform signals in a vector of size M

        :param vector: signals
        :param paa_dim: the new size
        :return:
    '''

    Y = np.array(vector)
    if Y.shape[0] % paa_dim == 0:
        sectionedArr = np.array_split(Y, paa_dim)
        res = np.array([item.mean() for item in sectionedArr])
    else:
        value_space = np.arange(0, Y.shape[0] * paa_dim)
        output_index = value_space // Y.shape[0]
        input_index = value_space // paa_dim
        uniques, nUniques = np.unique(output_index, return_counts=True)
        res = [Y[indices].sum() / Y.shape[0] for indices in
               np.split(input_index, nUniques.cumsum())[:-1]]
    return res


def signal_to_txt(sig, txt_path):
    with open(txt_path, 'w+') as f:
        for i, item in enumerate(sig):
            f.write("%s" % item)
            if i < len(sig) - 1:
                f.write(", ")


def load_files(directory):
    """
        Load each name file in the directory
        :param directory:
        :return:
    """
    train = []
    validation = []
    test = []

    classes = set(LABELS)
    classes_dict = dict()

    for key in classes:
        classes_dict[key] = [f for f in listdir(directory) if key in f if f[-5] == '0']
        random.shuffle(classes_dict[key])

    for _, item in classes_dict.items():
        train += item[: int(len(item) * _split_validation_percentage)]
        val = item[int(len(item) * _split_validation_percentage):]
        validation += val[: int(len(val) * _split_test_percentage)]
        test += val[int(len(val) * _split_test_percentage):]

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)
    return train, validation, test
