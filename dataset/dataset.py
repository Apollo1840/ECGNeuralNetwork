import tqdm
import random
random.seed(2020)

from os.path import isfile, join
from os import listdir
from utilities.labels import LABELS, LABELS_MAPPING
import os
import wfdb
import numpy as np
from .data_transform import signal_to_im

_range_to_ignore = 20
_directory = '../Data/mitbih/'
_dataset_dir = '../Data/dataset_filtered/'
_dataset_ann_dir = '../Data/dataset_ann/'
_split_percentage = .70
_split_validation_percentage = 0.70
_split_test_percentage = 0.50
_width = 2503
_height = 3361


def create_img_from_dir(
        data_dir=_directory,
        save_dir=_dataset_dir,
        size=(128, 128),
        augmentation=True,
        smoothing=True):
    """
    For each beat for each patient creates img apply some filters
    from _directory to _dataset_dir

    :param data_dir: str, ends with '/'
    :param save_dir: str, ends with '/'
    :param size: tuple of int, the img size
    :param augmentation: Bool, create for each image another nine for each side
    :param smoothing: Bool
    """

    # prepare data_pathes
    for label in LABELS:
        dir = '{}/{}'.format(save_dir, label)
        if not os.path.exists(dir):
            os.makedirs(dir)

    records = [f[:-4] for f in listdir(data_dir) if isfile(join(data_dir, f)) if (f.find('.dat') != -1)]
    random.shuffle(records)

    count = 0
    for record in records:

        sig, field = wfdb.rdsamp(os.path.join(data_dir, record))
        ann = wfdb.rdann(os.path.join(data_dir, record), extension='atr')

        # loop through all annotation sample to get wanted beats
        beats_indexes = []
        for i in range(1, len(ann.sample) - 1):

            if ann.symbol[i] not in LABELS_MAPPING:
                continue

            start = ann.sample[i - 1] + _range_to_ignore
            end = ann.sample[i + 1] - _range_to_ignore
            label = LABELS_MAPPING[ann.symbol[i]]

            beats_indexes.append((start, end, label))

        # loop through all beats indexes to get all beats and change it to images
        for start, end, label in tqdm.tqdm(beats_indexes):

            # get beat
            ''' Get the Q-peak intervall '''
            id_lead = 0
            beat = [sig[j][id_lead] for j in range(start, end)]

            if smoothing:
                beat = piecewise_aggregate_approximation(beat, paa_dim=100)

            ''' Convert in gray scale and resize img '''
            signal_to_im(beat,
                         img_path='{}/{}/{}_{}{}{}0.png'.format(save_dir, label, label, record[-3:], start, end),
                         resize=size, use_cropping=augmentation)

        count += 1
        print("{}/{}".format(count, len(records)))


def create_img_from_dir_split(
        data_dir=_directory,
        save_dir=_dataset_dir,
        size=(128, 128),
        augmentation=True,
        smoothing=True):
    """
    For each beat for each patient creates img apply some filters
    from _directory to _dataset_dir

    :param data_dir: str, ends with '/'
    :param save_dir: str, ends with '/'
    :param size: tuple of int, the img size
    :param augmentation: Bool, create for each image another nine for each side
    :param smoothing: Bool
    """

    prepare_data_pathes(save_dir)

    records = [f[:-4] for f in listdir(data_dir) if isfile(join(data_dir, f)) if (f.find('.dat') != -1)]
    random.shuffle(records)

    train = records[: int(len(records) * _split_percentage)]
    # test = records[int(len(records) * _split_percentage):]

    count = 0
    for record in records:
        if record in train:
            filename_convention = '{}train/{}/{}_{}{}{}0.png'
        else:
            filename_convention = '{}validation/{}/{}_{}{}{}0.png'

        sig, _ = wfdb.rdsamp(os.path.join(data_dir, record))
        ann = wfdb.rdann(os.path.join(data_dir, record), extension='atr')

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
            filename = filename_convention.format(save_dir, label, label, record[-3:], start, end)

            ''' Convert in gray scale and resize img '''
            signal_to_im(beat, filename, resize=size, use_cropping=augmentation)

        count += 1
        print("{}/{}".format(count, len(records)))


def create_txt_from_dir(
        data_dir=_directory,
        save_dir=_dataset_ann_dir,
        smoothing=True):

    prepare_data_pathes(save_dir)

    records = [f[:-4] for f in listdir(data_dir) if isfile(join(data_dir, f)) if (f.find('.dat') != -1)]
    random.shuffle(records)

    train = records[: int(len(records) * _split_percentage)]
    # test = records[int(len(records) * _split_percentage):]

    count = 0
    for record in records:
        if record in train:
            filename_convention = '{}train/{}/{}_{}{}{}.txt'
        else:
            filename_convention = '{}validation/{}/{}_{}{}{}.txt'

        sig, _ = wfdb.rdsamp(os.path.join(data_dir, record))
        ann = wfdb.rdann(os.path.join(data_dir, record), extension='atr')

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
            filename = filename_convention.format(save_dir, label, label, record[-3:], start, end)

            signal_to_txt(beat, filename)

        count += 1
        print("{}/{}".format(count, len(records)))


def prepare_data_pathes(save_dir):
    for label in LABELS:
        dir = '{}train/{}'.format(save_dir, label)
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir = '{}validation/{}'.format(save_dir, label)
        if not os.path.exists(dir):
            os.makedirs(dir)


def piecewise_aggregate_approximation(vector, paa_dim: int):
    '''
    Transform signal in a vector of size M

    :param vector: signal
    :param paa_dim: the size of average window
    :return: np.array: smoothed signal
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
    :param directory: str.
    :return: 3 List[str]: each of them is a list of filenames, eg. xxx.png or xxx.txt
    """

    train = []
    validation = []
    test = []

    classes = set(LABELS)
    classes_dict = dict()

    for cls in classes:
        classes_dict[cls] = [f for f in listdir(os.path.join(directory, cls)) if cls in f if f[-5] == '0']
        # it is very import to keep f[-5] to be zero,
        # so that data augmentation will not cause overlap between train and test
        # eg. classes_dict["NOR"] = ["NOR10.png", "NOR20.png", ...]

        random.shuffle(classes_dict[cls])

    for _, item in classes_dict.items():
        # item is list of dir which has cls.
        train += item[: int(len(item) * _split_validation_percentage)]

        # val contains validation and test
        val = item[int(len(item) * _split_validation_percentage):]

        validation += val[: int(len(val) * _split_test_percentage)]
        test += val[int(len(val) * _split_test_percentage):]

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)

    for data in [(train, "train.txt"), (validation, "validation.txt"), (test, "test.txt")]:
        with open(os.path.join(directory, data[1]), "w") as f:
            f.writelines(data[0])

    return train, validation, test
