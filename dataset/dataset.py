import tqdm
import random

random.seed(2020)

import os
from os import listdir
from os.path import isfile, join, dirname


import wfdb
import numpy as np
import gc
from memory_profiler import profile

import sys

# add root of this project to sys.path

PROJECT_FOLDER = dirname(dirname(__file__))
print(PROJECT_FOLDER)
sys.path.append(PROJECT_FOLDER)

from dataset.data_transform import signal_to_im
from uitls_path_manager import path_to_img, _directory, _dataset_dir, _preprocessed_record_ids, _dataset_ann_dir
from configs import TYPE2_CLASSES as LABELS
from configs import MITBIH2TYPE2_MAPPING as LABELS_MAPPING
from configs import DS_bank


_range_to_ignore = 20
_split_percentage = .70
_split_validation_percentage = 0.80
_split_test_percentage = 0.50
_width = 2503
_height = 3361


def create_img_from_dir(
        data_dir=_directory,
        save_dir=_dataset_dir,
        labels_mapping=LABELS_MAPPING,
        records_wanted=None,
        size=(128, 128),
        id_lead=0,
        augmentation=True,
        smoothing=True,
        verbose=False,
):
    """
    For each beat for each patient creates img apply some filters
    from _directory to _dataset_dir

    :param data_dir: str, ends with '/'
    :param save_dir: str, ends with '/'
    :param records_wanted: List[int] or List[str]
    :param size: tuple of int, the img size
    :param id_lead: get which lead
    :param augmentation: Bool, create for each image another nine for each side
    :param smoothing: Bool
    :param verbose: Bool
    """

    # prepare data_pathes
    prepare_data_pathes(save_dir, labels=labels_mapping.values(), split_tvt=False)

    records = records_to_process(data_dir, records_wanted)

    if verbose:
        print("transforming beats to image for records:")
        print(records)

    count = 0
    for record in records:

        sig, field = wfdb.rdsamp(os.path.join(data_dir, record))
        ann = wfdb.rdann(os.path.join(data_dir, record), extension='atr')

        beats_indexes = beat_indices_from_annotaions(ann, labels_mapping)

        # loop through all beats indexes to get all beats and change it to images
        for i in tqdm.tqdm(range(len(beats_indexes))):

            start, end, label = beats_indexes[i]

            img_path = path_to_img(save_dir, label, record[-3:], start, end)
            if not os.path.isfile(img_path):
                # get beat
                ''' Get the Q-peak intervall '''

                beat = [sig[j][id_lead] for j in range(start, end)]
                if smoothing:
                    beat = piecewise_aggregate_approximation(beat, paa_dim=100)

                ''' Convert in gray scale and resize img '''
                # this function will store the image
                signal_to_im(beat,
                             img_path=img_path,
                             resize=size,
                             use_cropping=augmentation)

            gc.collect()

        count += 1
        print("{}/{}".format(count, len(records)))
        addlist(record, _preprocessed_record_ids)


def create_txt_from_dir(
        data_dir=_directory,
        save_dir=_dataset_ann_dir,
        smoothing=True):

    prepare_data_pathes(save_dir, LABELS)

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


def piecewise_aggregate_approximation(vector, paa_dim: int):
    '''
    Transform signal in a vector of size M

    :param vector: signal
    :param paa_dim: wanted dimension
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


def images_by_classes(directory, shuffle=True, verbose=False):
    classes_dict = {}
    for cls in LABELS:
        classes_dict[cls] = [f for f in listdir(os.path.join(directory, cls)) if cls in f if f[-5] == '0']
        # it is very import to keep f[-5] to be zero,
        # so that data augmentation will not cause overlap between train and test
        # eg. classes_dict["NOR"] = ["NOR10.png", "NOR20.png", ...]

        if shuffle:
            random.shuffle(classes_dict[cls])

    if verbose:
        for key, value in classes_dict.items():
            print("{}: {}\t".format(key, len(value)))

    return classes_dict


def images_by_record(directory, shuffle=True, verbose=False):

    record_dict = {}

    for cls in LABELS:
        imgs = [f for f in listdir(os.path.join(directory, cls)) if cls in f if f[-5] == '0']
        # it is very import to keep f[-5] to be zero,
        # so that data augmentation will not cause overlap between train and test
        # eg. classes_dict["NOR"] = ["NOR10.png", "NOR20.png", ...]

        record_ids = [record_id_from_filename(img) for img in imgs]

        for record_id, img in zip(record_ids, imgs):
            if record_id in record_dict:
                record_dict[record_id].append(img)
            else:
                record_dict[record_id] = [img]

    if shuffle:
        for record_id in record_dict:
            random.shuffle(record_dict[record_id])

    if verbose:
        for key, value in record_dict.items():
            print("{}: {}\t".format(key, len(value)))

    return record_dict


def load_files(directory, keep_ratio=1, verbose=False):
    """
    Load each name file in the directory

    :param directory: str.
    :param keep_ratio: float, 0 to 1
    :param verbose: Bool
    :return: 3 List[str]: each of them is a list of filenames, eg. xxx.png or xxx.txt
    """

    def make_tvt(directory, verbose=False):
        train = []
        validation = []
        test = []

        # use classes_dict to make sure class distribution are equal.
        classes_dict = images_by_classes(directory, verbose=verbose)

        for _, images in classes_dict.items():
            # item is list of dir which has cls.
            train += images[: int(len(images) * _split_validation_percentage)]

            # val contains validation and test
            val = images[int(len(images) * _split_validation_percentage):]

            validation += val[: int(len(val) * _split_test_percentage)]
            test += val[int(len(val) * _split_test_percentage):]

        random.shuffle(train)
        random.shuffle(validation)
        random.shuffle(test)

        variables = [train, validation, test]
        return variables

    def load_tvt(directory, verbose=False):
        if verbose:
            print("load tvt from {}".format(directory))

        variables = [[], [], []]
        files = ["train.txt", "validation.txt", "test.txt"]
        for i in range(len(files)):
            variables[i] = readlist(os.path.join(directory, files[i]))
        return variables

    def save_tvt(directory, variables):
        files = ["train.txt", "validation.txt", "test.txt"]
        for i in range(len(files)):
            writelist(variables[i], os.path.join(directory, files[i]))

    def exist_tvt(directory):
        files = ["train.txt", "validation.txt", "test.txt"]
        for i in range(len(files)):
            if not os.path.isfile(os.path.join(directory, files[i])):
                return False
        return True

    train, valid, test = load_data(directory,
                                   make_data_func=make_tvt,
                                   save_data_func=save_tvt,
                                   load_data_func=load_tvt,
                                   data_exist_func=exist_tvt,
                                   verbose=verbose)

    train = train[:int(len(train) * keep_ratio)]
    valid = valid[:int(len(valid) * keep_ratio)]
    test = test[:int(len(test) * keep_ratio)]
    return train, valid, test


def load_files_cross_patient(directory, keep_ratio=1, verbose=False):
    """
    Load each name file in the directory

    :param directory: str.
    :param keep_ratio: float, 0 to 1
    :param verbose: Bool
    :return: 3 List[str]: each of them is a list of filenames, eg. xxx.png or xxx.txt
    """

    def make_ds_tvt(directory, verbose=False):
        t_and_v = []
        test = []

        # use classes_dict to make sure class distribution are equal.
        record_dict = images_by_record(directory, verbose=verbose)

        for record in record_dict:
            if record in DS_bank["normal"]["DS1"]:
                t_and_v.extend(record_dict[record])
            elif record in DS_bank["normal"]["DS2"]:
                test.extend(record_dict[record])

        train = t_and_v[:int(len(t_and_v)*_split_validation_percentage)]
        validation = t_and_v[int(len(t_and_v)*_split_validation_percentage):]

        random.shuffle(train)
        random.shuffle(validation)
        random.shuffle(test)

        variables = [train, validation, test]
        return variables

    def load_ds_tvt(directory, verbose=False):
        if verbose:
            print("load DS tvt from {}".format(directory))

        variables = [[], [], []]
        files = ["train_DS1.txt", "validation_DS1.txt", "test_DS2.txt"]
        for i in range(len(files)):
            variables[i] = readlist(os.path.join(directory, files[i]))
        return variables

    def save_ds_tvt(directory, variables):
        files = ["train_DS1.txt", "validation_DS1.txt", "test_DS2.txt"]
        for i in range(len(files)):
            writelist(variables[i], os.path.join(directory, files[i]))

    def exist_ds_tvt(directory):
        files = ["train_DS1.txt", "validation_DS1.txt", "test_DS2.txt"]
        for i in range(len(files)):
            if not os.path.isfile(os.path.join(directory, files[i])):
                return False
        return True

    train, valid, test = load_data(directory,
                                   make_data_func=make_ds_tvt,
                                   save_data_func=save_ds_tvt,
                                   load_data_func=load_ds_tvt,
                                   data_exist_func=exist_ds_tvt,
                                   verbose=verbose)

    train = train[:int(len(train) * keep_ratio)]
    valid = valid[:int(len(valid) * keep_ratio)]
    test = test[:int(len(test) * keep_ratio)]
    return train, valid, test


def writelist(list_a, filename):
    """
    do not use writelines(), it is not good.

    :param list_a:
    :param filename:
    :return:
    """
    with open(filename, "w") as f:
        for line in list_a:
            f.write(line + "\n")


def addlist(item, filename):
    """
    do not use writelines(), it is not good.

    :param item: str
    :param filename:
    :return:
    """
    with open(filename, "a") as f:
        f.write(item + "\n")


def readlist(filename):
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines]
    else:
        return []


def load_data(data_filename,
              make_data_func,
              save_data_func,
              load_data_func,
              data_exist_func=None,
              verbose=False):
    """
    or do it in decorators way. make save to be bool, make exist and load into decorator

    :param data_filename:
    :param make_data_func:
    :param save_data_func:
    :param load_data_func:
    :param data_exist_func:
    :param verbose: Bool
    :return:
    """
    is_data_exist = data_exist_func(data_filename) if data_exist_func else os.path.isfile(data_filename)
    if is_data_exist:
        data = load_data_func(data_filename, verbose=verbose)
    else:
        data = make_data_func(data_filename, verbose=verbose)
        save_data_func(data_filename, data)
    return data


def prepare_data_pathes(save_dir, labels, split_tvt=True):

    if split_tvt:
        for label in labels:
            dir = '{}train/{}'.format(save_dir, label)
            if not os.path.exists(dir):
                os.makedirs(dir)

            dir = '{}valid/{}'.format(save_dir, label)
            if not os.path.exists(dir):
                os.makedirs(dir)

            dir = '{}test/{}'.format(save_dir, label)
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        for label in labels:
            dir = '{}/{}'.format(save_dir, label)
            if not os.path.exists(dir):
                os.makedirs(dir)


def records_to_process(data_dir, records_wanted=None):
    records = [f[:-4] for f in listdir(data_dir) if isfile(join(data_dir, f)) if (f.find('.dat') != -1)]
    # records = ["101", "102", ...]

    records_preprocesed = readlist(_preprocessed_record_ids)
    records = [r for r in records if r not in records_preprocesed]

    if records_wanted:
        records_wanted = [str(r) for r in records_wanted]
        records = [r for r in records if r in records_wanted]

    return records


def beat_indices_from_annotaions(anntations, labels_mapping):
    # loop through all annotation sample to get wanted beats

    beats_indexes = []
    for i in range(1, len(anntations.sample) - 1):

        if anntations.symbol[i] not in labels_mapping:
            continue

        start = anntations.sample[i - 1] + _range_to_ignore
        end = anntations.sample[i + 1] - _range_to_ignore
        label = labels_mapping[anntations.symbol[i]]

        beats_indexes.append((start, end, label))
    return beats_indexes


def record_id_from_filename(imgname):
    return int(imgname[4:7])


if __name__ == "__main__":
    data_dir = "../data/mit_bih/"
    save_dir = "../data/beats_img/"
    create_img_from_dir(data_dir, save_dir)
