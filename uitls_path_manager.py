import os
from os.path import dirname

PROJECT_FOLDER = dirname(__file__)

_directory = os.path.join(PROJECT_FOLDER, "data", "mit_bih")
_dataset_dir = os.path.join(PROJECT_FOLDER, "data", "beats_img")
_dataset_ann_dir = '../Data/dataset_ann/'
_preprocessed_record_ids = os.path.join(PROJECT_FOLDER, "data", "beats_img", "preprocessed_records_ids.txt")


def path_to_img(save_dir, label, record, start, end, **kwargs):
    img_path = '{}/{}/{}_{}{}{}0.png'.format(save_dir, label, label, record[-3:], start, end)
    return img_path