import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score

from utilities import result
from cnn.cnn import predict_model, valid_filenames_for_aami
from dataset.dataset import readlist

from configs import TEST_DATA_PATH
from configs import AAMI_CLASSES_REDUCED as AAMI_LABELS
from configs import AAMI2TYPE2_MAPPING_REDUCED as AAMI_LABEL_MAPPING
from configs import TYPE2_CLASSES as LABELS


def thislabel2aami(thislabel):
    """

    :param thislabel: str
    :return: str
    """
    for aami_label in AAMI_LABELS:
        if thislabel in AAMI_LABEL_MAPPING[aami_label]:
            return aami_label
    return None


def index2onehot(indx, len_onehot):
    onehot = np.zeros(len_onehot)
    onehot[indx] = 1
    return onehot


def aamilabel2onehot(ammi_label):
    return index2onehot(AAMI_LABELS.index(ammi_label), len(AAMI_LABELS))


def thislabel2onehot(this_label):
    return index2onehot(LABELS.index(this_label), len(LABELS))


def evaluate_cnn(model, label_type, keep_ratio=1, verbose=True):
    """

    :param model:
    :param keep_ratio:
    :param verbose:
    :return:
    """

    test_cnn_all = readlist(TEST_DATA_PATH)
    test_cnn_clip = test_cnn_all[:int(len(test_cnn_all)*keep_ratio)]

    if label_type == "AAMI":
        test_cnn = valid_filenames_for_aami(test_cnn_clip)
        true = [aamilabel2onehot(thislabel2aami(filename[:3])) for filename in test_cnn]
    else:
        test_cnn = test_cnn_clip
        true = [thislabel2onehot(filename[:3]) for filename in test_cnn]

    pred = predict_model(model, test_cnn, label_type, verbose)

    y_true = [np.argmax(y) for y in true]
    y_pred = [np.argmax(y) for y in pred]

    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print("precision: ", precision)
    print("recall: ", recall)
    print("f1score: ", fscore)

    print("marco f1 score", f1_score(y_true, y_pred, average='macro'))
    print(classification_report(y_true, y_pred, target_names=AAMI_LABELS[:3]))

    return y_true, y_pred
