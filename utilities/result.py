from sklearn.metrics import precision_recall_fscore_support
from graphics import confusion_matrix as cm
from utilities.labels import LABELS
import numpy as np

AAMI_LABEL_MAPPING = {
    "N": ["NOR", "LBB", "RBB"],
    "SVEB": ["APC"],
    "VEB": ["VEB", "PVC"],
    "F": []
}


AAMI_LABELS = sorted(AAMI_LABEL_MAPPING.keys())


def thislabel2aami(thislabel):
    """

    :param thislabel: str
    :return: str
    """
    for aami_label in AAMI_LABELS:
        if thislabel in AAMI_LABEL_MAPPING[aami_label]:
            return aami_label
    return None


def calculate_support(y=None, test=None, label_type="", print_confusion_matrix=True):
    """
        Calculate Precision, Recall, F-Measure
        :param y: Predicted classes
        :param test: List[str], list of txt filename, filename[:3] is the true label
        :param print_confusion_matrix:
        :return: Precision, Recall, F-Measure
    """

    y_true = []
    y_pred = []
    labels = set()
    for i in range(len(test)):
        pred_i = np.argmax(y[i])
        y_pred.append(pred_i)

        if label_type == "AAMI":
            aami_label = thislabel2aami(test[i][:3])
            if aami_label:
                y_true.append(AAMI_LABELS.index(aami_label))
            labels.add(AAMI_LABELS[pred_i])

        else:
            y_true.append(LABELS.index(test[i][:3]))
            labels.add(LABELS[pred_i])

    assert len(y_true) == len(y_pred)
    assert max(y_true) == max(y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    if print_confusion_matrix:
        cm.ConfusionMatrix(y_true, y_pred, list(labels))

    return precision, recall, fscore

