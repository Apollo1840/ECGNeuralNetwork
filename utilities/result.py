from sklearn.metrics import precision_recall_fscore_support
from graphics import confusion_matrix as cm
from utilities.labels import LABELS
import numpy as np


def calculate_support(y=None, test=None, print_confusion_matrix=False):
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
        max = np.argmax(y[i])
        y_pred.append(LABELS[str(max)])
        y_true.append(test[i][:3])
        labels.add(LABELS[str(max)])

    scores = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    if print_confusion_matrix:
        cm.ConfusionMatrix(y_true, y_pred, list(labels))

    return scores[0], scores[1], scores[2]

