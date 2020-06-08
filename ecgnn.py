from utilities import result
from ann import ann
from cnn import cnn
from dataset.dataset import readlist

'''
    Use this script such as main

    In order to see the the information about 
    the neural network performe this command:
    tensorboard --logdir=logs
'''

TEST_DATA_PATH = 'dataset/test.txt'


def combine_result(y1, y2, filenames, weighted=True, print_precision=True, print_confusion_matrix=True):
    """
        Combine the result of two prediction models
        :param y1: y_pred
        :param y2: y_pred
        :param filenames: List[str], list of .txt files. Used to calculate the true value of test,
            since the name of the file is the true label
        :return:
    """

    precision_y1, _, f_measure_y1 = result.calculate_support(y1, filenames)
    precision_y2, _, f_measure_y2 = result.calculate_support(y2, filenames)
    y_pred_ensemble = ensemble_prediction(y1, y2, f_measure_y1, f_measure_y2, weighted)

    if print_precision:
        print('[CONFUSION MATRIX]\n')
        precision_y_pred_ensemble, _, f_measure_y_pred_ensemble = result.calculate_support(
            y_pred_ensemble, filenames,
            print_confusion_matrix=print_confusion_matrix)

        print('[ACCURACY CNN]: %s' % precision_y1)
        print('[ACCURACY ANN]: %s' % precision_y2)
        print('[ACCURACY NEW]: %s' % precision_y_pred_ensemble)

    return y_pred_ensemble


def ensemble_prediction(y1, y2, f_measure_y1, f_measure_y2, weighted):
    y_pred_ensemble = [[0 for _ in range(len(y_ann[0]))] for _ in range(len(y1))]

    for i in range(len(y1)):
        for j in range(len(y1[i])):
            if weighted:
                y_pred_ensemble[i][j] = (y1[i][j] * f_measure_y1 + y2[i][j] * f_measure_y2) / 2
            else:
                y_pred_ensemble[i][j] = (y1[i][j] + y2[i][j]) / 2

    return y_pred_ensemble


def save_test(test):
    """
    Save the list of test file in a txt

    :param test:
    :return:
    """
    with open(TEST_DATA_PATH, 'w') as f:
        for item in test:
            f.write('%s\n' % item)


def convert_to_txt(filenames):
    return [filenames[i][:-5] + '.txt' for i in range(len(filenames))]


def convert_dataset_to_ann(test, train=None, validation=None):
    """
    Convert file .png to .txt

    :param train: List[str]
    :param validation: List[str]
    :param test: List[str]
    :return: List[str]
    """

    if train is not None:
        train_ann = convert_to_txt(train)
    else:
        train_ann = None

    if validation is not None:
        val_ann = convert_to_txt(validation)
    else:
        val_ann = None

    test_ann = convert_to_txt(test)

    return train_ann, val_ann, test_ann


if __name__ == "__main__":

    """ Load test set from file """
    test_cnn = readlist(TEST_DATA_PATH)  # List of image filename end with .png

    _, _, test_ann = convert_dataset_to_ann(test_cnn)  # List of signal end with .txt

    """ Load ann and cnn models """
    ann_model = ann.load_ann_model()
    cnn_model = cnn.load_cnn_model()

    """ Predict new classes and combine the answers """
    y_ann = ann.predict_model(model=ann_model, filenames=test_ann)
    y_cnn = cnn.predict_model(model=cnn_model, filenames=test_cnn)
    y_new = combine_result(y_cnn, y_ann, test_ann, weighted=True, print_precision=True, print_confusion_matrix=True)
