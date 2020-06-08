import matplotlib.pyplot as plt
import cv2

import os
import sys
# add root of this project to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset.data_augmentation import cropping, cropping2


def signals_to_im(signals, directory='.'):
    """

    :param signals: List[List[float]]: list of signals
    :param directory:
    :return: None
    """

    for count, i in enumerate(signals):
        filename = directory + '/' + str(count) + '.png'
        signal_to_im(i, filename)


def signal_to_im(sig, img_path, resize=(128, 128), use_cropping=True):
    """
    plot the signal and save the image.
    with preprocessing steps

    :param sig: List[float]
    :param img_path: str. path/to/save/the/img
    :param resize: Tuple[int] or None/False, resize or not.
    :param use_cropping: Bool
    :return: None
    """

    im_gray = sig_to_im(sig, img_path)

    if resize:
        im_gray = cv2.resize(im_gray, resize, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(img_path, im_gray)

    if resize and use_cropping:
        cropping2(im_gray, img_path, size=resize)


def sig_to_im(sig, img_path):
    """
    plot the signal and save the image.

    :param sig: List[float]
    :param img_path: str. path/to/save/the/img
    :return: image info from cv2.imread()
    """

    fig = plot_clean_sig(sig)
    fig.savefig(img_path)

    plt.cla()
    plt.clf()
    plt.close('all')

    im_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return im_gray


def plot_clean_sig(sig):
    """

    :param sig: List[float]
    :return:
    """
    fig = plt.figure(frameon=False)

    plt.plot(sig)
    plt.xticks([]), plt.yticks([])

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    return fig
