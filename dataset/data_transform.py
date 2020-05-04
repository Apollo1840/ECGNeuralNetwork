import matplotlib.pyplot as plt
import cv2
from .data_augmentation import cropping, cropping2


def signals_to_im(signals, directory='.'):
    """

    :param signals: List[List[float]]: list of signals
    :param directory:
    :return: None
    """

    for count, i in enumerate(signals):
        filename = directory + '/' + str(count) + '.png'
        signal_to_im(i, filename)


def signal_to_im(sig, img_path, resize=128, use_cropping=True):
    """
    plot the signal and save the image.
    with preprocessing steps

    :param sig: List[float]
    :param img_path: str. path/to/save/the/img
    :param resize: Bool, resize or not.
    :param use_cropping: Bool
    :return: None
    """

    im_gray = sig_to_im(sig, img_path)

    if resize:
        im_gray = cv2.resize(im_gray, (resize, resize), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(img_path, im_gray)

    if resize and use_cropping:
        cropping(im_gray, img_path, size=(resize, resize))


def sig_to_im(sig, img_path):
    """
    plot the signal and save the image.

    :param sig: List[float]
    :param img_path: str. path/to/save/the/img
    :return: image info from cv2.imread()
    """

    fig = plt.figure(frameon=False)

    plt.plot(sig)
    plt.xticks([]), plt.yticks([])

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    fig.savefig(img_path)

    plt.cla()
    plt.clf()
    plt.close('all')

    im_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return im_gray


def signal_to_im2(sig, img_path, resize=128, use_cropping=True):
    """
    plot the signal and save the image.
    with preprocessing steps

    :param sig: List[float]
    :param img_path: str. path/to/save/the/img
    :param resize: Bool, resize or not.
    :param use_cropping: Bool
    :return: None
    """

    im_gray = sig_to_im2(sig, img_path)

    if resize:
        im_gray = cv2.resize(im_gray, (resize, resize), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(img_path, im_gray)

    if resize and use_cropping:
        cropping2(im_gray, img_path, size=(resize, resize))


def sig_to_im2(sig, img_path):
    """
    plot the signal and save the image.

    :param sig: List[float]
    :param img_path: str. path/to/save/the/img
    :return: image info from cv2.imread()
    """

    plot_x = [i * 1 for i in range(len(sig))]
    plot_y = sig

    fig = plt.figure(frameon=False)

    plt.plot(plot_x, plot_y)
    plt.xticks([]), plt.yticks([])

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    fig.savefig(img_path)

    plt.cla()
    plt.clf()
    plt.close('all')

    im_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return im_gray
