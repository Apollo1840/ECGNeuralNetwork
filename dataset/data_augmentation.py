import cv2


def cropping(image, filename, size=(128, 128)):
    """
        :param image: the image to crop
        :param filename:
        :param size: prefered size
        :return:
    """
    img_filename = filename[:-4]
    
    # Left Top Crop
    crop = image[:96, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'leftTop' + '.png', crop)
    
    # Center Top Crop
    crop = image[:96, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'centerTop' + '.png', crop)
    
    # Right Top Crop
    crop = image[:96, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'rightTop' + '.png', crop)
    
    # Left Center Crop
    crop = image[16:112, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'leftCenter' + '.png', crop)
    
    # Center Center Crop
    crop = image[16:112, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'centerCenter' + '.png', crop)
    
    # Right Center Crop
    crop = image[16:112, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'rightCenter' + '.png', crop)
    
    # Left Bottom Crop
    crop = image[32:, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'leftBottom' + '.png', crop)
    
    # Center Bottom Crop
    crop = image[32:, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'centerBottom' + '.png', crop)
    
    # Right Bottom Crop
    crop = image[32:, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(img_filename + 'rightBottom' + '.png', crop)


def cropping2(image, filename, size):
    """
        :param image: the image to crop
        :param filename:
        :param size: prefered size
        :return:
    """
    # Left Top Crop
    crop = image[:96, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '1' + '.png', crop)

    # Center Top Crop
    crop = image[:96, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '2' + '.png', crop)

    # Right Top Crop
    crop = image[:96, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '3' + '.png', crop)

    # Left Center Crop
    crop = image[16:112, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '4' + '.png', crop)

    # Center Center Crop
    crop = image[16:112, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '5' + '.png', crop)

    # Right Center Crop
    crop = image[16:112, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '6' + '.png', crop)

    # Left Bottom Crop
    crop = image[32:, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '7' + '.png', crop)

    # Center Bottom Crop
    crop = image[32:, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '8' + '.png', crop)

    # Right Bottom Crop
    crop = image[32:, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '9' + '.png', crop)


def augmentated_filenames2(filenames):
    """
    cropping() first to understand this function

    :param filenames: List[str]
    :return:
    """

    res_filenames = []
    for filename in filenames:
        res_filenames.extend([filename[:-5] + str(i) + '.png' for i in range(10)])

    return res_filenames

