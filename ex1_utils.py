"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 209399294


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Upload the picture
    img = cv2.imread(filename)

    # Convert image to grayscale image
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image to RGB image
    if representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalization
    img = img.astype(np.float64) / 255.0
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # Upload the picture with a function imReadAndConvert that I wrote
    img = imReadAndConvert(filename, representation)

    # Display grayscale image
    if representation == 1:
        plt.imshow(img, cmap='gray')

    # Display RGB image
    if representation == 2:
        plt.imshow(img)

    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # RGB to YIQ conversion matrix
    transMatrix = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])

    # Reshape the input image
    # (height*width)x3
    imRGB = imgRGB.reshape(-1, 3)

    # Matrix multiplication
    imYIQ = np.dot(imRGB, transMatrix.T)

    # Reshape the output image back to its original shape
    imYIQ = imYIQ.reshape(imgRGB.shape)

    return imYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # YIQ to RGB conversion matrix
    transMatrix = np.array([[1.0, 0.956, 0.621],
                            [1.0, -0.272, -0.647],
                            [1.0, -1.106, 1.703]])

    # Reshape the input image
    # (height*width)x3
    imYIQ = imgYIQ.reshape(-1, 3)

    # Matrix multiplication
    imRGB = np.dot(imYIQ, transMatrix.T)

    # Reshape the output image back to its original shape
    imRGB = imRGB.reshape(imgYIQ.shape)

    return imRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    global histOrg, histEq
    # Check if the input image is grayscale or RGB
    # image is grayscale
    if imgOrig.ndim == 2:
        gray = True
    # image is grayscale
    elif imgOrig.ndim == 3 and imgOrig.shape[2] == 1:
        gray = True
   # image is RGB
    else:
        gray = False

    if not gray:
        # Convert RGB to YIQ th a function transformRGB2YIQ that I wrote
        imYIQ = transformRGB2YIQ(imgOrig)

        # equalization to the Y channel of the YIQ image
        # by recursively calling the histogramEqualize function with only the Y channel as input.
        #Update the equalized Y channel in the YIQ image
        imYIQ[:, :, 0] = hsitogramEqualize(imYIQ[:, :, 0])[0]

        # Convert YIQ to RGB th a function transformYIQ2RGB that I wrote
        imEq = transformYIQ2RGB(imYIQ)

    else:

        # Normalization
        imOrig = (imgOrig * 255).astype(np.uint8)

        # Calculate the histogram of the original image
        histOrg, bins = np.histogram(imOrig, bins=256, range=(0, 255))

        # Calculate the cumulative sum of the histogram
        cumsum = histOrg.cumsum()
        # normalize the sum of the histogram
        cumsum_norm = (cumsum - cumsum.min()) / (cumsum.max() - cumsum.min())

        # creates a lookup table (LUT) to map the pixel values of the input image to new pixel values
        LUT_table = (cumsum_norm * 255).astype(np.uint8)

        # Apply the lookup table to the original image
        imEq = LUT_table[imOrig]

        # Calculate the histogram of the equalized image
        histEq, bins = np.histogram(imEq, bins=256, range=(0, 255))

        # Normalization
        imEq = imEq.astype(np.float32) / 255.0


    return imEq, histOrg, histEq




def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # Check if RGB image
    if imOrig.ndim == 3 and imOrig.shape[2] == 3:
        # Convert RGB to YIQ th a function transformRGB2YIQ that I wrote
        imYIQ = transformRGB2YIQ(imOrig)
        #Y channel of the image converted from RGB to YIQ color space
        # using the transformRGB2YIQ function.
        image = imYIQ[:, :, 0]
    else:
        image = imOrig.copy()

    # Normalization
    image = (image * 255).astype(np.uint8)

    # segment division
    segments = 256 // nQuant
    z = np.arange(0, 256, segments)
    z[-1] = 255


    errors = []
    qImages = []

    # Find optimal quantization values
    for i in range(nIter):
        quantization_values = []
        for k in range(nQuant):
            indices = np.logical_and(image >= z[k], image < z[k + 1])
            if np.sum(indices) == 0:
                # Avoid division by zero
                quant_k = 0
            else:
                quant_k = np.mean(image[indices])
            quantization_values.append( quant_k)

        # Quantize image using optimal values
        qImage = np.zeros_like(image)
        for k in range(nQuant):
            indices = np.logical_and(image >= z[k], image < z[k + 1])
            qImage[indices] = quantization_values[k]

        # Calculate error
        error = np.sum(np.power(image - qImage, 2))
        errors.append(error)

        # Update division
        for k in range(1, nQuant):
            z[k] = (quantization_values[k - 1] + quantization_values[k]) / 2

        # result list
        qImages.append(qImage.astype(np.float32) / 255)


    # Check if RGB image
    # convert back from YIQ
    if imOrig.ndim == 3 and imOrig.shape[2] == 3:
        list_rgb = []
        for qImage in qImages:
            qImageYIQ = np.zeros_like(imYIQ)
            qImageYIQ[:, :, 0] = qImage
            qImageYIQ[:, :, 1:] = imYIQ[:, :, 1:]
            list_rgb.append(transformYIQ2RGB(qImageYIQ))
        return list_rgb, errors
    else:
        return qImages, errors

