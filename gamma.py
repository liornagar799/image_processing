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
import numpy as np
import cv2
from ex1_utils import LOAD_GRAY_SCALE


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """


    def gamma_Adjust(gamma):

        # Convert  to float
        gamma = gamma / 100.0

        # gamma correction to the image
        gamma_img = np.power(image / 255.0, gamma)
        gamma_img = np.uint8(gamma_img * 255)

        # Display the image
        cv2.imshow('Gamma Correction', gamma_img)

    # Read the image
    image = cv2.imread(img_path)
    # convert it to the specified representation
    if rep == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a window
    cv2.namedWindow('Gamma Correction', cv2.WINDOW_NORMAL)
    cv2.imshow('Gamma Correction', image)

    # Create a trackbar to adjust the gamma value
    cv2.createTrackbar('Gamma', 'Gamma Correction', 100, 200, gamma_Adjust)

    # Wait for user input
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
