"""
Modified by Khang Vu, 2017

This script takes the SVC classifier called svc_digits_cls.pkl
to predict an image with numbers on it. It uses SVC method by sklearn.

Related: training_data.py, training_svc.py
"""

import numpy as np
import cv2
from image_helper import *
from skimage.feature import hog
from sklearn.externals import joblib


def recognize(classifer_path=None, digit_image=None, will_show_img=True):
    # Load the classifier
    clf, pp = joblib.load(classifer_path)

    # Read the input image
    im = cv2.imread(digit_image)

    if im is None:
        return

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    im2, contours, hierarchy = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)

            # Draw the rectangles in the original image
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put the image into a black image
            roi = im_th[y:y + h, x:x + w]
            if h > w:
                b_size = int(h * 1.6)
            else:
                b_size = int(w * 1.6)

            b_y = int((b_size - h) / 2)
            b_x = int((b_size - w) / 2)
            black_image = np.zeros((b_size, b_size), np.uint8)
            black_image[b_y: b_y + h, b_x: b_x + w] = roi

            show_image(black_image)

            # Resize the image
            roi = cv2.resize(black_image, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float32'))

            # Predict image
            nbr = clf.predict(roi_hog_fd)

            # Put yellow text (predicted digit) above  digit
            print int(nbr[0])
            cv2.putText(im, str(int(nbr[0])), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    if will_show_img:
        cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
        cv2.imshow("Resulting Image with Rectangular ROIs", im)
        cv2.waitKey()


if __name__ == "__main__":
    # recognize("svc_digits_cls.pkl", "test_imgs/photo_4.jpg", True)
    recognize("svc_digits_cls.pkl", "test_imgs/photo_1.jpg", True)
