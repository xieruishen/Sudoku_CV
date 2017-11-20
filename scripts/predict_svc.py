"""
Modified by Khang Vu, 2017

This script takes the SVC classifier called svc_digits_cls.pkl
to predict an image with numbers on it. It uses SVC method by sklearn.

Related: training_data.py, training_svc.py
"""

import numpy as np
import cv2
import image_helper as imhelp
from skimage.feature import hog
from sklearn.externals import joblib


def recognize(classifer_path=None, digit_image=None, will_show_img=True):
    # Load the classifier
    clf, pp = joblib.load(classifer_path)

    # Read the input image
    im = cv2.imread(digit_image)

    if im is None:
        return

    # If the image is too big, resize it
    max_size = 800.0
    if im.shape[0] > max_size or im.shape[1] > max_size:
        if im.shape[0] > im.shape[1]:
            ratio = max_size / im.shape[0]
        else:
            ratio = max_size / im.shape[1]
        im = cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

    # Output image
    out = np.zeros(im.shape, np.uint8)

    # Image process: blur, binary, threshold
    blur = cv2.GaussianBlur(im, (11, 11), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    imhelp.show_image(th)

    # Find contours in the image
    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 20:
                # Draw the rectangles in the original image
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Put the image into a black image
                black_image = imhelp.put_to_black(th, x, y, w, h)

                # Resize the image
                roi = cv2.resize(black_image, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (10, 10))

                # imhelp.show_image(roi)

                # Calculate the HOG features
                roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14),
                                 cells_per_block=(1, 1), visualise=False, block_norm='L2-Hys')
                roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float32'))

                # Predict the image
                nbr = clf.predict(roi_hog_fd)

                # Put text on the output image
                string = str(int(nbr[0]))
                cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

    if will_show_img:
        # Show the output image
        imhelp.show_images([im, out])


if __name__ == "__main__":
    recognize("svc_digits_cls.pkl", "test_imgs/photo_2.jpg", True)
    # recognize("svc_digits_cls_no_mnist.pkl", "test_imgs/photo_1.jpg", True)

