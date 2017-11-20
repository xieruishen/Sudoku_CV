import math
import numpy as np

import cv2

from training_knn import *

# Image process: blur, binary, threshold
im = cv2.imread('test_imgs/photo_2.jpg')
im = cv2.resize(im, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
out = np.zeros(im.shape, np.uint8)
rows, cols, channel = im.shape

# rotate the image
# #M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
# im = cv2.warpAffine(im,M,(cols,rows))

blur = cv2.GaussianBlur(im, (11, 11), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# ret,th1 = cv2.threshold(gray,127, 255,cv2.THRESH_BINARY_INV)
cv2.imshow('out', th)
cv2.waitKey(0)

im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print contours
for cnt in contours:
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 20:
            # Draw the rectangles in the original image
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put the image into a black image
            roi = th[y:y + h, x:x + w]
            if h > w:
                b_size = int(h * 1.6)
            else:
                b_size = int(w * 1.6)

            b_y = int((b_size - h) / 2)
            b_x = int((b_size - w) / 2)
            black_image = np.zeros((b_size, b_size), np.uint8)
            black_image[b_y: b_y + h, b_x: b_x + w] = roi

            roi_small = cv2.resize(black_image, (28, 28), interpolation=cv2.INTER_AREA)
            # show_image(black_image)
            # roi = cv2.dilate(roi, (3, 3))

            # Calculate the HOG features
            # roi_hog_fd = hog(roismall, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            # roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float32'))

            roi_small.shape = (1, 784)
            roi_small = np.float32(roi_small)
            # cv2.imshow('roi',black_image)
            # cv2.waitKey(0)
            retval, results, neigh_resp, dists = clf.findNearest(roi_small, k=8)
            string = str(int((results.ravel())))
            cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

cv2.imshow('im', im)
cv2.imshow('out', out)
cv2.waitKey(0)
