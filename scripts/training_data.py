import cv2
import numpy as np
from sklearn import datasets
from image_helper import *

"""
Generate training data
"""

# Load the dataset (handwriting data MNIST)
dataset = datasets.fetch_mldata("MNIST Original")

# im_4 = dataset.data[0]
# im_4.shape = (28, 28)
# cv2.imshow('out', im_4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

samples = np.array(dataset.data, 'float32')
responses = np.array(dataset.target, 'int')

temp_samples = []
temp_responses = []

# Add more handwriting data from more_chars
for n in range(10):
    for m in range(1, 1000):
        filename = "more_chars/%d_%d.png" % (n, m)

        im = cv2.imread(filename, 0)
        if im is None:
            break

        # blur = cv2.GaussianBlur(im, (5, 5), 0)
        # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
        # th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # ret,th1 = cv2.threshold(gray,127, 255,cv2.THRESH_BINARY_INV)
        im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        count = 0

        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h < 22 and w < 1.5 * h:
                    count += 1

                    # Draw the rectangles in the original image
                    # cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 255, 255), 1)

                    # Put the image into a black image
                    roi = im[y:y + h, x:x + w]
                    if h > w:
                        b_size = int(h * 1.6)
                    else:
                        b_size = int(w * 1.6)
                    black_image = np.zeros((b_size, b_size), np.uint8)
                    b_y = int((b_size - h) / 2)
                    b_x = int((b_size - w) / 2)
                    black_image[b_y: b_y + h, b_x: b_x + w] = roi

                    # Reshape the image
                    roismall = cv2.resize(black_image, (28, 28))
                    roismall.shape = 784
                    roismall = np.float32(roismall)

                    # Add to data set
                    temp_samples.append(roismall)
                    temp_responses.append(n)

        print n, m, count


# Add fonts data from font_images
for n in range(10):
    for m in range(1, 1000000):
        filename = "font_images/%d_%d.jpg" % (n, m)

        im_f = cv2.imread(filename, 0)
        if im_f is None:
            break

        # blur = cv2.GaussianBlur(im, (5, 5), 0)
        # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
        # th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # ret,th1 = cv2.threshold(gray,127, 255,cv2.THRESH_BINARY_INV)
        im2_f, contours, hierarchy = cv2.findContours(im_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0

        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h > 30 and 10 < w < 1.5 * h:
                    count += 1

                    # Draw the rectangles in the original image
                    # cv2.rectangle(im_f, (x, y), (x + w, y + h), (255, 255, 255), 1)

                    # Put the image into a black image
                    roi = im_f[y:y + h, x:x + w]
                    if h > w:
                        b_size = int(h * 1.6)
                    else:
                        b_size = int(w * 1.6)

                    b_y = int((b_size - h) / 2)
                    b_x = int((b_size - w) / 2)
                    black_image = np.zeros((b_size, b_size), np.uint8)
                    black_image[b_y: b_y + h, b_x: b_x + w] = roi

                    # Reshape the image
                    roismall = cv2.resize(black_image, (28, 28))
                    roismall.shape = 784
                    roismall = np.float32(roismall)

                    # Add to data set
                    for _ in range(30):
                        temp_samples.append(roismall)
                        temp_responses.append(n)

        print n, m, count

# samples = np.concatenate((samples, temp_samples), axis=0)
# responses = np.concatenate((responses, temp_responses))

# print samples[70010]
# for i in range(70001, 70002):
#     im3 = samples[i]
#     im3.shape = (28, 28)
#     show_image(im3)

print samples.shape

print "data complete"
