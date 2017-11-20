from collections import Counter

import cv2
import numpy as np
from sklearn import datasets
import image_helper as imhelp

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

                    # Put the image into a black image
                    black_image = imhelp.put_to_black(im, x, y, w, h)

                    # Reshape the image
                    roi = cv2.resize(black_image, (28, 28), interpolation=cv2.INTER_AREA)
                    # if n == 2:
                        # imhelp.show_image(roi)
                    roi.shape = 784
                    roi = np.float32(roi)

                    # Add to data set
                    for _ in range(30):
                        temp_samples.append(roi)
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

                    # Put the image into a black image
                    black_image = imhelp.put_to_black(im_f, x, y, w, h)

                    # Reshape the image
                    roi = cv2.resize(black_image, (28, 28), interpolation=cv2.INTER_AREA)
                    roi.shape = 784
                    roi = np.float32(roi)

                    # Add to data set
                    # for _ in range(30):
                    #     temp_samples.append(roi)
                    #     temp_responses.append(n)

        print n, m, count

samples = np.concatenate((samples, temp_samples), axis=0)
responses = np.concatenate((responses, temp_responses))

# samples = np.array(temp_samples, dtype=np.float32)
# responses = np.array(temp_responses, dtype=np.int)

# print samples[70010]
# for i in range(1, 70000):
#     if responses[i] == 1:
#         im3 = samples[i]
#         im3.shape = (28, 28)
#         imhelp.show_image(im3)

print samples.shape

print "Count of digits in dataset", Counter(responses)

print "Data complete"
