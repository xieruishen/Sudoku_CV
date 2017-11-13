import cv2
import numpy as np
from sklearn import datasets

# Load the dataset (handwriting data MNIST)
dataset = datasets.fetch_mldata("MNIST Original")

# im_4 = dataset.data[0]
# im_4.shape = (28, 28)
# cv2.imshow('out', im_4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
Training Part
"""

samples = np.array(dataset.data, 'float32')
responses = np.array(dataset.target, 'int')

temp_samples = []
temp_responses = []

for n in range(10):
    for m in range(1, 1000):
        filename = "more_chars/%d_%d.png" % (n, m)

        # Add more chars
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
                    # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 1)
                    roi = im[y:y + h, x:x + w]
                    if h > w:
                        b_size = int(h * 1.2)
                    else:
                        b_size = int(w * 1.2)
                    black_image = np.zeros((b_size, b_size), np.uint8)
                    b_y = int((b_size - h) / 2)
                    b_x = int((b_size - w) / 2)
                    black_image[b_y: b_y + h, b_x: b_x + w] = roi
                    roismall = cv2.resize(black_image, (28, 28))

                    roismall.shape = 784
                    roismall = np.float32(roismall)

                    temp_samples.append(roismall)
                    temp_responses.append(n)

        print n, m, count

samples = np.concatenate((samples, temp_samples), axis=0)
responses = np.concatenate((responses, temp_responses))

# print samples[70010]
# for i in range(70001, 70002):
#     im3 = samples[i]
#     im3.shape = (28, 28)
#     cv2.imshow('im', im3)
#     cv2.waitKey(0)
#
# print samples.shape

print "training complete"
