import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from collections import Counter


#######   training part    ###############
samples = np.loadtxt('general_mnist_samples.data',np.float32)
responses = np.loadtxt('general_mnist_responses.data',np.float32)
# print samples[0]
# for i in range(1):
#     print responses[i]
#     my_sample = samples[i]
#     my_sample.shape = (28, 28)
#
#     cv2.imshow('im', my_sample)
#     cv2.waitKey(0)

model = cv2.ml.KNearest_create()

model.train(samples,cv2.ml.ROW_SAMPLE,responses)

print "training complete"
