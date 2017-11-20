import numpy as np

import cv2
from skimage.feature import hog
from sklearn import preprocessing


def extract_hog_28(samples):
    list_hog_fd = []
    for i, sample in enumerate(samples):
        if i % 2000 == 0:
            print "Working on sample #%i" % i
        fd = hog(sample.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualise=False, block_norm='L2-Hys')
        list_hog_fd.append(fd)

    hog_features = np.array(list_hog_fd, 'float32')

    # Normalize the features
    pp = preprocessing.StandardScaler().fit(hog_features)
    hog_features = pp.transform(hog_features)

    return hog_features, pp


def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
