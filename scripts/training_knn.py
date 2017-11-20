from collections import Counter

from sklearn.externals import joblib

from image_helper import *
from training_data import samples, responses


# hog_features, pp = extract_hog_28(samples)

print "Count of digits in dataset", Counter(responses)

print "Performing the training"
clf = cv2.ml.KNearest_create()

clf.train(samples, cv2.ml.ROW_SAMPLE, responses)

print "Saving the classifier"
# joblib.dump((clf, pp), "knn_digits_cls.pkl", compress=3)

print "KNN training complete"
