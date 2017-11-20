"""
Modified by Khang Vu, 2017

This script generates a Linear SVC classifier (by sklearn).
It takes samples and responses from training_data.py to train
and dumps the classifier named svc_digits_cls.pkl at the end.

Related: training_data.py, predict_svc.py
"""

from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from image_helper import *
from training_data import samples, responses


hog_features, pp = extract_hog_28(samples)

print "Performing the training"
clf = LinearSVC()
clf.fit(hog_features, responses)

print "Saving the classifier"
joblib.dump((clf, pp), "svc_digits_cls_no_mnist.pkl", compress=3)

print "SVC training complete"
