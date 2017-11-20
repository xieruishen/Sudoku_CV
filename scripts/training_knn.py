import cv2

from training_data import samples, responses

print "Performing the training"
clf = cv2.ml.KNearest_create()

clf.train(samples, cv2.ml.ROW_SAMPLE, responses)

print "KNN training complete"
