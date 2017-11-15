import cv2
import numpy as np
from sklearn import datasets
from training_data import samples, responses

model = cv2.ml.KNearest_create()

model.train(samples, cv2.ml.ROW_SAMPLE, responses)

print "training complete"
