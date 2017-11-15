import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from collections import Counter
import math
from training_knn import model

#Image process: blur, binary, threshold
im = cv2.imread('test_imgs/photo_3.jpg')
im = cv2.resize(im,None,fx=0.4, fy=0.4, interpolation = cv2.INTER_AREA)
out = np.zeros(im.shape,np.uint8)
rows,cols,channel = im.shape

#rotate the image
# #M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
# im = cv2.warpAffine(im,M,(cols,rows))

blur = cv2.GaussianBlur(im,(11,11),0)
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
#thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
#ret,th1 = cv2.threshold(gray,127, 255,cv2.THRESH_BINARY_INV)
cv2.imshow('out',th)
cv2.waitKey(0)

im2, contours,hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# print contours
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>20:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = th[y:y+h,x:x+w]
            # cv2.imshow('roi',roi)
            # cv2.waitKey(0)
            # print(roi.shape)
            black_image = np.zeros((h*2, h*2), np.uint8)
            black_image[h - int(math.floor(0.5*h)):h + int(math.ceil(0.5*h)), h-int(math.floor(0.5*w)): h+ int(math.ceil(0.5*w))] = roi
            roismall = cv2.resize(black_image,(28,28))
            roismall.shape = (1,784)
            roismall = np.float32(roismall)
            # cv2.imshow('roi',black_image)
            # cv2.waitKey(0)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 10)
            string = str(int((results[0][0])))
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

cv2.imshow('im',im)
cv2.imshow('out',out)
cv2.waitKey(0)
