import numpy as np
import cv2
from matplotlib import pyplot as plt

im = cv2.imread('sudoku.png', 0)
#im_th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11, 2)
ret, im_th = cv2.threshold(im, 90, 255, cv2.THRESH_BINARY_INV)#Second argument is the threshold value which is used to classify the pixel values
#Third argument is the maxVal: the value to be given if pixel value is more than (sometimes less than) the threshold value
im_contour, ctrs, hier = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#calculate the moment
cnt = ctrs[0]
M = cv2.moments(cnt)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
#im3 = cv2.drawContours(im_contour,[box],0,(-1,255,3),2)
im3 = cv2.drawContours(im, ctrs, -1, 255, 3)
#plot the images
plt.subplot(121),plt.imshow(im_th,cmap = 'gray')
plt.title('Threshold image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(im3,cmap = 'gray')
plt.title('Image with contour'), plt.xticks([]), plt.yticks([])
plt.show()
