import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sudoku.png',0)
#Threshold the image
ret, im_th = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
#im_th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,15, 1)

#edge detection
edges = cv2.Canny(im_th,20,100)
plt.subplot(121),plt.imshow(im_th,cmap = 'gray')
plt.title('Threshold image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.imshow(edges, cmap = 'gray', interpolation = 'bicubic')

# Find contours in the image
im_contour, ctrs, hier = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print im_contour

#calculate moment
cnt = ctrs[0]
M = cv2.moments(cnt)

#contour approximation
epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
# Get rectangles contains each contour
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(im_th,[box],0,(0,0,255),2)


plt.show()


img = cv2.imread('star.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
