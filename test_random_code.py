import cv2
img = cv2.imread('photo_5.jpg',0)
img = cv2.resize(img,None,fx=0.4, fy=0.4, interpolation = cv2.INTER_AREA)
print(img.size)
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
print(M.size)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('rotate', dst)
cv2.waitKey(0)


# (h, w) = image.shape[:2]
# center = (w / 2, h / 2)
#
# # rotate the image by 180 degrees
# M = cv2.getRotationMatrix2D(center, 180, 1.0)
# rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("rotated", rotated)
# cv2.waitKey(0)
