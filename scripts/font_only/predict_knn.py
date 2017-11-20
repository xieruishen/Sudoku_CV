import numpy as np

import cv2

import image_helper as imhelp
from training_knn import clf


def recognize(digit_image=None, will_show_img=True):
    # Read the input image
    im = cv2.imread(digit_image)

    if im is None:
        return

    # TODO: Resize if image is too big
    if False:
        im = cv2.resize(im, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)

    # Output image
    out = np.zeros(im.shape, np.uint8)

    # Image process: blur, binary, threshold
    blur = cv2.GaussianBlur(im, (11, 11), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    imhelp.show_image(th)

    # Find contours in the image
    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 20:
                # Draw the rectangles in the original image
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Put the image into a black image
                black_image = imhelp.put_to_black(th, x, y, w, h)

                # Resize the image
                roi = cv2.resize(black_image, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (10, 10), iterations=2)

                # imhelp.show_image(roi)

                # Reshape the image
                roi.shape = (1, 784)
                roi = np.float32(roi)

                # Predict the image
                retval, results, neigh_resp, dists = clf.findNearest(roi, k=10)

                # Put text on the output image
                string = str(int((results.ravel())))
                cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

    if will_show_img:
        # Show the output image
        imhelp.show_images([im, out])


if __name__ == "__main__":
    # recognize("test_imgs/photo_1.jpg", True)
    recognize("test_imgs/photo_3.jpg", True)
