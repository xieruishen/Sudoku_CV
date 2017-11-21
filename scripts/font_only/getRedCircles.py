import cv2
import numpy as np
import itertools
import math
from sets import Set
import time

'''
Author:Nick Steelman
Use: This script will take a video and take out a certain color (right now red)
and try and find a square of filled circles of that color, it will currently
display the original image with a box connecting the circles, the mask showing
the red taken from the image, and the stransformation whoing what is inside the
box
Deps:
You can either import or run this script from main. To get it running, you can
instantiate a circleFinder object and call run()
'''

error = 0.1  # error allowed in the angles for the rectangle


class circleFinder:
    '''This is the main object that handle getting the image, processing it,
    and finding the square insid ethe image'''

    permCorners = None
    kernel = np.ones((3, 3), np.uint8)
    outX = 300  # output dimensions for the box found
    outY = 300
    new = True
    movementError = 50
    valid = np.array([0, 0, 0, 0])
    counters = np.array([0, 0, 0, 0])
    counterLim = 20

    def __init__(self, videoNum=0):
        self.error = error
        self.cap = cv2.VideoCapture(videoNum)  # get video recorder
        a, b, c = (0, 100, 100)
        x, y, z = (0, 255, 255)
        self.lower_l = np.array((a, b, c), dtype="uint8")  # determine which color range to select (HSV)
        self.upper_l = np.array((x, y, z), dtype="uint8")
        a, b, c = (165, 50, 50)
        x, y, z = (175, 255, 255)
        self.lower_h = np.array((a, b, c), dtype="uint8")
        self.upper_h = np.array((x, y, z), dtype="uint8")

    def readFrame(self):
        '''Get a frame from the camera'''

        ret = 0
        while not ret:
            ret, frame = self.cap.read()
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        self.frame = frame

    def getMask(self):
        '''This part is a little tricky and requires some background on erosion
        and dialation.
        Look here : https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        I am converting the image to HSV, grabbing only colors in the select
        range as a binary, eroding once to reduce noise. Then I am computing
        the difference between erosion and dialation. This amkes the circles
        really easy to find by the CV2 detection algorithm.
        '''

        # Convert BGR to HSV
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV

        mask1 = cv2.inRange(hsv, self.lower_l, self.upper_l)
        mask2 = cv2.inRange(hsv, self.lower_h, self.upper_h)
        mask = cv2.bitwise_or(mask1, mask2)
        # Threshold the HSV image to get only blue colors
        mask = cv2.erode(mask, self.kernel, iterations=1)
        self.mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel)

    def getCircles(self):
        '''Get circles from the mask'''

        return cv2.HoughCircles(self.mask, cv2.HOUGH_GRADIENT, 1, 85,
                                param1=25, param2=15, minRadius=0, maxRadius=40)

    def drawCircles(self, circles):
        '''Draw circles on the frame'''

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(self.frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(self.frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    def drawFrame(self, points):
        '''Draw a borader betweeen points on the frame'''
        points = [tuple(point) for point in points]
        cv2.line(self.frame, points[0], points[1], (0, 0, 255), 4)
        cv2.line(self.frame, points[0], points[2], (0, 0, 255), 4)
        cv2.line(self.frame, points[3], points[1], (0, 0, 255), 4)
        cv2.line(self.frame, points[3], points[2], (0, 0, 255), 4)

    def getCenter(self, points):
        height, width, channels = self.frame.shape
        center = np.mean(points, axis=0)
        return center[0] - width / 2, height / 2 - center[1]

    def getRotation(self, points):
        return math.atan((points[1][1] - points[0][1]) / abs(points[1][0] - points[0][0]))

    def findSquare(self, circles):
        self.drawCircles(circles)
        ret, corners = getProbFour(circles[0, :])
        if ret:
            self.valid = np.array([1, 1, 1, 1])
            self.permCorners = np.squeeze(changePoints(corners))
            pts2 = np.float32([[0, 0], [self.outX, 0], [0, self.outY], [self.outX, self.outY]])  # newe points
            self.M = cv2.getPerspectiveTransform(self.permCorners, pts2)  # transform

    def fitPerm(self, circles):
        '''This code will track an arbitrary amount of circles. When one of the
        circles has excessive distance to the closest circle, the point is dropped'''
        valid = np.squeeze(np.argwhere(self.valid))  # get index of which points are valid
        # print(self.valid)
        a = np.array(circles[0, :])[:, 0:2]  # get array of circle centers
        b = self.permCorners[valid]  # get valied points from the saved rect
        # if b.shape
        if len(b.shape) != 2:
            self.trackOne(a, b, valid)
            return
        c = a - b[:, None]  # subtract each point from each circle will make c X p X 2 matrix
        c = np.hypot(c[:, :, 0], c[:, :, 1])  # get hypot of each point
        dist = np.amin(c, axis=1)  # get min for each point
        notError = np.argwhere(dist < self.movementError)  # where we are good
        Error = np.argwhere(dist >= self.movementError)  # where the circle is lost
        if len(notError):
            closest = np.argmin(c, axis=1)[notError]  # get the
            self.permCorners[valid[notError]] = a[closest]  # reset the corners

        for i in valid[Error]:
            if self.counters[i] >= self.counterLim:
                self.valid[i] = 0
            else:
                self.counters[i] += 1
        for i in valid[notError]:
            if self.counters[i] > 0:
                self.counters[i] -= 1

    def trackOne(self, a, b, valid):
        c = a - b  # subtract each point from each circle will make c X p X 2 matrix
        c = np.hypot(c[:, 0], c[:, 1])  # get hypot of each point
        dist = np.amin(c)  # get min for each point
        # print dist

        if dist < self.movementError:
            closest = np.argmin(c)  # get the
            self.permCorners[valid] = a[closest]  # reset the corners
            if self.counters[valid] > 0:
                self.counters[valid] -= 1
        else:
            if self.counters[valid] >= self.counterLim:
                self.valid[valid] = 0
            else:
                self.counters[valid] += 1

    def runOnce(self):

        self.readFrame()
        self.getMask()
        circles = self.getCircles()
        if circles is not None:
            if self.new or self.permCorners is None:
                self.findSquare(circles)
                self.new = False
            else:
                self.fitPerm(circles)
                pts2 = np.float32([[0, 0], [self.outX, 0], [0, self.outY], [self.outX, self.outY]])  # newe points
                self.M = cv2.getPerspectiveTransform(self.permCorners, pts2)  # transform
                if sum(self.valid) == 0:
                    self.new = True

        if sum(self.valid) == 4:
            self.drawFrame(self.permCorners)
            self.board = cv2.warpPerspective(self.frame, self.M, (self.outX, self.outY))  # new image
            cv2.imshow('board', self.board)
        cv2.imshow('mask', self.mask)
        cv2.imshow('frame', self.frame)
        k = cv2.waitKey(5) & 0xFF

    def run(self):
        '''This is the main script to run everything. it read the frame, gets the
        mask and circles ,then it will try to find the square and save it to the
        object if successfull. If there is a set of points saved to memory it
        will draw the rectangle on the screen and draw a new frame of just the
        portion in the points'''

        while 1:
            self.readFrame()
            self.getMask()
            circles = self.getCircles()
            if circles is not None:
                if self.new or self.permCorners is None:
                    self.findSquare(circles)
                    self.new = False
                else:
                    self.fitPerm(circles)
                    if sum(self.valid) == 0:
                        self.new = True

            if sum(self.valid) == 4:
                self.drawFrame(self.permCorners)
                self.board = cv2.warpPerspective(self.frame, self.M, (self.outX, self.outY))  # new image
                cv2.imshow('board', self.board)
            cv2.imshow('mask', self.mask)
            cv2.imshow('frame', self.frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()


def slope(x1, y1, x2, y2):
    '''Absolute value of slope between 2 points'''
    m = (y2 - y1) / (x2 - x1)
    return abs(m)


def whichSlope(node1, node2):
    '''In order to not deal with slopes of like 1,000 for verticle lines, this
    sfunction determines whether delta x or delta y is bigger and returns the
    slope of  the smaller over the bigger. So it will always be below 0.5'''
    if (abs(node1[0] - node2[0]) > abs(node1[1] - node2[1])):
        return 1, slope(node1[0], node1[1], node2[0], node2[1])
    else:
        return 0, slope(node1[1], node1[0], node2[1], node2[0])
    return cslope


def getProbFour(circles):
    '''
    This script will find a rectagle given a list of points. It starts off with
    first point and check all the others against each other for a right angle
    Then checks the ramaining points for a right angle with the same 2 points
    '''

    if len(circles) < 4:
        return 0, None
    slopeDict = {}
    circles = np.array(circles, dtype="float32")
    cNode = None
    for i in circles:
        indivNode = dict()
        dirDict = dict()
        for j in circles:
            if (i == j).all():
                pass
            else:
                d, cslope = whichSlope(i, j)
                # print(indivNode)
                for k, v in indivNode.iteritems():
                    if k + error > cslope and k - error < cslope and d != dirDict[k]:
                        ret, sq = getRightAngle(indivNode[k], j, circles, i)
                        if ret: return 1, [tuple(i[0:2]), tuple((indivNode[k])[0:2]), tuple(j[0:2]), tuple(sq[0:2])]
                if cslope:
                    # print(cslope)
                    indivNode[cslope] = j
                    dirDict[cslope] = d
    return 0, None


def getRightAngle(node1, node2, circles, originalNode):
    '''Determine if any of the points in circles contain a right angle between
    node 1 and 2'''

    for i in circles:
        if any((i == node).all() for node in [node1, node2, originalNode]):
            pass
        elif isRightAngle(i, node1, node2):
            return 1, i
    return 0, None


def isRightAngle(vertex, node1, node2):
    '''Check to see if the nodes for a right angle'''

    d, k = whichSlope(vertex, node1)
    d2, cslope = whichSlope(vertex, node2)

    return k + error > cslope and k - error < cslope and d != d2


#


def changePoints(points):
    '''Given a set of four points, determine the top right and bottom left nodes'''

    dist = np.sum(points, axis=1)
    mindex = np.argmin(dist)
    if mindex == 0 or mindex == 3:
        if points[1][0] < points[2][0]:
            return np.float32([points[mindex], points[2], points[1], points[3 - mindex]])
        else:
            return np.float32([points[mindex], points[1], points[2], points[3 - mindex]])
    else:
        if points[0][0] < points[3][0]:
            return np.float32([points[mindex], points[3], points[0], points[3 - mindex]])
        else:
            return np.float32([points[mindex], points[0], points[3], points[3 - mindex]])


if __name__ == "__main__":
    circleF = circleFinder()
    circleF.run()
