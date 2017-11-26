import numpy as np

import cv2
import rospy
import time

from cv_bridge import CvBridgeError, CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from irl.srv import arm_cmd
import get_sudoku

class SudokuMain(object):
    """
    The master class of the game Set
    """

    def __init__(self, n=4):
        # init ROS nodes
        rospy.init_node('sudoku_gamemaster', anonymous=True)

        # init ROS subscribers to camera and status
        self.status_sub = rospy.Subscriber('arm_cmd_status', String, self.status_callback, queue_size=10)
        self.image_sub = rospy.Subscriber("usb_cam/image_raw", Image, self.img_callback)
        self.pub = rospy.Publisher('arm_cmd', String, queue_size=10)

        # For the image
        self.bridge = CvBridge()

        self.status = 0
        self.frame = None
        self.x = 0
        self.y = 0
        self.z = 0

        self.num_cards = n
        self.sudoku_image = None
        self.sudoku = None

    def status_callback(self, data):
        print "Arm status callback", data.data
        if data.data == "busy" or data.data == "error":
            print "busy"
            self.status = 0
        elif data.data == "free":
            print "free"
            self.status = 1

    def img_callback(self, data):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e

    def request_cmd(self, cmd):
        rospy.wait_for_service('arm_cmd', timeout=15)
        cmd_fnc = rospy.ServiceProxy('arm_cmd', arm_cmd)
        print "I have requested the command"

        try:
            cmd_fnc(cmd)
            print "command done"

        except rospy.ServiceException, e:
            print ("Service call failed: %s" % e)

    def check_completion(self):
        """
        Makes sure that actions run in order by waiting for response from service
        """
        time.sleep(1)
        while self.status == 0:
            pass

    def xyz_move(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        msg = "data: move_to:: %i, %i, %i, %i" % (self.x, self.y, self.z, 0)
        print ("Sending", msg)
        self.request_cmd(msg)

    def move_wrist(self, value):
        msg = "data: rotate_wrist:: " + str(value)
        print ("sending: ", msg)
        self.request_cmd(msg)

    def move_hand(self, value):
        msg = "data: rotate_hand:: " + str(value)
        print ("sending: ", msg)
        self.request_cmd(msg)

    def move_head(self, hand_value=None, wrist_value=None):
        """
        Always move hand first, wrist second
        :param hand_value:
        :param wrist_value:
        :return: None
        """
        self.move_hand(hand_value)
        self.check_completion()
        self.move_wrist(wrist_value)

    def move_to_center(self):

        self.xyz_move(x=-1500, y=3100, z=4700)
        self.check_completion()
        self.move_head(hand_value=3400, wrist_value=4280)

    def capture_piture(self):
        while self.frame is None:
            pass

        self.sudoku_image = self.frame.astype(np.uint8)

        cv2.imshow('Image', self.sudoku_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def capture_video(self):
        while self.frame is None:
            pass

        while self.frame is not None:
            cv2.imshow('Image', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        self.move_to_center()
        self.check_completion()
        self.capture_piture()
        self.check_completion()
        self.sudoku = get_sudoku.from_image(im=self.sudoku_image, n=self.num_cards)
        self.sudoku.print_sudoku()
        # self.capture_video()


if __name__ == '__main__':
    sudoku_game = SudokuMain(n=4)
    sudoku_game.run()