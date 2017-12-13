import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridgeError, CvBridge
from irl.msg import Edwin_Shape
from irl.srv import arm_cmd
from sensor_msgs.msg import Image
from std_msgs.msg import String

import get_sudoku


class SudokuMain(object):
    """
    The master class of the game Sudoku
    """

    def __init__(self, n=4):
        # init ROS nodes
        rospy.init_node('sudoku_gamemaster', anonymous=True)

        # init ROS subscribers to camera and status
        rospy.Subscriber('arm_cmd_status', String, self.status_callback, queue_size=10)
        rospy.Subscriber('writing_status', String, self.writing_status_callback, queue_size=20)
        self.image_sub = rospy.Subscriber("usb_cam/image_raw", Image, self.img_callback)

        self.write_pub = rospy.Publisher('/write_cmd', Edwin_Shape, queue_size=10)

        # For the image
        self.bridge = CvBridge()

        # Edwin's status: 0 = busy, 1 = free
        self.status = 0
        self.writing_status = 1

        # Video frame
        self.frame = None

        # x, y, z positions of Edwin
        self.x = 0
        self.y = 0
        self.z = 0

        # Sudoku size, either 4 or 9
        self.n = n

        # Captured image from the camera
        self.sudoku_image = None

        # Sudoku object
        self.sudoku = None

    def status_callback(self, data):
        print "Arm status callback", data.data
        if data.data == "busy" or data.data == "error":
            print "busy"
            self.status = 0
        elif data.data == "free":
            print "free"
            self.status = 1

    def writing_status_callback(self, data):
        print "writing status callback", data.data
        if data.data == "writing":
            print "busy"
            self.writing_status = 0
        elif data.data == "done":
            print "free"
            self.writing_status = 1

    def img_callback(self, data):
        """
        Get image from usb camera
        :param data: image
        :return: None
        """
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
        while self.status == 0 or self.writing_status == 0:
            if self.status == 0:
                print "busy"
            else:
                print "writing"
            pass

    def move_xyz(self, x, y, z):
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
        """
        Move edwin to the center position where it can take a good picture
        :return: None
        """
        # self.move_xyz(x=-1500, y=3100, z=4700)
        # self.check_completion()
        # self.move_head(hand_value=3400, wrist_value=4280)

        self.move_xyz(x=0, y=3400, z=4700)
        self.check_completion()
        self.move_head(hand_value=3350, wrist_value=4020)

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

    def write_number(self, row, col, number):
        x, y, z = self.get_cordinates(row, col)
        self.move_xyz(x, y, z + 200)
        self.check_completion()
        data = Edwin_Shape(x=x, y=y, z=z - 40, shape=str(number))
        self.write_pub.publish(data)
        self.check_completion()
        self.move_xyz(x, y, z + 200)
        self.check_completion()

    def write_numbers(self):
        solution = self.sudoku.solution
        for cell in solution:
            row, col, number = cell.get_rc_num()
            print row, col, number
            self.write_number(row, col, number)
            if not self.continue_or_not():
                break

    def test_write_numbers(self):
        for i in range(4):
            for j in range(4):
                print "writing", i, j
                self.write_number(i, j, 8)
                self.check_completion()
                # if not self.continue_or_not():
                #     break

    def continue_or_not(self):
        answer = raw_input("Do you want me to continue (yes/no)?\n")
        answer = answer.lower()
        print answer
        if answer == "yes" or answer == "y":
            return True
        return False

    def run(self):
        """
        Main function that runs everything
        :return: None
        """
        # self.capture_video()
        self.move_to_center()
        self.check_completion()
        # self.capture_piture()
        # self.check_completion()
        # if self.continue_or_not():
        #     self.sudoku = get_sudoku.from_image(im=self.sudoku_image, n=self.n)
        #     self.sudoku.print_sudoku()
        #     self.write_numbers()
        #     self.move_to_center()

        self.write_number(3, 3, 8)
        # self.move_xyz(x=0, y=3400, z=4700)
        self.test_write_numbers()

    def get_cordinates(self, row, col):
        if row == 0 and col == 0:
            x = -1500
            y = 6500
            z = -780

        elif row == 0 and col == 1:
            x = -400
            y = 6500
            z = -770

        elif row == 0 and col == 2:
            x = 800
            y = 6500
            z = -770

        elif row == 0 and col == 3:
            x = 1900
            y = 6500
            z = -770

        elif row == 1 and col == 0:
            x = -1500
            y = 5400
            z = -765

        elif row == 1 and col == 1:
            x = -450
            y = 5400
            z = -765

        elif row == 1 and col == 2:
            x = 700
            y = 5400
            z = -760

        elif row == 1 and col == 3:
            x = 1900
            y = 5400
            z = -760

        elif row == 2 and col == 0:
            x = -1500
            y = 4200
            z = -740

        elif row == 2 and col == 1:
            x = -400
            y = 4300
            z = -740

        elif row == 2 and col == 2:
            x = 700
            y = 4300
            z = -745

        elif row == 2 and col == 3:
            x = 1900
            y = 4400
            z = -750

        elif row == 3 and col == 0:
            x = -1500
            y = 3200
            z = -735

        elif row == 3 and col == 1:
            x = -450
            y = 3200
            z = -738

        elif row == 3 and col == 2:
            x = 700
            y = 3200
            z = -730

        elif row == 3 and col == 3:
            x = 1900
            y = 3200
            z = -735

        else:
            x = 0
            y = 3400
            z = 4700

        return x, y, z


if __name__ == '__main__':
    sudoku_game = SudokuMain(n=4)
    sudoku_game.run()
