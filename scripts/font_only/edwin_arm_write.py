#!/usr/bin/env python
import time

import rospy
from irl.msg import *
from irl.srv import arm_cmd
from std_msgs.msg import String


class Writer:
    def __init__(self, init_param=False):
        if init_param:
            pass
        else:
            rospy.init_node('edwin_write', anonymous=True)
            rospy.Subscriber('/write_cmd', Edwin_Shape, self.write_callback, queue_size=10)
            rospy.Subscriber('arm_cmd_status', String, self.status_callback, queue_size=10)

        self.behavior_pub = rospy.Publisher('behaviors_cmd', String, queue_size=10)
        self.arm_pub = rospy.Publisher('arm_cmd', String, queue_size=10)
        self.writing_pub = rospy.Publisher("writing_status", String, queue_size=10)

        print "starting edwin writer...."

        self.status = 0
        self.w = 200
        self.letter_dictionary = {}
        self.make_letter_dictionary()

    def status_callback(self, data):
        print "Arm status callback", data.data
        if data.data == "busy" or data.data == "error":
            print "busy"
            self.status = 0
        elif data.data == "free":
            print "free"
            self.status = 1

    def request_cmd(self, cmd):
        rospy.wait_for_service('arm_cmd', timeout=15)
        cmd_fnc = rospy.ServiceProxy('arm_cmd', arm_cmd)
        print "I have requested the command"

        try:
            resp1 = cmd_fnc(cmd)
            print "command done"


        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def make_letter_dictionary(self):
        w = self.w
        # u for up, d for down

        self.letter_dictionary["a"] = [(w / 2, 0, "u"), (w / 2, 0, "d"), (w, w, "d"),
                                       (w / 2, 0, "u"), (w / 2, 0, "d"), (0, w, "d"),
                                       (w, w / 2, "u"), (w, w / 2, "d"), (0, w / 2, "d")]

        self.letter_dictionary["b"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w, "d"), (w, w, "d"),
                                       (w, w / 2, "u"), (w, w / 2, "d"), (0, w / 2, "d")]

        self.letter_dictionary["c"] = [(0, 0, "u"), (0, 0, "d"), (w, 0, "d"),
                                       (w, w, "d"), (0, w, "d")]

        self.letter_dictionary["d"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, w / 2, "d"), (w, w, "d")]

        self.letter_dictionary["e"] = [(0, 0, "u"), (0, 0, "d"), (w, 0, "d"),
                                       (w, w, "d"), (0, w, "d"), (w, w / 2, "u"), (w, w / 2, "d"), (0, w / 2, "d")]

        self.letter_dictionary["f"] = [(0, 0, "u"), (0, 0, "d"), (w, 0, "d"),
                                       (w, w, "d"), (w, w / 2, "u"), (w, w / 2, "d"), (0, w / 2, "d")]

        self.letter_dictionary["g"] = [(0, 0, "u"), (0, 0, "d"), (w, 0, "d"),
                                       (w, w, "d"), (0, w, "d"), (w / 2, w / 2, "u"),
                                       (w / 2, w / 2, "d"), (0, w / 2, "d")]

        self.letter_dictionary["h"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (0, 0, "u"), (0, 0, "d"), (0, w, "d"),
                                       (w, w / 2, "u"), (w, w / 2, "d"), (0, w / 2, "d")]

        self.letter_dictionary["i"] = [(w / 2, 0, "u"), (w / 2, 0, "d"), (w / 2, w, "d")]

        self.letter_dictionary["j"] = [(w, 0, "u"), (w, 0, "d"), (0, 0, "d"),
                                       (w / 2, 0, "u"), (w / 2, 0, "d"), (w / 2, w, "d"), (w, w, "d")]

        self.letter_dictionary["k"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (0, 0, "u"), (0, 0, "d"), (w, w / 2, "d"), (0, w, "d")]

        self.letter_dictionary["l"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"), (0, w, "d")]

        self.letter_dictionary["m"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w, "d"),
                                       (w / 2, 0, "u"), (w / 2, 0, "d"), (w / 2, w, "d")]

        self.letter_dictionary["n"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w, "d")]

        self.letter_dictionary["o"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w, "d"), (w, w, "d")]

        self.letter_dictionary["p"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"),
                                       (0, w / 2, "d"), (w, w / 2, "d")]

        self.letter_dictionary["q"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"),
                                       (0, w, "d"), (w, w, "d"), (w / 2, w / 2, "u"), (w / 2, w / 2, "d"), (w, w, "d")]

        self.letter_dictionary["r"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"),
                                       (0, w / 2, "d"), (w, w / 2, "d"), (0, w, "d")]

        self.letter_dictionary["s"] = [(w, 0, "u"), (w, 0, "d"), (0, 0, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (w, w / 2, "d"),
                                       (0, w / 2, "d"), (0, w, "d"), (w, w, "d")]

        self.letter_dictionary["t"] = [(w, 0, "u"), (w, 0, "d"), (0, 0, "d"),
                                       (w / 2, 0, "u"), (w / 2, 0, "d"), (w / 2, w, "d")]

        self.letter_dictionary["u"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"), (0, w, "d"), (0, 0, "d")]

        self.letter_dictionary["v"] = [(w, 0, "u"), (w, 0, "d"), (w / 2, w, "d"), (0, 0, "d")]

        self.letter_dictionary["w"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (0, w, "d"), (0, 0, "d"),
                                       (w / 2, 0, "u"), (w / 2, 0, "d"), (w / 2, w, "d")]

        self.letter_dictionary["x"] = [(w, 0, "u"), (w, 0, "d"), (0, w, "d"),
                                       (0, 0, "u"), (0, 0, "d"), (w, w, "d")]

        self.letter_dictionary["y"] = [(w, 0, "u"), (w, 0, "d"), (0, w / 2, "d"),
                                       (0, 0, "u"), (0, 0, "d"), (0, w, "d")]

        self.letter_dictionary["z"] = [(w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (w, w, "d"), (0, w, "d")]

        self.letter_dictionary["0"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w, "d"),
                                       (w, w, "d")]

        self.letter_dictionary["1"] = [(w / 2, 0, "u"), (w / 2, 0, "d"), (w / 2, w, "d")]

        self.letter_dictionary["2"] = [(w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w / 2, "d"),
                                       (w, w / 2, "d"), (w, w, "d"), (0, w, "d")]

        self.letter_dictionary["3"] = [(w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w / 2, "d"),
                                       (w, w / 2, "d"), (w, w / 2, "u"), (0, w / 2, "u"), (0, w / 2, "d"),
                                       (0, w, "d"), (w, w, "d")]

        self.letter_dictionary["4"] = [(w, 0, "u"), (w, 0, "d"), (w, w / 2, "d"), (0, w / 2, "d"),
                                       (0, w / 2, "u"), (0, 0, "u"), (0, 0, "d"), (0, w, "d")]

        self.letter_dictionary["5"] = [(0, 0, "u"), (0, 0, "d"), (w, 0, "d"),
                                       (w, w / 2, "d"), (0, w / 2, "d"), (0, w, "d"),
                                       (w, w, "d")]

        self.letter_dictionary["6"] = [(0, 0, "u"), (0, 0, "d"), (w, 0, "d"), (w, w, "d"),
                                       (0, w, "d"), (0, w, "u"), (w, w / 2, "u"), (w, w / 2, "d"),
                                       (0, w / 2, "d"), (0, w, "d")]

        self.letter_dictionary["7"] = [(w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (w / 2, w, "d")]

        self.letter_dictionary["8"] = [(w, 0, "u"), (w, 0, "d"), (w, w, "d"),
                                       (w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w, "d"), (w, w, "d"),
                                       (w, w / 2, "u"), (w, w / 2, "d"), (0, w / 2, "d")]

        self.letter_dictionary["9"] = [(0, 0, "u"), (0, 0, "d"), (w, 0, "d"), (w, w / 2, "d"),
                                       (0, w / 2, "d"), (0, w / 2, "u"), (0, 0, "u"),
                                       (0, 0, "d"), (0, w, "d"), (w, w, "d")]

        self.letter_dictionary["."] = [(w, w, "u"), (w, w, "d")]

        self.letter_dictionary["!"] = [(w, 0, "u"), (w, 0, "d"), (w, 3 * w / 4, "d"), (w, 3 * w / 4, "u"),
                                       (w, w, "u"), (w, w, "d")]

        self.letter_dictionary["?"] = [(w, 0, "u"), (w, 0, "d"), (0, 0, "d"), (0, w / 2, "d"),
                                       (w / 2, w / 2, "d"), (w / 2, w / 2 + w / 4, "d"),
                                       (w / 2, w / 2 + w / 4, "u"), (w / 2, w, "u"), (w / 2, w, "d")]

        self.letter_dictionary["["] = [(0, 0, "u"), (0, 0, "d"), (w / 3, 0, "d"), (w / 3, w, "d"),
                                       (0, w, "d")]

        self.letter_dictionary["]"] = [(w, 0, "u"), (w, 0, "d"), (2 * w / 3, 0, "d"), (2 * w / 3, w, "d"),
                                       (w, 0, "d")]

    def write_letter(self, letter, data):
        strokes = self.letter_dictionary.get(letter, None)
        if strokes == None:
            print "letter not in dictionary"
            return

        for stroke in strokes:
            if stroke[2] == "u":
                z = data.z + 250
            elif stroke[2] == "d":
                z = data.z
            else:
                print "depth error in arm_write"
                return

            msg = "move_to:: " + str(data.x - stroke[0]) + ", " + str(data.y - stroke[1]) + ", " + str(z) + ", " + str(
                0)
            print "sending: ", msg
            self.request_cmd(msg)
            time.sleep(2)

    def check_completion(self):
        """
        Makes sure that actions run in order by waiting for response from service
        """
        time.sleep(1)
        while self.status == 0:
            pass

    def write_callback(self, data):
        self.writing_pub.publish("writing")
        time.sleep(1)

        # getting into position
        ready_motions = ["move_to:: " + str(data.x) + ", " + str(data.y) + ", " + str(data.z + 250) + ", " + str(0)]
        for motion in ready_motions:
            print "sending: ", motion
            self.request_cmd(motion)
            self.check_completion()

        self.check_completion()
        # data.shape is the string we want Edwin to write
        for letter in data.shape:
            self.write_letter(letter, data)
            data.x += self.w + 100

        self.writing_pub.publish("done")

    def run(self):
        print "running"
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            r.sleep()


if __name__ == "__main__":
    write = Writer()
    write.run()