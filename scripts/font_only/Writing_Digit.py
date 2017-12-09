import rospy
from cell.py import Cell


def get

def solve_sudoku_server():
    rospy.init_node('solve_sudoku_server')
    s = rospy.Service('solve_sudoku', AddTwoInts,
    print "Ready to give solution."
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()

