import socket
import time
from ReadFromUR5 import ReadFromRobot as RFR

# defining variables and the UR5 parameters, you can add other variables
HOST = "132.72.96.97"  # The remote host
PORT = 30002  # The same port as used by the servers = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
time.sleep(0.5)
s.send(("set_gravity([0.0, 0.0, 9.82])" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_tool_voltage(0)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_safety_mode_transition_hardness(1)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_tcp(p[0.0,0.0,0.0,0.0,0.0,0.0])" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_payload(0.5)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_standard_analog_input_domain(0, 1)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_standard_analog_input_domain(1, 1)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_tool_analog_input_domain(0, 1)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_tool_analog_input_domain(1, 1)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_analog_outputdomain(0, 0)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_analog_outputdomain(1, 0)" + "\n").encode('utf8'))
time.sleep(0.1)
s.send(("set_input_actions_to_default()" + "\n").encode('utf8'))
time.sleep(0.1)
s.settimeout(10)
time.sleep(0.2)
counter = 0


def read_tcp_pos():
    # time.sleep(4)
    robotR = RFR()
    x, y, z, Rx, Ry, Rz = robotR.get_tcp_position()
    # time.sleep(4)
    return x, y, z, Rx, Ry, Rz


def read_vel():
    robotR = RFR()
    xV, yV, zV, RxV, RyV, RzV = robotR.get_tcp_velocities()
    return xV, yV, zV, RxV, RyV, RzV


def move_command(isLinear, pos, tSleep):
    time.sleep(0.05)
    if isLinear:
        s.send(("movel(p[" + (
                "%f,%f,%f,%f,%f,%f" % (
            pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])) + "], a=0.5, v=0.7,r=0)" + "\n").encode(
            "utf8"))
    else:
        s.send(("movej(p[" + (
                "%f,%f,%f,%f,%f,%f" % (
            pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])) + "], a=0.5, v=0.7,r=0)" + "\n").encode(
            "utf8"))
    time.sleep(tSleep)


def spray_command(spray):
    if spray:
        s.send(("set_digital_out(0, True)" + "\n").encode("utf8"))
    else:
        s.send(("set_digital_out(0, False)" + "\n").encode("utf8"))
