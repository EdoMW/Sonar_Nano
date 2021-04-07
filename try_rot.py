import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import socket
import time
from ReadFromUR5 import ReadFromRobot as RFR
import math

Vs_TCP = np.array([-0.075, 0.1, 0.101])
Vc_TCP = np.array([-0.015, 0.1, 0.101])
Vm_TCP = np.array([0.11, 0.055, 0.110])
x_cam = 0
y_cam = 0

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

# in this part we read the current position of the tcp. you can change it and read other parameters by changing
# robotR.get_tcp_position command
def read_tcp_pos():
    # time.sleep(4)
    robotR = RFR()
    x, y, z, Rx, Ry, Rz = robotR.get_tcp_position()
    # time.sleep(4)
    return x, y, z, Rx, Ry, Rz


def rot_deg(deg, rotation_axis, vec):
    rotation_degrees = deg
    rotation_radians = np.radians(rotation_degrees)
    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vec)
    print(rotated_vec)
    return rotated_vec



def Rotation_matrix(k, ct, st):
    R = np.array(
        [[pow(k[0], 2) * (1 - ct) + ct, k[0] * k[1] * (1 - ct) - k[2] * st, k[0] * k[2] * (1 - ct) + k[1] * st],
         [k[0] * k[1] * (1 - ct) + k[2] * st, pow(k[1], 2) * (1 - ct) + ct, k[0] * k[2] * (1 - ct) - k[0] * st],
         [k[0] * k[2] * (1 - ct) - k[1] * st, k[0] * k[2] * (1 - ct) + k[0] * st,
          pow(k[2], 2) * (1 - ct) + ct]])  # rotation matrix
    return R

if __name__ == '__main__':
    rotation_axis = np.array([0, 0, 1])
    rot_deg(225, rotation_axis, Vc_TCP)
    xR, yR, zR, RxR, RyR, RzR = read_tcp_pos()
    print("TCP position:", xR, yR, zR, RxR, RyR, RzR)
    x_TCP = xR
    y_TCP = yR
    z_TCP = zR

    # move = ("movej(p[" + (
    #         "%f,%f,%f,%f,%f,%f" % (x_TCP, y_TCP, z_TCP, RxR, RyR, RzR)) + "], a=0.5, v=0.7,d=0)" + "\n").encode(
    #     "utf8")
    # s.send(move)
    # time.sleep(5)

    R_vector = (RxR, RyR, RzR)
    print("R_vector: ", R_vector)
    t = np.linalg.norm(R_vector)
    k = R_vector / t
    print(k)
    ct = math.cos(t)
    st = math.sin(t)
    R = Rotation_matrix(k, ct, st)
    R_t = np.transpose(R)
    print("Rotation matrix:\n", R)
    V = np.array([x_cam, y_cam, 0])
    V = np.append(V, [1], axis=0)
    T_TCP_B = np.append(R, [[0, 0, 0]], axis=0)
    T_TCP_B = np.append(T_TCP_B, [[x_TCP], [y_TCP], [z_TCP], [1]], axis=1)

    t2 = 225
    k2 = [0,0,1]
    ct = math.cos(t2)
    st = math.sin(t2)
    R = Rotation_matrix(k2, ct, st)
    # T_TCP_C =
