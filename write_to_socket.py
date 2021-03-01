# Echo client program
import socket
import time
import csv
import os
import g_param
from datetime import datetime
import struct
import codecs
import read_from_socket as rfs



def get_local_time():
    # Hours: minutes
    t = datetime.now().strftime("%H_%M_%S_%f")[:-4]
    # t = time.localtime()
    return t


class Send2Robot:
    def __init__(self):
        self.HOST = "132.72.96.97"  # The remote host
        self.PORT_30002 = 30002  # The same port as used by the servers = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_to_robot()
        self.rob_init()

    def __del__(self):
        print(' send socket closed')
        self.disconnect_to_robot()

    def connect_to_robot(self):
        try:
            print("connecting to UR5.. \nopen write to socket", self.s)
            self.s.settimeout(10)
            self.s.connect((self.HOST, self.PORT_30002))
            time.sleep(0.1)
        except socket.error as socketerror:
            print("Error: ", socketerror)

    def disconnect_to_robot(self):
        try:
            print("disconnecting to UR5.. \nclose write to socket", self.s)
            self.s.close()
            time.sleep(0.1)
        except socket.error as socketerror:
            print("Error: ", socketerror)

    def rob_init(self):
        self.s.send(("set_gravity([0.0, 0.0, 9.82])" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_tool_voltage(0)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_safety_mode_transition_hardness(1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_tcp(p[0.0,0.0,0.0,0.0,0.0,0.0])" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_payload(0.5)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_standard_analog_input_domain(0, 1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_standard_analog_input_domain(1, 1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_tool_analog_input_domain(0, 1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_tool_analog_input_domain(1, 1)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_analog_outputdomain(0, 0)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_analog_outputdomain(1, 0)" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.send(("set_input_actions_to_default()" + "\n").encode('utf8'))
        time.sleep(0.1)
        self.s.settimeout(10)
        time.sleep(0.2)

    def move_command(self, isLinear, pos, tSleep):
        time.sleep(0.05)
        if isLinear:
            self.s.send(("movel(p[" + (
                    "%f,%f,%f,%f,%f,%f" % (
                pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])) + "], a=0.5, v=0.7,r=0)" + "\n").encode(
                "utf8"))
        else:
            self.s.send(("movej(p[" + (
                    "%f,%f,%f,%f,%f,%f" % (
                pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])) + "], a=0.5, v=0.7,r=0)" + "\n").encode(
                "utf8"))
        time.sleep(tSleep)
        g_param.read_write_object.write_location_to_csv(pos=pos)

    def spray_command(self, spray):
        if spray:
            self.s.send(("set_digital_out(0, True)" + "\n").encode("utf8"))
        else:
            self.s.send(("set_digital_out(0, False)" + "\n").encode("utf8"))
