import g_param
import numpy as np
import time
import transform
import read_write
import write_to_socket
import read_from_socket
import math

g_param.read_write_object = read_write.ReadWrite()
rs_rob = read_from_socket.ReadFromRobot()  # FIXME: make it possible to run on "load" when robot is turned off
ws_rob = write_to_socket.Send2Robot()
start_pos = np.array([-0.21790651, -0.44937682, 0.50891671, 0.09135264, -1.75525897, 1.30217356])  # angle = 135
# start_pos = np.array([-0.252, -0.24198481, 0.43430055, -0.6474185, -1.44296026, 0.59665296])  # angle = 180

def read_position():
    cl = rs_rob.read_tcp_pos()  # cl = current_location, type: tuple
    cur_location = np.asarray(cl)
    return cur_location



def volcani_trans(x,y, angle):
    radians_ang = np.radians(angle % 360)
    x_tag = x * math.cos(radians_ang) + y * math.sin(radians_ang)
    y_tag = -x * math.sin(radians_ang) + y * math.cos(radians_ang)
    return x_tag, y_tag


ws_rob.move_command(True, start_pos, 5, 0.4)
print(read_position())
delta_x, delta_y = volcani_trans(0.2,0,135)
new_pos = read_position()
new_pos[0] = new_pos[0]+delta_x
new_pos[1] = new_pos[1]+delta_y


print("10 cm movement to the right: \n", new_pos)
input("Press Enter to confirm location>>>")
ws_rob.move_command(True, new_pos, 5, 0.1)

