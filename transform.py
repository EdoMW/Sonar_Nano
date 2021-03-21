
import math
import numpy as np
import g_param
import time


def ang_vec2rot(rv):
    t = np.linalg.norm(rv)
    k = rv / t
    ct = math.cos(t)
    st = math.sin(t)
    rot = np.array(
        [[pow(k[0], 2) * (1 - ct) + ct, k[0] * k[1] * (1 - ct) - k[2] * st, k[0] * k[2] * (1 - ct) + k[1] * st],
        [k[0] * k[1] * (1 - ct) + k[2] * st, pow(k[1], 2) * (1 - ct) + ct, k[1] * k[2] * (1 - ct) - k[0] * st],
        [k[0] * k[2] * (1 - ct) - k[1] * st, k[1] * k[2] * (1 - ct) + k[0] * st,
        pow(k[2], 2) * (1 - ct) + ct]])  # rotation matrix
    return rot


class Trans:
    def __init__(self):
        self.t_cam2tcp=T_C_TCP = np.array([[0.7071, -0.7071, 0, 0.08216], [0.7071, 0.7071, 0, -0.060677], [0, 0, 1, 0], [0, 0, 0, 1]]) # FIXME: Old values
        # T_TCP_C = np.array([[0.7071, -0.7071, 0, 0.065], [0.7071, 0.7071, 0, -0.075], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.v_sonar_tcp = np.array([0.152735, -0.00282843, 0])
        self.v_spray_tcp = np.array([-0.0311127, -0.10323759, 0.12])
        #Vm_TCP = np.array([-0.045, -0.1, 0.12])
        #Vs_TCP = np.array([0.11, 0.07, 0])
        self.t_base2world = np.identity(4)
        self.t_cam2base = np.identity(4)
        self.t_cam2world = np.identity(4)
        self.t_tcp2base = np.identity(4)
        self.ang_vec_tcp = np.array([])
        self.capture_pos = np.array([])
    # Turns axis angles  into rotation matrix

    def set_capture_pos(self, location):
        self.capture_pos = location

    def update_base2world(self, mobile_location):
        pass

    def update_cam2base(self, cur_location):
        self.ang_vec_tcp = cur_location[3:]
        rot = ang_vec2rot(self.ang_vec_tcp)
        self.t_tcp2base = np.append(rot, [[0, 0, 0]], axis=0)
        self.t_tcp2base = np.append(self.t_tcp2base, [[cur_location[0]], [cur_location[1]], [cur_location[2]], [1]], axis=1)
        self.t_cam2base = np.matmul(self.t_tcp2base, self.t_cam2tcp)
        self.t_cam2world = np.matmul(self.t_base2world, self.t_cam2base)
        # if g_param.process_type == "record":
        #     g_param.read_write_object.write_transformations_to_csv()
        # elif g_param.process_type == "load":
        #     g_param.read_write_object.read_transformations_from_csv()
        time.sleep(0.01)

    # TODO- make it world and not just base
    def grape_world(self, x_cam, y_cam):
        grape_cam = np.array([float(x_cam), float(y_cam), 0, 1])
        grape_world = np.matmul(self.t_cam2world, grape_cam)
        return grape_world[:-1]

    def tcp_base(self, tcp):
        tcp = np.append(tcp, 1)
        tcp_base = np.matmul(self.t_tcp2base, tcp)
        tcp_base = np.concatenate((tcp_base[:-1], self.ang_vec_tcp), axis=0)
        return tcp_base

    def aim_sonar(self, x_cam, y_cam):
        grape_cam = np.array([float(x_cam), float(y_cam), 0, 1])
        grape_tcp = np.matmul(self.t_cam2tcp, grape_cam)
        grape_tcp = grape_tcp[:-1]
        new_tcp = grape_tcp - self.v_sonar_tcp
        return new_tcp

    def aim_spray(self,  x_cam, y_cam, distance):
        dist = distance - g_param.safety_dist
        grape_cam = np.array([float(x_cam), float(y_cam), dist, 1])
        grape_tcp = np.matmul(self.t_cam2tcp, grape_cam)
        grape_tcp = grape_tcp[:-1]
        new_tcp = grape_tcp - self.v_spray_tcp
        return new_tcp
