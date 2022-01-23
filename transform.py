import math
import numpy as np
import g_param
import time
from math import cos, sin
np.set_printoptions(precision=3)


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


def rotation_coordinate_sys(x, y, angle):
    """
    225 degrees rotation about z axis.
    :param x:
    :param y:
    :param angle:
    :return:
    """
    radians_ang = np.radians(angle % 360)
    x_tag = x * math.cos(radians_ang) + y * math.sin(radians_ang)
    y_tag = -x * math.sin(radians_ang) + y * math.cos(radians_ang)
    return x_tag, y_tag


class Trans:
    def __init__(self):
        self.t_cam2tcp = T_C_TCP = np.array(
            [[0.7071, -0.7071, 0, 0.08216], [0.7071, 0.7071, 0, -0.060677], [0, 0, 1, 0],
             [0, 0, 0, 1]])  # FIXME: Old values
        # T_TCP_C = np.array([[0.7071, -0.7071, 0, 0.065], [0.7071, 0.7071, 0, -0.075], [0, 0, 1, 0], [0, 0, 0, 1]])
        # self.v_sonar_tcp = np.array([0.152735, -0.00282843, 0])
        self.v_sonar_tcp = np.array([0.04, 0.075, 0])
        self.v_spray_tcp = np.array([-0.0311127, -0.10323759, 0.12])
        # Vm_TCP = np.array([-0.045, -0.1, 0.12])
        # Vs_TCP = np.array([0.11, 0.07, 0])
        #self.t_base2world = np.identity(4)
        # t_base2world: rotation around 225.
        ang = 45/180*math.pi
        self.t_base2world = np.array(   # FIXME- Sigal Edo Fix to rotate! maybe mul by -1
            [[cos(ang), -sin(ang), 0, 0], [sin(ang), cos(ang), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # [[-0.1736, -0.9848, 0, 0], [0.9848, -0.1736, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # [[-0.7071, -0.7071, 0, 0], [0.7071, -0.7071, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.t_cam2base = np.identity(4)
        self.t_cam2world = self.t_base2world
        self.t_tcp2base = np.identity(4)
        self.ang_vec_tcp = np.array([])
        self.capture_pos = np.array([])
        self.prev_capture_pos = np.array([])
    # Turns axis angles  into rotation matrix

    def set_prev_capture_pos(self, location):
        self.prev_capture_pos = location.copy()

    def set_capture_pos(self, location):
        self.capture_pos = location

    def update_cam2world(self, step_size):
        self.t_base2world[1][3] += step_size
        self.t_cam2world = np.matmul(self.t_base2world, self.t_cam2base)
        # FIXME! Sigal Edo

    def update_cam2base(self, cur_location):
        """
        Updates both cam2base, cam2world
        """
        self.ang_vec_tcp = cur_location[3:]
        rot = ang_vec2rot(self.ang_vec_tcp)
        self.t_tcp2base = np.append(rot, [[0, 0, 0]], axis=0)
        self.t_tcp2base = np.append(self.t_tcp2base, [[cur_location[0]], [cur_location[1]], [cur_location[2]], [1]],
                                    axis=1)
        self.t_cam2base = np.matmul(self.t_tcp2base, self.t_cam2tcp)
        self.t_cam2world = np.matmul(self.t_base2world, self.t_cam2base)
        if g_param.process_type == "record":
            g_param.read_write_object.write_transformations_to_csv()
        # elif g_param.process_type == "load":
        #     g_param.read_write_object.read_transformations_from_csv()
        time.sleep(0.01)

    # TODO- Omer - change the cam2word matrix due to platform steps
    def grape_world(self, x_cam, y_cam, d):  # FIXME Sigal Edo, add distance to signature. -V-
        grape_cam = np.array([float(x_cam), float(y_cam), float(d), 1])  # d- check for *-1
        grape_world = np.matmul(self.t_cam2world, grape_cam)
        # delta_x, delta_y = rotation_coordinate_sys(0, -g_param.sum_platform_steps, g_param.base_rotation_ang)
        # grape_world[0] = grape_world[0] + delta_x
        # grape_world[1] = grape_world[1] + delta_y
        return grape_world[:-1]

    def grape_base(self, x_cam, y_cam):
        grape_cam = np.array([float(x_cam), float(y_cam), 0, 1])
        grape_base = np.matmul(self.t_cam2base, grape_cam)
        return grape_base[:-1]

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

    def aim_spray(self, x_cam, y_cam, distance):
        dist = distance - g_param.safety_dist
        grape_cam = np.array([float(x_cam), float(y_cam), dist, 1])
        grape_tcp = np.matmul(self.t_cam2tcp, grape_cam)
        grape_tcp = grape_tcp[:-1]
        new_tcp = grape_tcp - self.v_spray_tcp
        return new_tcp
