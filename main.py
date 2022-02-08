"""
Look for all ocurances of fixme Sigal Edo, and do the changes. (rotation_coordinate_sys)
"""

import sys
import time
from datetime import datetime as dt

import matplotlib.pyplot as plt

starting_time = dt.now()

import cv2
import tensorflow as tf  # Added next 2 lines insted
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import pandas as pd
import cv2 as cv
import DAQ_BG
from self_utils import utils
from test_one_record import test_spec
from preprocessing_and_adding import preprocess_one_record
from distance import distance2

import Target_bank as TB_class
import transform
import read_write
from sty import fg, Style, RgbFg
from random import randint
from termcolor import colored
from transform import rotation_coordinate_sys
import scipy
import g_param
from g_param import get_image_num_sim, build_array
from utilitis import *
import math

from masks_for_lab import capture_update_TB, show_in_moved_window, \
    point_meter_2_pixel, image_resize, take_manual_image, sort_results, to_display

import write_to_socket
import read_from_socket
from self_utils.visualize import *
import results
from results import get_results


# np.set_printoptions(precision=3)
# pd.set_option("display.precision", 3)
# from Target_bank import print_grape
# uncomment this line and comment next for field exp
# from mask_rcnn import capture_update_TB as capture_update_TB, show_in_moved_window

########################################################################################################################
# parameters ###########################################################################################################
########################################################################################################################

fg.orange = Style(RgbFg(255, 150, 50))
fg.red = Style(RgbFg(247, 31, 0))
fg.green = Style(RgbFg(31, 177, 31))
fg.yellow = Style(RgbFg(255, 255, 70))
g_param.read_write_object = read_write.ReadWrite()
if g_param.process_type != 'load':
    rs_rob = read_from_socket.ReadFromRobot()
    ws_rob = write_to_socket.Send2Robot()
# TODO: check next 4 lines!!!!!!!!!!!
weight_file_name = r'\saved_CNN_clasifier_noise0.03_learn123_test4_3classes_77_2classes_92.1_try2_class_w0.350.350.3.h5'
# config = tf.ConfigProto() #TODO
# config.gpu_options.allow_growth = True #TODO
# sess = tf.Session(config=config) #TODO

sys.path.append("C:/Users/omerdad/Desktop/RFR/")
start_pos = np.array([-0.3, -0.24198481, 0.46430055, -0.6474185, -1.44296026, 0.59665296])  # start pos
# start_pos = np.array([-0.432, 0.086, 0.238, -1.144, -1.035, -0.128])  # start pos volcani
# start_pos = np.array([-0.173, 0.364, 0.542, -1.144, -1.035, -0.128])
step_size = g_param.step_size
# number_of_steps = math.floor(step_size / 0.1)  # amount of steps before platform move
number_of_steps = 3  # FIXME - on the exp the value was 3.
steps_counter = 0
moving_direction = "right"  # right/left
sleep_time = 5
step_direction = ["right", "up", "right", "down"]  # the order of movement # ["right", "up", "right", "down"] !!!
direction = None
g_param.init()
g_param.distances_gt = pd.read_csv(r'C:\Users\Administrator\Desktop\grapes\2d_distances.csv', header=None)
g_param.eval_mode = True
first_run = True
external_signal_all_done = False
not_finished = True
velocity = 0.7
g_param.table_of_matches_pred = pd.DataFrame(-1, index=np.arange(15), columns=np.arange(41))
g_param.table_of_matches_gt = pd.DataFrame(-1, index=np.arange(15), columns=np.arange(41))
g_param.table_of_stats = pd.DataFrame(0.0, index=['total_pred', 'total_gt', 'recall', 'precision'], columns=np.arange(41))
track_gt_pred = pd.DataFrame(-1, index=np.arange(15), columns=np.arange(41))
g_param.pred_gt_tracking = pd.DataFrame(columns=['frame', 'frame_id_gt', 'frame_id_pred',
                                                 'global_id','global_id_TB', 'IoU'])

# trans_volcani = np.array([[-0.7071, 0.7071, 0], [-0.7071, -0.7071, 0], [0, 0, 1]])


########################################################################################################################
# ---------- controlling the robot ----------
# --------------------- Omer part of the code ----------------------

def read_position():
    """
    will include creating connection between the different elements (end effectors)
    """
    cl = rs_rob.read_tcp_pos()  # cl = current_location, type: tuple
    cur_location = np.asarray(cl)
    return cur_location


def move2sonar(grape_1):
    # input("Press Enter to move to sonar") # Uncomment if field
    print(fg.yellow + "wait" + fg.rs, "\n")
    x_cam, y_cam = grape_1.x_center, grape_1.y_center
    tcp = g_param.trans.aim_sonar(x_cam, y_cam)
    new_location = g_param.trans.tcp_base(tcp)
    print(">>>>>new location", new_location)
    if g_param.process_type != "load":
        if possible_move(new_location, grape_1):
            ws_rob.move_command(False, new_location, 5, velocity)
            # check_update_move(new_location)
    print(fg.green + "continue" + fg.rs, "\n")


def two_step_movement(old_pos, new_pos):
    """
    Inverse rotation -225 to world to find x,y motion. Forward rotation of them sepratly.
    Used only one of them.
    :param old_pos:
    :param new_pos:
    :return:
    """

    r_new_pos = np.copy(new_pos)
    r_old_pos = np.copy(old_pos)
    r_new_pos[0], r_new_pos[1] = rotation_coordinate_sys(r_new_pos[0], r_new_pos[1], -g_param.base_rotation_ang)
    r_old_pos[0], r_old_pos[1] = rotation_coordinate_sys(r_old_pos[0], r_old_pos[1], -g_param.base_rotation_ang)
    r_y_move = r_new_pos[1] - r_old_pos[1]
    r_x_move = r_new_pos[0] - r_old_pos[0]
    x_move1, y_move1 = rotation_coordinate_sys(0, r_y_move, g_param.base_rotation_ang)
    x_move2, y_move2 = rotation_coordinate_sys(r_x_move, 0, g_param.base_rotation_ang)
    yz_movement = np.array([x_move1, y_move1])
    x_movement = np.array([x_move2, y_move2])
    return yz_movement, x_movement


def move2spray(grape_1, spray_dist):
    tcp = g_param.trans.aim_spray(grape_1.x_center, grape_1.y_center, spray_dist)
    # tcp = g_param.trans.aim_spray(grape_1.x_center, grape_1.y_center, grape_1.distance)
    print("grape dist: ", grape_1.distance)
    new_location = g_param.trans.tcp_base(tcp)
    print("spray_procedure location: ", new_location)
    location = read_position()
    # yz_move = np.concatenate([location[0:1], new_location[1:6]])
    # print("yz_move", yz_move)
    input("press enter to move to spray_procedure location")
    print(fg.yellow + "wait" + fg.rs, "\n")
    if g_param.process_type != "load":
        if possible_move(new_location, grape_1):
            dx, dy = two_step_movement(location, new_location)[0]
            first_move = np.concatenate([location[0:1] + dx, location[1:2] + dy, new_location[2:6]])
            ws_rob.move_command(False, first_move, 4, velocity)
            # check_update_move(first_move)
            ws_rob.move_command(False, new_location, 2, velocity)
            # check_update_move(new_location)
            return True
        else:
            return False
    print(fg.green + "continue" + fg.rs, "\n")


def move2capture():
    if g_param.process_type != "load":
        print(g_param.trans.capture_pos)
        location = read_position()
        print("old location ", location)
        x_move = np.concatenate([g_param.trans.capture_pos[0:1], location[1:3], g_param.trans.capture_pos[3:6]])
        print("x_move", x_move)
        input("Press Enter to move to capture location")
        print(fg.yellow + "wait" + fg.rs, "\n")
        # if g_param.process_type != "load": (old position of this line)
        dx, dy = two_step_movement(location, g_param.trans.capture_pos)[1]
        first_move = np.concatenate([location[0:1] + dx, location[1:2] + dy, location[2:6]])
        ws_rob.move_command(False, first_move, 4, velocity)
        # check_update_move(first_move)
        ws_rob.move_command(False, g_param.trans.capture_pos, 5, velocity)
        # check_update_move(g_param.trans.capture_pos)
    print(fg.green + "continue" + fg.rs, "\n")


# I switched between the names
def check_update_move(goal_pos):
    while True:
        move = input("Press enter if movement was ok or any key if not")
        if move != "":
            input("Try move it again")
            ws_rob.move_command(False, goal_pos, sleep_time, velocity)
        else:
            break


def possible_move(goal_pos, grape_1):
    """
    :param goal_pos: goal position for to move to.
    :param grape_1: The grape according to the parameters are checked
    :return: True if goal position is in reach of the robot, False else
    """

    # x and y in the rotate coordinate system:
    x_r, y_r = rotation_coordinate_sys(goal_pos[0], goal_pos[1], -g_param.base_rotation_ang)
    euclid_dist = np.linalg.norm(np.array([0, 0, 0]) - goal_pos[0:3])
    print("euclid_dist ", round(euclid_dist, 3))
    if abs(y_r) > g_param.y_max:
        print("Target too right, move platform")
        g_param.TB[grape_1.index].in_range = "right"
        g_param.time_to_move_platform = True
        return False
    elif goal_pos[2] > g_param.z_max:
        print("Target too high!")
        g_param.TB[grape_1.index].in_range = "high"
        return False
    elif goal_pos[2] < g_param.z_min:
        print("Target too low!")
        g_param.TB[grape_1.index].in_range = "low"
        return False
    elif euclid_dist > g_param.max_euclid_dist:
        print("The arm can not reach the goal position!!!, dist is: ", euclid_dist)
        g_param.TB[grape_1.index].in_range = "distant"
        return False
    else:
        g_param.TB[grape_1.index].in_range = "ok"
    return True


def move_const(size_of_step, direction_to_move, location):
    """
    calculate if the new position of the arm is in reach. return the position and boolean.
    grape1 = the TB object of the grape
    I assumed that
    type_of_move = move/ spray_procedure/ sonar/ take_picture
    move = advance X cm to the right (arbitrary), not receiving grape as input (not relevant)
    spray_procedure - move to spraying position
    sonar - move to sonar position
    take_picture - move to take_picture from centered position #
    :param size_of_step: step size
    :param direction_to_move:
    :param location:
    :return:
    """
    # old_location = np.copy(location)
    size_of_step = calc_step_size(size_of_step)
    # print("move const ", step_size, " start at:", location)
    if g_param.process_type != "load":
        if direction_to_move == "right":
            delta_x, delta_y = rotation_coordinate_sys(0, -size_of_step, g_param.base_rotation_ang) # fixme Sigal Edo, (0, size_of_step, -g_param.base_rotation_ang)
            location[0] = location[0] + delta_x
            location[1] = location[1] + delta_y
            ws_rob.move_command(False, location, sleep_time, velocity)
            # check_update_move(location)
        elif direction_to_move == "left": # added for tests in lab.
            delta_x, delta_y = rotation_coordinate_sys(0, size_of_step, g_param.base_rotation_ang)
            location[0] = location[0] + delta_x
            location[1] = location[1] + delta_y
            ws_rob.move_command(False, location, sleep_time, velocity)
        elif direction_to_move == "up":
            location[2] = location[2] + size_of_step
            ws_rob.move_command(False, location, sleep_time, velocity)
            # check_update_move(location)
        elif direction_to_move == "down":
            location[2] = location[2] - size_of_step
            ws_rob.move_command(False, location, sleep_time, velocity)
            # check_update_move(location)
        elif direction_to_move == "stay":
            pass
        else:
            delta_x, delta_y = rotation_coordinate_sys(0, size_of_step, g_param.base_rotation_ang)
            location[0] = location[0] + delta_x
            location[1] = location[1] + delta_y
            ws_rob.move_command(False, location, sleep_time, velocity)
            # check_update_move(location)
        # check_update_move(location) # FIXME: omer
        pos = read_position()
        if g_param.process_type == "record":
            g_param.read_write_object.write_location_to_csv(pos=pos)
    else:
        pos = g_param.read_write_object.read_location_from_csv()
    g_param.trans.set_capture_pos(pos)
    if not g_param.time_to_move_platform:
        g_param.trans.update_cam2base(pos)


def print_current_location(cur_location):
    """
    :param cur_location:
    prints current location
    """
    x_base = "x base: " + str(round(cur_location[0], 3)) + " "
    y_base = "y base: " + str(round(cur_location[1], 3)) + " "
    z_base = "z base: " + str(round(cur_location[2], 3)) + " "
    print("current location : ", x_base + y_base + z_base)


def init_arm_and_platform():
    """
    move arm before platform movement
    platform movement step size
    move arm after platform movement for first picture position
    """
    print_line_sep_time()
    if g_param.process_type != "load":
        ws_rob.move_command(False, start_pos, 4, velocity)
        # check_update_move(start_pos)
        move_platform()
        # input("Finish moving platform? ") #uncomment in field
        if external_signal_all_done is True:
            return
        g_param.time_to_move_platform = False
        if g_param.process_type == "record":
            g_param.read_write_object.write_location_to_csv(pos=read_position())
        current_location = read_position()  # 1 + 2  establish connection with the robot
        g_param.trans.set_capture_pos(current_location)
        g_param.trans.update_cam2base(current_location)  # 4
    else:
        move_platform()
        g_param.time_to_move_platform = False
        current_location = g_param.read_write_object.read_location_from_csv()
        g_param.trans.set_capture_pos(current_location)
        g_param.trans.update_cam2base(current_location)


def line_division(p1, p2, ratio):
    x = p1[0] + ratio * (p2[0] - p1[0])
    y = p1[1] + ratio * (p2[1] - p1[1])
    return np.array([x, y])


def move_and_spray(start, end, target_grape, adj_v):
    if g_param.process_type != "load":
        if possible_move(start, target_grape) and possible_move(end, target_grape):
            ws_rob.move_command(False, start, 3, adj_v)
            # check_update_move(start)
            ws_rob.spray_command(True)
            ws_rob.move_command(False, end, 2, adj_v)
            ws_rob.spray_command(False)
        else:
            pass
    return


def calc_path_points(pos, x_c, y_c, p):
    x_cam_movement = -(x_c - p[0])
    delta_x, delta_y = rotation_coordinate_sys(0, x_cam_movement, g_param.base_rotation_ang)
    pos[0] += delta_x
    pos[1] += delta_y
    pos[2] = pos[2] + (y_c - p[1])
    return pos


def calc_spray_distance(s_grape, overlap_rate):
    dist_min = g_param.last_grape_dist - g_param.min_spray_dist
    dist_max = g_param.last_grape_dist - g_param.max_spray_dist
    N = np.array([4, 3, 2])  # num of paths
    W = s_grape.w_meter
    print("W: ", W)
    all_dist = []
    for n in N:
        print("<<<<<", n, ">>>>>>")
        d_req = W / (n - (n - 1) * overlap_rate)
        print("d_req", d_req)
        dist_req = 1.6637 * d_req + 0.0423
        print("dist_req", dist_req)
        if dist_req <= g_param.min_spray_dist:
            dist_pos = g_param.min_spray_dist
        elif dist_req >= g_param.max_spray_dist:
            dist_pos = g_param.max_spray_dist
        else:
            dist_pos = dist_req
            return dist_pos, n
        all_dist.append(dist_pos)
    print("all distances: ", all_dist)
    return g_param.min_spray_dist, 2  # TODO: num of paths


def calc_spray_velocity(spray_dist):
    dist_min = g_param.last_grape_dist - g_param.min_spray_dist
    dist_max = g_param.last_grape_dist - g_param.max_spray_dist
    slope = (g_param.max_vel - g_param.min_vel) / (dist_min - dist_max)
    intercept = g_param.max_vel - slope * dist_min
    spray_v = slope * spray_dist + intercept
    while True:  # TODO: change for the president
        # selection = input(
        #     f"Select speed setting (1 or 2): \n1 - Adjusted speed: {spray_v} " f"2 - Constant speed: {g_param.const_vel}")
        selection = '2'
        if selection == "1":
            return spray_v
        elif selection == "2":
            return g_param.const_vel


# tell me which input you want. I don't think that I need any output,
# maybe only if there is a problem such as no more spraying material.
def spray_procedure(g, k, vel, num_of_paths):
    """
    :param g: The target grape for spraying
    :param vel: spraying velocity
    :param k:
    """
    # s = g.w_p * g.h_p
    center = read_position()
    flip = False
    p1, p2, p3, p4 = g.corners
    x_c, y_c = g.x_center, g.y_center
    print("target's width: ", g.w_meter)
    N = num_of_paths
    print("The number of paths is: ", N)
    for n in range(N):
        r = (n + 1) / (N + 1)
        a = 1
        # if N > 2 and (n == 0 or n == N - 1):
        #     a = 0.8
        #     # a = k * (g.pixels_count/s)
        # else:
        #     a = 1
        bottom_point = line_division(p1, p2, r)
        top_point = line_division(p4, p3, r)
        bottom_point = line_division(top_point, bottom_point, a)
        pixel_path = point_meter_2_pixel(g.distance, [bottom_point, top_point])
        display_path_of_spray(g, pixel_path, flip)
        end_p = calc_path_points(np.copy(center), x_c, y_c, bottom_point)
        start_p = calc_path_points(np.copy(center), x_c, y_c, top_point)
        # print("end_p", end_p)
        # print("start_p", start_p)
        # input("press enter for check spray procedure")
        # print(fg.yellow + "wait" + fg.rs, "\n")
        if not flip:
            move_and_spray(start_p, end_p, g, vel)
            flip = True
        else:
            move_and_spray(end_p, start_p, g, vel)
            flip = False
        print(fg.green + "continue" + fg.rs, "\n")


def spray_procedure_pixels(g):
    """
    display the path of spraying.
    :param g:
    :return: not in used for now
    """

    p1 = g.p_corners[0]
    p2 = g.p_corners[1]
    p3 = g.p_corners[2]
    p4 = g.p_corners[3]

    p1 = np.concatenate([p1, [0]], axis=0)
    p2 = np.concatenate([p2, [0]], axis=0)
    p3 = np.concatenate([p3, [0]], axis=0)
    p4 = np.concatenate([p4, [0]], axis=0)

    w = np.linalg.norm(p1 - p2)
    h = np.linalg.norm(p2 - p3)
    if w > h:
        p1, p2, p3, p4 = p1, p4, p3, p2

    bottom_points = line_division(p1, p2, 0.5)
    top_points = line_division(p3, p4, 0.5)

    # display_points_spray(g, bottom_points, top_points)


# TODO: check in work mode
def spray_process(grape_to_spray):
    if g_param.process_type != 'load':
        distance_from_target, num_of_paths = calc_spray_distance(grape_to_spray, 0.25)
        spray_distance = g_param.last_grape_dist - distance_from_target
        spray_velocity = calc_spray_velocity(spray_distance)
        print(f"spray properties: \nDistance from target: {round(distance_from_target, 3)}\n"
              f"Number of paths: {num_of_paths}"
              f" \n Speed: {round(spray_velocity, 3)}")
        aimed_spray = move2spray(grape_to_spray, spray_distance)  # 25+26
        if aimed_spray:
            spray_procedure(grape_to_spray, 1, spray_velocity, num_of_paths)
    else:
        pass


# ____________________________________
# ---------- visualizations ----------
# ____________________________________


def init_variables():
    """
    initializing Trans, read_write_object
    """
    g_param.trans = transform.Trans()
    read_write_init()


def read_write_init():
    """
    initializing all directories needed for recording/running the program.
    """
    g_param.read_write_object.create_directory()
    if g_param.process_type == "load":
        g_param.read_write_object.create_simulation_config_file()


def display_path_of_spray(grape_spray, path_points, flip):
    """
    :param grape_spray: grape to draw the spraying path on top of it
    :param path_points:  points that are the edges of the spraying route
    :return: display image if g_param.show_images is True: show the image and zoomed in image of the path
    """
    for index in range(len(path_points) - 2, -1, -1):
        start = path_points[index]
        end = path_points[index + 1]
        if flip:
            start, end = end, start
        display_points_spray(start, end)
    if g_param.show_images:
        zoomed_in = g_param.masks_image.copy()
        cornes_to_crop = grape_spray.p_corners
        result_min = list(map(min, zip(*cornes_to_crop)))
        result_max = list(map(max, zip(*cornes_to_crop)))
        min_left_corner = 0
        max_right_corner = 1024
        space_from_end = 20
        result_min = [max(result_min[0] - space_from_end, min_left_corner),
                      max(result_min[1] - space_from_end, min_left_corner)]
        result_max = [min(result_max[0] + space_from_end, max_right_corner),
                      min(result_max[1] + space_from_end, max_right_corner)]
        x = result_min[0]
        y = result_min[1]
        w = result_max[0] - result_min[0]
        h = result_max[1] - result_min[1]
        crop_img = zoomed_in[y:y + h, x:x + w]
        new_x = g_param.masks_image.shape[1]
        new_y = g_param.masks_image.shape[0]
        crop_img = cv.resize(crop_img, (new_x, new_y))
        numpy_horizontal_concat = np.concatenate((g_param.masks_image, crop_img), axis=1)
        numpy_horizontal_concat = image_resize(numpy_horizontal_concat, height=930)
        show_in_moved_window("masks, mask zoomed in", numpy_horizontal_concat, None)
        cv.waitKey()
        cv.destroyAllWindows()


def display_points_spray(start, end):
    """
    display 4 points
    :param end:
    :param start:
    :return:
    """
    end = [int(end[0]), int(end[1])]
    start = [int(start[0]), int(start[1])]
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (start[0], start[1]),
                                    radius=2, color=(0, 0, 255), thickness=2)
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (end[0], end[1]),
                                    radius=2, color=(0, 0, 255), thickness=2)
    g_param.masks_image = cv.arrowedLine(g_param.masks_image, tuple(end), tuple(start), (0, 0, 255), thickness=2)


def get_class_record(mask_id):
    if g_param.process_type == "record" or g_param.process_type == "work":
        record_class = DAQ_BG.rec()
        if g_param.process_type == "record":
            g_param.read_write_object.write_sonar_class_to_csv(record_class, mask_id)
    else:
        record_class = g_param.read_write_object.read_sonar_class_from_csv(mask_id)
    return record_class


def write_classes(mask_id, class_from_sonar, class_confirm):
    if g_param.process_type != "record":
        return
    if class_confirm == "":
        classes = [class_from_sonar, class_confirm]
    else:
        classes = [class_from_sonar, float(class_confirm)]
    g_param.read_write_object.write_sonar_classes(mask_id, classes)


def write_distances(mask_id, dist_from_sonar, dist_confirm):
    if g_param.process_type != "record":
        return
    if dist_confirm == "":
        distances = [dist_from_sonar, dist_from_sonar]
    else:
        distances = [dist_from_sonar, float(dist_confirm)]
    g_param.read_write_object.write_sonar_distances(mask_id, distances)


def get_distances(mask_id, dist_from_sonar, real_dist):
    """
    TODO (after exp): check if to load with real/measured distance
    :param mask_id:
    :param dist_from_sonar:
    :param real_dist:
    :return:
    """
    if g_param.process_type == "record":
        write_distances(mask_id, dist_from_sonar, real_dist)
    if g_param.process_type == "load":
        measured, real = g_param.read_write_object.read_sonar_distances(mask_id, real_dist)
        print(f'Real distance to grape {real} will be used. value measured by sonar is {measured}')
        return real
    return real_dist


def get_classes(mask_id, class_from_sonar, real_class):
    """
    same logic as get distances
    """
    if g_param.process_type == "record":
        write_classes(mask_id, class_from_sonar, real_class)
    if g_param.process_type == "load":
        # meas, real = g_param.read_write_object.read_sonar_class_from_csv(mask_id)
        # print(f'Real class {"grape" if is_grape else "not grape"} will be used.'
        #       f' value measured by sonar is {"grape" if is_grape else "not grape"}')
        # real = g_param.read_write_object.read_sonar_class_from_csv(mask_id, class_from_sonar, real_class)
        real = g_param.read_write_object.read_sonar_classes(mask_id, class_from_sonar)
        return real, real
    return real_class, real_class


def update_database_no_grape_sonar_human(mask_id):
    """
    update the datanase about what NN score was and what the user said (only for exp!)
    :param mask_id:
    :return:
    """
    pass


def get_dist_from_csv(index_of_tb):
    """
    will search for the matching grape cluster in the GT csv file and return the distance.
    :param mask_id: mask ID in TB, global cluster ID
    :return: distance (float), real_grape (boolean)
    ['frame', 'general_id', 'frame_id']
    Return 0.6, False if grape not in GT.
    else return estimated distance and True (REAL GRAPE)
    """
    cluster_id = g_param.pred_gt_tracking[(g_param.pred_gt_tracking['frame'] == get_image_num_sim(g_param.image_number)) &
                        (g_param.pred_gt_tracking['frame_id_pred'] == g_param.TB[index_of_tb].id_in_frame)]['global_id'].tolist()
    # ### OLD version ###
    # cluster_id = g_param.pred_df.index[(g_param.pred_df['frame'] == g_param.image_number) &
    #                       (g_param.pred_df['frame_id'] == g_param.TB[index_of_tb].id_in_frame)].tolist()
    g_param.TB[index_of_tb].GT_cluster_ID = cluster_id # update gt cluster id for the grape.
    if len(cluster_id) == 0:
        return 0.6, False
    elif cluster_id[0] == -1:
        return 0.6, False
    return g_param.distances_gt.iloc[cluster_id].values[0][0], True


def activate_sonar(mask_id, not_grape, index_of_tb):
    """
    activate sonar- read distance and grape/no grape.
    :return: distance to grape, grape (yes=1,no =0)
    The original method (which includes the try, catch) is been done using the assumption that all the grapes
    (with 100% certainty) are recorded during the experiment. Has it turned out, it's difficult to validate that,
    because the separation between the rows in the images is not always clear enough, as well as the decision
    'what counts as a grape cluster'.

    For evaluation mode (for evaluating May exp) a different schema is executed:
    For each grape detected in the current capturing position (image), if IoU between pred and GT grape cluster is above
    the threshold (0.5), than a distance will be taken from the csv file corresponding to the GT distances file.
    """
    if g_param.eval_mode:
        distance, real_g = get_dist_from_csv(index_of_tb)
        return distance, real_g # distance, real_g are true if grape is in the gt.
    try:
        counter = 0  # initiate flag
        record = get_class_record(mask_id)
        # if counter == 1 means new acquisition # adding (way of pre-process) - gets 'noise' or 'zero_cols'
        counter, x_test = preprocess_one_record(record, counter, adding='noise')
        # there are 2 weights files, each one suits to other pre-process way (adding)
        preds_3classes, no_grapes = test_spec(x_test, weight_file_name)
        # pred[0] - no grape, pred[1] - 1-5 bunches, pred[2] - 6-10 bunches
        # no grapes - True if there are no grapes (pred[0]> 0.5)
        preds_2classes = [round(preds_3classes[0][0], 3),
                          round(preds_3classes[0][1], 3) +
                          round(preds_3classes[0][2], 3)]
        # print("classes", preds_2classes)

        print(f'Probabilty this is a grape by sonar CNN : {preds_2classes[1]} %.')
        one = "\033[1m" + "1" + "\033[0m"
        zero = "\033[1m" + "0" + "\033[0m"
        # next 7 lines commented for exp
        # while True:
        #     time.sleep(0.01)
        #     target_is_grape = input( "Press " + one + " for real grape, " + zero + " if it is not a grape: ")
        #     if target_is_grape == '0' or target_is_grape == '1':
        #         break
        target_is_grape = '1'
        if target_is_grape == '0':
            update_database_no_grape_sonar_human(mask_id)
            real_class = False
        else:
            real_class = True
        # real_class = True
        # Sonar distance prediction
        # transmition_Chirp = DAQ_BG.chirp_gen(DAQ_BG.chirpAmp, DAQ_BG.chirpTime, DAQ_BG.f0, DAQ_BG.f_end,
        #                                      DAQ_BG.update_freq, DAQ_BG.trapRel)
        # D = correlation_dist(transmition_Chirp, record)
        dist_from_sonar = round(distance2(record, mask_id), 3)
        # real dist, real distance measured: to the outer edge of the sonar (10 CM from the Flach)
        print(f'Distance measured by the sonar is :{dist_from_sonar} meters.')
        # TODO: uncomment to enter it manually
        # real_dist = input(f' press enter to confirm, or enter real distance (in meters): ')
        real_dist = 0.6
        if real_dist != "":
            real_dist = float(real_dist)
        else:
            real_dist = dist_from_sonar
            # make it more efficent in 1 line ?
        # real_dist = float(real_dist) if real_dist != "" else dist_from_sonar
        distance = get_distances(mask_id, dist_from_sonar, real_dist)
        grape_class, grape_class = get_classes(mask_id, preds_2classes[1], real_class)
        # return distance, preds_2classes[1]  # make sure it converts to True/ False
        if not not_grape:
            return float(distance), 1.0
        return float(distance), grape_class  # make sure it converts to True/ False
    except Exception as e:
        while True:
            print("There was a problem with the sonar. enter distance and class manually")
            distance = 0.53
            grape_class = 1
            # TODO un comment this 2 lines when finish checking
            # distance = input("Enter distance (from camera) in meters: ")
            # grape_class = input("Enter grape class. 1 for grape, 0 otherwise. : ")
            try:
                float(distance)
            except ValueError:
                print('entered wrong type of distance value.')
                continue
            try:
                int(grape_class)  # -0.075 because we measure from the camera.
                # distance = float(distance) - 0.075 # Removed for simulation testing
                return float(distance), int(grape_class)
            except ValueError:
                print('entered wrong type of class value.')
    # return distance, grape_class #


def restart_target_bank():
    """
    restart the target bank (empty the list)
    """
    val = input("Enter 1 to restart the program, or Enter to continue: ")
    if val == 1:
        g_param.TB = []
        print("restarted")
    else:
        print("continue the program")


def move_platform():
    """
    For moving the platform manually.
    """
    print_line_sep_time()
    orig_step_size = g_param.platform_step_size
    if g_param.process_type == "record" or g_param.process_type == "work":
        if g_param.image_number == 0:
            return
        while True:
            print("Current platform step size: ", g_param.platform_step_size, '\n',
                  "insert number in Meters to change it or press Enter to continue")
            temp_step_size = input("Enter platform step size") # Uncomment for exp
            temp_step_size = ""
            try:
                if temp_step_size == "":
                    break
                elif float(temp_step_size):
                    temp_step_size = float(temp_step_size)
                    break
            except ValueError:
                print("enter float number")
        temp_step_size = float(g_param.platform_step_size)  # TODO : check press enter
        if temp_step_size == 'end':
            external_signal_all_done = True
            return
        # temp_step_size = ""
        if temp_step_size != "":
            temp_step_size = temp_step_size
            g_param.platform_step_size = temp_step_size
        else:
            temp_step_size = orig_step_size
            g_param.platform_step_size = temp_step_size
        g_param.read_write_object.write_platform_step(g_param.platform_step_size)
    else:
        g_param.platform_step_size = g_param.read_write_object.read_platform_step_size_from_csv()
    g_param.trans.update_cam2world(g_param.platform_step_size) # FIXME Sigal Edo
    g_param.sum_platform_steps += g_param.platform_step_size


def calc_step_size(step_size_to_move):
    """
    :param step_size_to_move: step size

    :return: step size to move:
    if direction is up/down:
        move g_param.height_step_size * step_size_to_move.
        for example: default is to move 0.6 * horizontal_step_size.
    else:
        move step_size_to_move to the right.
    """

    direction_of_move = g_param.direction
    if direction_of_move == "up" or direction_of_move == "down":
        step_size_to_move = step_size_to_move * g_param.height_step_size
    else:
        step_size_to_move *= g_param.steps_gap
    return step_size_to_move


def update_database_no_grape(index):
    """
    Mark that the grape is fake and remove mask (to save space in memory)
    Update that according to the sonar output, the object detected wasn't a grape.
    :param index: The index of the grape
    """
    g_param.TB[index].fake_grape = True
    # g_param.TB[index].mask = None
    g_param.TB[index].sprayed = True # TODO- consider change that, according to IoU.


def update_x_y_p(index_of_g):
    if get_image_num_sim(g_param.image_number) == g_param.TB[index_of_g].first_frame: # it was first_frame, I changed it
        corners = g_param.TB[index_of_g].p_corners
        g_param.TB[index_of_g].x_p = int([ sum(x) for x in zip(*corners) ][0] / 4)
        g_param.TB[index_of_g].y_p = int([ sum(x) for x in zip(*corners) ][1] / 4)


def update_database_sprayed(index_of_grape):
    """
    Change the grape to sprayed
    :param index_of_grape:
    """
    g_param.TB[index_of_grape].sprayed = True
    # print(g_param.TB[-6:]) # Uncomment to print Target bank
    # g_param.TB[index_of_grape].mask = None  # (to save space in memory)
    update_x_y_p(index_of_grape)
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (g_param.TB[index_of_grape].x_p, g_param.TB[index_of_grape].y_p),
                                    radius=4, color=(0, 0, 255), thickness=4)


def update_wait_another_round():
    """
    for future work - currently not in use, too many Hypotheses to check. (already working- tested)
    mark grapes that were not sprayed yet and would be closer to the center of the image in the next
    capture image position (next step).
    :return:
    """
    print("before update", g_param.TB)
    if len(g_param.TB) > 0:
        for ind in range(len(g_param.TB)):
            if check_more_than_half_away(g_param.TB[ind].x_meter, step_size / 2):  # 17
                g_param.TB[ind].wait_another_step = True
            else:
                g_param.TB[ind].wait_another_step = False
    print("after update", g_param.TB)


def count_un_sprayed():
    """
    :return: amount of grapes that hadn't been sprayed yet
    """
    amount = 0
    if len(g_param.TB) > 0:
        for ind in range(len(g_param.TB)):
            if g_param.TB[ind].sprayed is False and not g_param.TB[ind].wait_another_step:
                amount += 1
    return amount


def mark_sprayed_and_display():
    """
    mark all grapes that already been sprayed with a red dot at their center.
    mark all grapes that are too high/low with orange dot.
    mark all fake grapes blue dot.
    """
    print_line_sep_time()
    cam_0 = np.array([0, 0, 0, 1])
    cam_0_world = np.matmul(g_param.trans.t_cam2world, cam_0)
    print("TB after sorting \n", g_param.TB)

    half_image_left = g_param.half_width_meter
    for a in range(len(g_param.TB)):
        update_x_y_p(a)
        target = g_param.TB[a]
        # sprayed and still should appear in the image # FIXME Sigal Edo- CHANGE cam_0_base TO WORLD
        # and abs(target.grape_world[1] - cam_0_world[1]) < half_image_left*2
        if target.sprayed and target.in_range == "ok" and not target.fake_grape:
            g_param.masks_image = cv.circle(g_param.masks_image, (int(target.x_p), int(target.y_p)),
                                            radius=4, color=(0, 0, 255), thickness=4)
        # mark orange dot on grapes that are too high/low.
        if target.in_range == "high" or target.in_range == "low":
            g_param.masks_image = cv.circle(g_param.masks_image, (int(target.x_p), int(target.y_p)),
                                            radius=4, color=(255, 165, 0), thickness=4)
        # and abs(target.grape_world[1] - cam_0_world[1]) < half_image_left*2
        if target.sprayed and target.in_range == "ok" and target.fake_grape:
            g_param.masks_image = cv.circle(g_param.masks_image, (int(target.x_p), int(target.y_p)),
                                            radius=4, color=(255, 0, 0), thickness=4)
        g_param.masks_image = cv.putText(g_param.masks_image, str(target.index),   # visualization
                                         org=(int(target.x_p + 15), int(target.y_p) + 15),
                                         fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5,
                                         color=(255, 255, 255), thickness=1, lineType=2)
    if g_param.show_3D_plot:
        plot_tracking_map()
    if g_param.show_images:
        show_in_moved_window("Checking status", g_param.masks_image, None, 0, 0, g_param.auto_time_display)


def calc_single_axis_limits(axis: int) -> (float, float):
    """
    axis: The index (0 - x, 1 - y, 2 - z) to calc limits for.
    calc the values and range between the 2 points that are the farthest away from each other on a single axis.
    :return: min and max value of values on this axis.
    """
    min_lim_x, max_lim_x = g_param.TB[0].grape_world[axis], g_param.TB[0].grape_world[axis]
    for grape_ind in range(len(g_param.TB)):
        if g_param.TB[grape_ind].grape_world[axis] < min_lim_x:
            min_lim_x = g_param.TB[grape_ind].grape_world[axis]
        if g_param.TB[grape_ind].grape_world[axis] > max_lim_x:
            max_lim_x = g_param.TB[grape_ind].grape_world[axis]
    min_max_range = max_lim_x - min_lim_x
    # print('calc axis: ', float(min_lim_x - 0.5 * min_max_range), float(max_lim_x + 0.5 * min_max_range))
    # if axis == 0:  # for x (distance) axis, for visualization only.
    #     return float(min_lim_x - 5.5 * min_max_range), float(max_lim_x + 2.5 * min_max_range)
    return float(min_lim_x - 2.5 * min_max_range), float(max_lim_x + 2.5 * min_max_range)


def calc_axis_limits():
    """
    Calculate the limits of each axis on the 3d plot
    """
    g_param.x_lim = calc_single_axis_limits(0)
    # g_param.y_lim = calc_single_axis_limits(1)
    # g_param.z_lim = calc_single_axis_limits(2)


def get_projections():
    """
    For each grape on the map (TB), add x,y,z coordinates to a list to be displayed on the 3D plot.
    :return:
    """
    x_list, y_list, z_list = [], [], []
    for item in range(len(g_param.TB)):
        x_list.append(g_param.TB[item].grape_world[0])
        y_list.append(g_param.TB[item].grape_world[1])
        z_list.append(g_param.TB[item].grape_world[2])
    return x_list, y_list, z_list


def three_d_color(index):
    grape = g_param.TB[index]
    if grape.fake_grape:
        return 'b'
    return 'r' if grape.sprayed else 'g'


def find_item(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return -1


def  calc_y_diff(y_list):
    """
    :param y_list: list of y values of the corrdiantes
    :return: average difference between two values that represent veritcal movement of the capturing position.
    """
    diff, diff_locs = [], []
    for i in range(0, len(y_list) - 1):
        if abs(y_list[i] - y_list[i + 1]) < 50:
            diff.append(abs(y_list[i] - y_list[i + 1]))
            diff_locs.append(i)
    if len(diff) > 0:
        return int(sum(diff) / len(diff)), diff_locs
    return 0, 0


def plot_2_d_track():
    # ADD TITLE WITH IMAGE NUMBER. CHECK JUMP FROM IMAGE 5 TO 10.
    """
    2D plot of the location of the grape clusters in the image (in pixels) over time.
    Print twice the same plot every movement of the platform (Bug, but doesn't disturb anything).
    """
    if not g_param.plot_2_d_track:
        return
    for i in range(len(g_param.TB)):
        index = find_item(g_param.two_dim_track, 'index', g_param.TB[i].index)
        if index < 0:
            g_param.two_dim_track.append({'index': g_param.TB[i].index, 'values':[[g_param.TB[i].x_p, g_param.TB[i].y_p]]})
        else:
            g_param.two_dim_track[index]['values'].append([g_param.TB[i].x_p, g_param.TB[i].y_p])
    print(g_param.two_dim_track)
    if len(g_param.two_dim_track) > 0:
        list_to_print = [col for col in zip(*[d.values() for d in g_param.two_dim_track])]
        # b ={item['index']:item for item in g_param.two_dim_track}
        xs, ys=list_to_print[1], list_to_print[0]
        fig, (ax) = plt.subplots(ncols=1)
        colors = ['b','r','g','c','k','y','m']
        for i in range(min(len(ys), 4)):
            x, y = map(list, zip(*xs[i]))
            x, y = np.array(x), np.array(y)


            # y_diff, locs = calc_y_diff(y) # for better visulization- substruct y_diff from all relevent locations in y.
            # UNCOMMENT NEXT 2 LINES! ONLY FOR CHECKING NOW
            # if abs(x[-1] - x[0] > 1000 or len([t for t in x if t < - 512]) > 0): # if the grape cluster had gone out of the frame, don't track it anymore.
            #     continue
            ax.plot(x, y, marker="o", markerfacecolor=colors[i % len(colors)])
        plt.title(f'{get_image_num_sim(g_param.image_number)} -2D track')
        plt.show()


def plot_tracking_map():
    """
    Visualize all grapes centers on a 3d map.
    This function generates a plot that represents the TB in 3D.
    """
    if not to_display():
        return
    x_cors, y_cors, z_cors, colors = [], [], [], []
    for i in range(len(g_param.TB)):
        x_cor, y_cor, z_cor = g_param.TB[i].grape_world[0], g_param.TB[i].grape_world[1], g_param.TB[i].grape_world[2]
        x_cors.append(x_cor)
        y_cors.append(y_cor)
        z_cors.append(z_cor)
        # FIXME, consider changing colors to real&sprayed/real&not sprayed/ not real
        # color = 'r' if g_param.TB[i].sprayed else 'g'
        color = three_d_color(i)
        colors.append(color)
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': '3d'})

    if len(g_param.TB) < 14:
        calc_axis_limits()
    ax.set_xlim(g_param.x_lim)
    ax.set_ylim(g_param.y_lim)
    ax.set_zlim(g_param.z_lim)

    ax.zaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)

               # project each points on all planes.
    x, y, z = get_projections()
    ax.plot(x, z, '+', c='r', zdir='y', zs=g_param.y_lim[1])  # Red pluses (+) on XZ plane
    ax.plot(y, z, 's', c='g', zdir='-x', zs=g_param.x_lim[0])  # Green squares on YZ plane
    ax.plot(x, y, '*', c='b', zdir='-z', zs=g_param.z_lim[0])  # Blue pluses (*) on XY plane

    # labels titles
    ax.set_xlabel('X - Distance [m]', fontsize=14)
    ax.set_ylabel('Y - Path (left to right) [m]', fontsize=14)
    ax.set_zlabel('Z - Height [m]', fontsize=14)

    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 10
    # change color of each plane
    ax.w_yaxis.set_pane_color((1.0, 0, 0.1))  # xy plane is red
    ax.w_xaxis.set_pane_color((0, 1.0, 0, 0.1))  # xy plane is green
    ax.w_zaxis.set_pane_color((0, 0, 1.0, 0.1))  # xy plane is blue

    xx, yy = np.meshgrid([-3, 0, 3], [-3, 0, 3])
    zz = yy
    # plot plane that goes through z axis and perpendicular to xy plane. at 45 degrees (the line of x=y)
    # ax.plot_surface(xx, yy, zz, alpha=0.5)
    s = ax.scatter(x_cors, y_cors, z_cors, s=300, c=colors)  # x,y,z coordinates, size of each point, colors.
    # controls the alpha channel. all points have the same value, ignoring their distance
    s.set_edgecolors = s.set_facecolors = lambda *args: None
    ax.title.set_text(f'Imgae number {g_param.image_number}, '
                      f'real image number{get_image_num_sim(g_param.image_number)}')
    # elev = 5.0
    # azim = 135.5
    # ax.view_init(elev, azim)
    plt.show()


def check_end_program_time():
    """
    check if to end program (before movement of the platform), log statistics and exit.
    :return: True if end of program, False otherwise.
    """
    eleven = "\033[1m" + "11" + "\033[0m"
    enter = "\033[1m" + "Enter" + "\033[0m"
    # Uncomment for allow manual exit points for the program
    # temp_input = input(f"Press {eleven} to end the program. Press {enter} to continue.")
    temp_input = ""
    if temp_input == "11":
        log_statistics()
        return True
    return False


def display_points(g_to_display):
    """
    display 4 points
    :param g_to_display: grape to display
    :return:
    """
    color_index = 10
    for ii in range(len(g_param.TB)):
        color = (255 - color_index, 255 - color_index * 2, 255 - color_index * 3)
        box = np.array(g_param.TB[ii].p_corners)
        if not g_param.TB[ii].sprayed:
            cv.drawContours(g_param.masks_image, [box], 0, color)
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (int(g_to_display.p_corners[0][0]), int(g_to_display.p_corners[0][1])),
                                    radius=2, color=(0, 0, 255), thickness=2)
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (int(g_to_display.p_corners[1][0]), int(g_to_display.p_corners[1][1])),
                                    radius=2, color=(0, 0, 255), thickness=2)
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (int(g_to_display.p_corners[2][0]), int(g_to_display.p_corners[2][1])),
                                    radius=2, color=(0, 0, 255), thickness=2)
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (int(g_to_display.p_corners[3][0]), int(g_to_display.p_corners[3][1])),
                                    radius=2, color=(0, 0, 255), thickness=2)


def check_more_than_half_away(x_center, half_step_size):
    """
    :param x_center: location of middle of the grape in meter
    :param half_step_size: half step size in meters
    :return:
    true if  x_center > half_step_size, then spray_procedure only after next image (when grape will be captured when
    it is closer to the center of the point, which in high probability produce more accurate mask.
    else, return False, meaning that the grape wen't get an image when it is closer to the center.
    """
    return x_center > half_step_size


def init_program():
    """
    init the program
    """
    init_variables()
    init_arm_and_platform()
    g_param.images_in_run = g_param.read_write_object.count_images()


def remove_by_visual(grape_index_to_check):
    """
    Check visually if the grape that was detected is actually a grape or just noise.
    If it's a grape do nothing. also,
    else (if it's a noise), mark it as fake grape
    :param grape_index_to_check: the grape which is the current target
    """
    # if g_param.process_type == "load": #TODO uncomment 2 lines
    #     return False

    one = "\033[1m" + "0" + "\033[0m"
    zero = "\033[1m" + "1" + "\033[0m"
    real_grape = '1'  # next 7 lines commented for exp
    print("Real grape?")
    while True:
        time.sleep(0.01)
        # TODO: uncomment for manually decide
        # real_grape = input(colored("Yes: ", "cyan") + "press " + zero +
        #                    colored(" No: ", "red") + "Press " + one + " \n")
        real_grape = '1'
        if real_grape == '0' or real_grape == '1':
            break
    if real_grape == '0':
        update_database_no_grape(grape_index_to_check)
        return True
    return False





def update_database_visualization():
    """
    Update location of the center point of all grapes that could still appear in the image
    """
    create_centers_df()
    for j in range(len(g_param.TB)):

        TB_class.update_grape_center(j)
        grape_temp = [g_param.TB[j].x_center, g_param.TB[j].y_center]
        x_temp, y_temp = point_meter_2_pixel(d=g_param.TB[j].distance, point=[grape_temp[0], grape_temp[1]])
        # Update all grapes in database that still should appear in the image. can be reduced to -600 < x_temp < 600
        if -2000 < x_temp < 2000:
            g_param.TB[j].x_p = int(x_temp)
            g_param.TB[j].y_p = int(y_temp)
    create_centers_df() # FIXME- Check index/location in TB
        # else:
        #     break  # rest of the images will be out of range.


def print_image_details(step_dir):
    """
    prints image direction
    :param step_dir: direction of movement
    """
    im_numb = g_param.image_number
    image_det = f"Picture number {im_numb} in the simulation. " \
                f"Real image number {get_image_num_sim(g_param.image_number)}, next moving direction: {step_dir}"
    print(colored(image_det, 'green'))


def switch_materiel():
    """
    if it's the 10th' treatment for this materiel, print a reminder massage.
    """
    if len(g_param.TB) % 10 == 0 and len(g_param.TB) > 0:
        input("10 grapes were sprayed using this materiel. press enter after replacing the tank")


def display_image_with_mask(image_path, mask, alpha=0.7):
    """

    :param image_path: path of the image
    :param mask: mask (2D numpy array)
    :param alpha: opacity of the mask on top of the image
    """
    # image = cv.imread(image_path)
    # image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    image = image_path
    im = image.copy()
    non_zeros = []
    for i in range(len(mask[0][0] + 1)):
        non_zeros.append(np.count_nonzero(mask[:, :, i]))
    min_index = non_zeros.index(max(non_zeros))
    for i in range(len(mask[0][0] + 1)):
        # if i != min_index:
        rgb = (randint(0, 255), randint(0, 255), randint(0, 255))
        mask_temp = mask[:, :, i].copy()
        image[mask_temp == 1] = rgb
    plt.figure(figsize=(12, 8))
    plt.imshow(im)
    plt.imshow(image, 'gray', interpolation='none', alpha=alpha,
               vmin=1)  # alpha: [0-1], 0 is 100% transperancy, 1 is 0%
    # plt.show()
    time.sleep(5)
    print("watched image?")


def calc_gt_box_trail(image_number):
    ROOT_DIR = os.path.abspath("C:/Drive/Mask_RCNN-master")
    sys.path.append(ROOT_DIR)
    center_of_masks, bboxs = [], []
    mask_count = g_param.read_write_object.count_masks_in_image(image_number)
    # masks = np.empty([1024, 1024, 1]) # for exp which is not 13_46
    for m in range(mask_count):
        try:
            mask = g_param.read_write_object.load_mask_file(m)
            mask = np.reshape(mask, (1024, 1024, 1))
            bboxs.append(utils.extract_bboxes(mask).astype('int32'))
            center_of_masks.append(np.asarray(scipy.ndimage.measurements.center_of_mass(mask[:, :]))[:2])
            if m > 0:
                masks = np.dstack((masks, mask))
        except AssertionError as msg:
            print(msg)
    return masks, bboxs, center_of_masks


def calc_gt_box():
    ROOT_DIR = os.path.abspath("C:/Drive/Mask_RCNN-master")
    sys.path.append(ROOT_DIR)
    center_of_masks, bboxs = [], []
    mask_count = g_param.read_write_object.count_masks_in_image()
    masks = g_param.read_write_object.load_mask_file()
    bboxs = utils.extract_bboxes(masks).astype('int32')
    for m in range(mask_count):
        try:
            center_of_masks.append(np.asarray(scipy.ndimage.measurements.center_of_mass(masks[:, :, m]))[:2])
            print("bbox: ", bboxs[m], " center_of_masks: ", center_of_masks[m])
        except AssertionError as msg:
            print(msg)
    return masks, bboxs, center_of_masks


def bbox_from_corners(corners):
    corners = np.array(corners)
    x1 = min(corners[:, 0])
    x2 = max(corners[:, 0])
    y1 = min(corners[:, 1])
    y2 = max(corners[:, 1])
    # bbox = np.array([x1, y1, x2, y2]).reshape(1, 4)
    bbox = np.array([y1, x1, y2, x2]).reshape(1, 4)
    return bbox


def keep_relevant_masks(prediction_masks):
    """
    The goal of the evaluation is to check the tracking system with respect to the masks that entered the TB
    (not evaluating the performance of the network. it had been done separately (and can easily be done here is well).
    This function returns the masks on this image that weren't removed by the tracking algorithm.
    :param prediction_masks: prediction masks of the current image (before filtering by the TB).
    :return: prediction_masks and their scores.
    """
    recently_updated = len([x for x in g_param.TB if x.last_updated == get_image_num_sim(g_param.image_number)]) # g_param.image_number
    if prediction_masks.shape[2] == 0 or recently_updated == 0:
        return np.empty([1024, 1024, 0]), np.array([]), None
        # return prediction_masks, np.array([]), None
    image_masks_ind = []
    scores = []
    indexes = []
    for ind in range(len(g_param.TB)):
        if g_param.TB[ind].last_updated == get_image_num_sim(g_param.image_number): #  g_param.image_number:
            # mask = np.reshape(g_param.TB[ind].mask, (1024, 1024, 1))
            mask = g_param.TB[ind].mask
            if len(image_masks_ind) == 0:
                masks = mask
            else:
                masks = np.dstack((masks, mask))
            image_masks_ind.append([ind])
            scores.append(g_param.TB[ind].mask_score)
            indexes.append(g_param.TB[ind].index)
    scores = np.array(scores)
    return masks, scores, indexes


def update_TB_id_in_frame(prediction_masks, pred_score):
    """
    Update the id_in_frame for each of the masks that appear in the image.
    Very not elegent way to solve it! should be improved in the future!
    :param prediction_masks: masks deteced in the image and enterd the TB (not too close the edges).
    :param pred_score:prediction score per mask
    :return:
    """
    indexs_tb = []
    masks_count = 0
    if len(pred_score) == 1:
        return [g_param.TB[0].index]
    for i in range(len(g_param.TB)):
        if masks_count == prediction_masks.shape[2]:
            break
        for j in range(prediction_masks.shape[2]):
            if np.all(g_param.TB[i].mask == prediction_masks[:,:,j]):
                g_param.TB[i].id_in_frame = j
                masks_count += 1
                indexs_tb.append(g_param.TB[i].index)
    return indexs_tb


def evaluate_detections(prediction_masks, pred_score):
    if not g_param.eval_mode:
        return
    g_param.read_write_object.write_tracking_gt(create_track_gt_df())
    prediction_masks, pred_score, _ = keep_relevant_masks(prediction_masks) # 3rd elemnt (indexes) not used.
    indexs_TB = update_TB_id_in_frame(prediction_masks, pred_score)
    gt_mask, gt_box, coms = calc_gt_box()
    r_dict_gt = {'masks': gt_mask, 'bbox': gt_box, 'rois': gt_box, 'scores': np.array([1] * gt_mask.shape[2]),
                 'class_ids': np.array([1] * gt_mask.shape[2])}
    r_dict_gt = sort_results(r_dict_gt)
    gt_mask, gt_box, gt_class_id = r_dict_gt['masks'], r_dict_gt['bbox'], r_dict_gt['class_ids']
    pred_boxes = utils.extract_bboxes(prediction_masks)
    if prediction_masks.ndim == 2:
        prediction_masks = np.reshape(prediction_masks, (1024, 1024, 1))
    pred_class_id = np.array([1] * prediction_masks.shape[2])
    class_names = ['BG', 'grape_cluster']
    img_path = g_param.read_write_object.load_image_path()
    img = cv.imread(img_path)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if g_param.display_eval_images and to_display():
        if len(gt_box) > 0:
            display_differences(rgb, gt_box, gt_class_id, gt_mask,
                                pred_boxes, pred_class_id, pred_score, prediction_masks,
                                class_names, title="Pred vs. GT", ax=None,
                                show_mask=True, show_box=True,
                                iou_threshold=g_param.iou, score_threshold=0.5)
        else:
            display_instances(rgb, pred_boxes, prediction_masks, np.array([1]), class_names,
                              scores=None, title="instances",  figsize=(16, 16), ax=None,
                              show_mask=True, show_bbox=True, colors=None, captions=None)
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask, pred_boxes, pred_class_id,
        pred_score, prediction_masks, iou_threshold=g_param.iou)
    mAP, precisions, recalls, overlaps = utils.compute_ap(
        gt_box, gt_class_id, gt_mask, pred_boxes,
        pred_class_id, pred_score, prediction_masks, iou_threshold=g_param.iou)
    indexs = pred_match > -1
    # FIXME: UNCOMMENT- next 2 lines are writing the GT id_in_frame.
    #  the next uncommented sections is writing the pred id_in_frame. the frame, cluster ID are matching.
    """
    for every mask detected in the image (pred), go from left to right. if it has a matching cluster in GT
    it will have a non-negative number. if so, look for the global id of the cluster.
    for each index, produce the record: [frame num, frame_id_gt, frame_id_pred, global_id_gt, global_id_pred, iou].
    """
    for i in range(len(pred_match)):
        flag_missing_val = False
        i_val = pred_match[i]
        if pred_match[i] == -1:
            max_val = max(overlaps[i], default=0)
            try:
                i_val = list(overlaps[i]).index(max_val)
            except ValueError:
                i_val = None
                flag_missing_val = True
            if i_val is None :
                i_val = 0
        # row = [get_image_num_sim(g_param.image_number), pred_match[i], i,
        #        None, round(overlaps[i][int(pred_match[int(i_val)])],3)]
        if flag_missing_val:
            rounded_iou = 0
        else:
            rounded_iou = round(overlaps[i][int(i_val)],3)
        row = [int(get_image_num_sim(g_param.image_number)), int(pred_match[i]), int(i),
               None, indexs_TB[i], float(rounded_iou)]
        general_id = g_param.gt_track_df[(g_param.gt_track_df['frame'] == get_image_num_sim(g_param.image_number)) &
                                           (g_param.gt_track_df['frame_id_gt'] == row[1])]['general_id'].tolist()
        if len(general_id) == 0:
            row[3] = int(-1)
        else:
            row[3] = int(general_id[0])
        g_param.pred_gt_tracking.loc[len(g_param.pred_gt_tracking)] = row

    #  TODO: find a way to sync pred, GT table of tracking.

    g_param.table_of_stats.at['total_pred', get_image_num_sim(g_param.image_number)] = prediction_masks.shape[2]
    g_param.table_of_stats.at['total_gt', get_image_num_sim(g_param.image_number)] = gt_mask.shape[2]
    if gt_mask.shape[2] > 0 and prediction_masks.shape[2] > 0:
        recall_val = float(len(pred_match[indexs]) / gt_mask.shape[2])
        precision_val = float(len(pred_match[indexs]) / prediction_masks.shape[2])
    else:
        recall_val = -1
        precision_val = -1
    g_param.table_of_stats.at['recall', get_image_num_sim(g_param.image_number)] = recall_val
    g_param.table_of_stats.at['precision', get_image_num_sim(g_param.image_number)] = precision_val
    g_param.read_write_object.write_tracking_pred(g_param.pred_gt_tracking)
    g_param.read_write_object.write_tracking_pred_filterd(create_track_pred_fillterd_df())

     # g_param.read_write_object.write_tracking_pred(create_track_pred_df())
    return prediction_masks.shape[2]


def put_text_and_display(index):
    """
    This functions put thite
    :param index: index of grape that is beeing itrated (and display with TB index in it's center).
    """
    g_param.masks_image = cv.putText(g_param.masks_image, str(g_param.TB[index].index),   # visualization
                                     org=(int(g_param.TB[index].x_p), int(g_param.TB[index].y_p)),
                                     fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
                                     color=(255, 255, 255), thickness=1, lineType=2)
    display_points(g_to_display=g_param.TB[index])
    if g_param.show_images:
        show_in_moved_window("Next target to be sprayed", g_param.masks_image, index, 0,
                             0, g_param.auto_time_display)


def print_time():
    """
    Print the total time it took for this run.
    """
    ending_time = dt.now()
    elaps = ending_time - starting_time
    print("HH:MM:SS: %02d:%02d:%02d" % (elaps.seconds // 3600, elaps.seconds // 60 % 60, elaps.seconds % 60))
    if not not_finished: # time to end program
        if g_param.process_type == "load":
            g_param.read_write_object.create_simulation_config_file()



def default_fake_grapes():
    for i in range(len(g_param.TB)):
        if g_param.TB[i].fake_grape and g_param.TB[i].amount_times_updated > 0:
            g_param.TB[i].fake_grape = False
            g_param.TB[i].sprayed = False


if __name__ == '__main__':
    init_program()
    print(">>> Start position: ")
    print_current_location(g_param.trans.capture_pos)
    # The main loop:
    while not_finished:
        plot_2_d_track()
        if not first_run:
            g_param.image_number += 1
            g_param.plat_position_step_number += 1
            print_line_sep_time()
            direction = step_direction[g_param.plat_position_step_number % 4]
            if g_param.time_to_move_platform:
                init_arm_and_platform()  # 3
                first_run = True
                steps_counter, g_param.plat_position_step_number = 0, 0
                direction = "stay"
            g_param.direction = direction
            g_param.trans.set_prev_capture_pos(g_param.trans.capture_pos)
            move_const(g_param.step_size, direction, g_param.trans.capture_pos)
            if direction == "right":
                steps_counter += 1
            if direction == 'stay':
                continue
        else:
            first_run = False
        # input("Press Enter to take picture")
        print(fg.yellow + "wait" + fg.rs, "\n")
        # take_manual_image() # TODO: uncomment for exp!!!
        update_database_visualization()
        prediction_maskss, pred_scores = capture_update_TB()  # 5 + 7-12 inside #
        amount_grapes_detected = evaluate_detections(prediction_maskss, pred_scores)  # FIXME PROBLEMN IN IMAGE NUM 32
        print_image_details(step_direction[(g_param.image_number + 1) % 4])
        # g.green + "continue" + fg.rs, "\n", "TB after detecting first grape:", "\n", g_param.TB[-6:]) #print TB
        grape_ready_to_spray = TB_class.sort_by_and_check_for_grapes('leftest_first')  # 6
        # input("press enter for continue to spraying")
        # g_param.masks_image = cv.cvtColor(g_param.masks_image, cv.COLOR_RGB2BGR)
        if not first_run and g_param.image_number > 0: # fixme- not working
            mark_sprayed_and_display()
        if grape_ready_to_spray:  # 15- yes (change to sorting according to 14, rap the next lines in a function)
            # update_wait_another_round()  # for future work- details inside.
            amount_of_grapes_to_spray = count_un_sprayed()
            # if g_param.eval_mode:
            #     amount_of_grapes_to_spray = amount_grapes_detected
            for i in range(amount_of_grapes_to_spray):
                grape = g_param.TB[i]  # 16 grape is the most to the left in the list, not sprayed
                put_text_and_display(i)
                not_grape = remove_by_visual(i)
                if not_grape:  # if it's not a grape, skip next part
                    continue
                move2sonar(grape)  # 17
                if g_param.time_to_move_platform:  # 18
                    break  # 19
                # TODO: uncomment when working with sonar
                g_param.last_grape_dist, is_grape = activate_sonar(grape.index, not_grape, i)  # 20
                # g_param.last_grape_dist, is_grape = 0.55, 1  # without sonar usage
                # maskk = g_param.read_write_object.load_mask(grape.index) #TODO uncomment 2 lines to validate mask
                # display_image_with_mask(g_param.masks_image, maskk)
                # switch_materiel()  # check if it is time to switch materiel
                print("distance :", g_param.last_grape_dist, "is_grape :", is_grape)
                TB_class.update_by_real_distance(i)
                if is_grape and g_param.TB[i].in_range == "ok":  # 21 - yes
                    spray_process(grape)
                    if g_param.time_to_move_platform:  # 27
                        break  #
                    move2capture() #
                    update_database_sprayed(i)  # 28
                    mark_sprayed_and_display()# fixme- not working
                else:
                    update_database_no_grape(i)  # 22
                    mark_sprayed_and_display()# fixme- not working
            # default_fake_grapes()  #TODO- check with and without it.
        else:  # 15- no grapes to spray_procedure
            print(print_line_sep_time(), "No more targets to spray_procedure. take another picture")
            if external_signal_all_done:  # 20- yes # every 4 movements and press 11.
                not_finished = False
                print("Finished- external signal all done")
                break  # 23
        if g_param.images_in_run - 1 <= get_image_num_sim(g_param.image_number):
            if g_param.process_type != 'load':
                print(f"Finished, {g_param.images_in_run} images were taken on this batch.")
            else:
                print(f"Finished, {g_param.image_number} images were processed in this simulation.")
            not_finished = False
        if steps_counter >= number_of_steps and step_direction[(g_param.plat_position_step_number + 1) % 4] == "right":
            g_param.time_to_move_platform = True
            print(print_line_sep_time(), '\n', " move platform", '\n')
            external_signal_all_done = check_end_program_time()
            # break
            # restart_target_bank()  # option to restart without initialize
    print_time()
    get_results()
