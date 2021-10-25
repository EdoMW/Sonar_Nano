import sys
import tensorflow as tf # Added next 2 lines insted
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


import g_param
import numpy as np
import cv2 as cv
import DAQ_BG
from test_one_record import test_spec
from preprocessing_and_adding import preprocess_one_record
from distance import distance2
import time
import Target_bank as TB_class
import transform
import read_write
from sty import fg, Style, RgbFg
from random import randint
from termcolor import colored
from transform import rotation_coordinate_sys
import matplotlib.pyplot as plt

from masks_for_lab import take_picture_and_run as capture_update_TB, show_in_moved_window, \
    point_meter_2_pixel, image_resize, take_manual_image
import write_to_socket
import read_from_socket
from self_utils.visualize import *

np.set_printoptions(precision=3)

# from Target_bank import print_grape
# uncomment this line and comment next for field exp
# from mask_rcnn import take_picture_and_run as capture_update_TB, show_in_moved_window


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
number_of_steps = 3
steps_counter = 0
moving_direction = "right"  # right/left
sleep_time = 5
step_direction = ["right", "up", "right", "down"]  # the order of movement
direction = None
g_param.init()
first_run = True
g_param.show_images = True  # g.show_images: if true, it is visualizes the process
external_signal_all_done = False
not_finished = True
velocity = 0.7


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
    input("Press Enter to move to sonar")
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
            delta_x, delta_y = rotation_coordinate_sys(0, -size_of_step, g_param.base_rotation_ang)
            location[0] = location[0] + delta_x
            location[1] = location[1] + delta_y
            ws_rob.move_command(False, location, sleep_time, velocity)
            # check_update_move(location)
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
        input("Finish moving platform? ")
        if external_signal_all_done is True:
            return
        g_param.time_to_move_platform = False
        if g_param.process_type == "record":
            g_param.read_write_object.write_location_to_csv(pos=read_position())
        current_location = read_position()  # 1 + 2  establish connection with the robot
        g_param.trans.set_capture_pos(current_location)
        g_param.trans.update_cam2base(current_location)  # 4
    else:
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
            # check_update_move(start)  # TODO:check with Sigal
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
        print(f"spray properties: \nDistance from target: {round(distance_from_target,3)}\n"
              f"Number of paths: {num_of_paths}"
              f" \n Speed: {round(spray_velocity,3)}")
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
        measured, real = g_param.read_write_object.read_sonar_distances(mask_id , real_dist)
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


# calling the 2 sonar methods to get distance and validate existence of grapes


def update_database_no_grape_sonar_human(mask_id):
    """
    update the datanase about what NN score was and what the user said (only for exp!)
    :param mask_id:
    :return:
    """
    pass


def activate_sonar(mask_id, not_grape):
    """
    activate sonar- read distance and grape/no grape.
    :return: distance to grape, grape (yes=1,no =0)
    """
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
        transmition_Chirp = DAQ_BG.chirp_gen(DAQ_BG.chirpAmp, DAQ_BG.chirpTime, DAQ_BG.f0, DAQ_BG.f_end,
                                             DAQ_BG.update_freq, DAQ_BG.trapRel)
        # D = correlation_dist(transmition_Chirp, record)
        dist_from_sonar = round(distance2(record, mask_id), 3)
        # real dist, real distance measured: to the outer edge of the sonar (10 CM from the Flach)
        print(f'Distance measured by the sonar is :{dist_from_sonar} meters.')
        real_dist = input(f' press enter to confirm, or enter real distance (in meters): ')
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
            distance = 0.6
            grape_class = 1
            #TODO un comment this 2 lines when finish checking
            # distance = input("Enter distance (from camera) in meters: ")
            # grape_class = input("Enter grape class. 1 for grape, 0 otherwise. : ")
            try:
                float(distance)
            except ValueError:
                print('entered wrong type of distance value.')
                continue
            try:
                int(grape_class)  # -0.075 because we measure from the camera.
                distance = float(distance) - 0.075
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


def print_line_sep_time():
    """
    prints a line separation of ___ and the current time
    """
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print('-' * 40, current_time, '-' * 40, '\n')


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
            temp_step_size = input("Enter platform step size")
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
    return step_size_to_move


def update_database_no_grape(index):
    """
    Mark that the grape is fake and remove mask (to save space in memory)
    Update that according to the sonar output, the object detected wasn't a grape.
    :param index: The index of the grape
    """
    g_param.TB[index].fake_grape = True
    g_param.TB[index].mask = None
    g_param.TB[index].sprayed = True
    # check with Sigal if I want to count amount of grapes sprayed
    # any way, I could just subtract the amount of grapes that are sprayed and not fake_grape


def update_database_sprayed(index_of_grape):
    """
    Change the grape to sprayed
    :param index_of_grape:
    """
    g_param.TB[index_of_grape].sprayed = True
    print(g_param.TB[-6:])
    g_param.TB[index_of_grape].mask = None  # (to save space in memory)
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (g_param.TB[index_of_grape].x_p, g_param.TB[index_of_grape].y_p),
                                    radius=4, color=(0, 0, 255), thickness=4)


def update_wait_another_round():
    """
    for future work (already working- tested)
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
    print(amount)
    return amount


def mark_sprayed_and_display():
    """
    mark all grapes that already been sprayed with a red dot at their center.
    mark all grapes that are too high/low with orange dot.
    mark all fake grapes blue dot.
    """
    print_line_sep_time()
    cam_0 = np.array([0, 0, 0, 1])
    cam_0_base = np.matmul(g_param.trans.t_cam2base, cam_0)
    print("TB after sorting \n", g_param.TB)
    half_image_left = g_param.half_width_meter
    for a in range(len(g_param.TB)):
        target = g_param.TB[a]
        # sprayed and steel should appear in the image #
        if target.sprayed and abs(target.grape_world[1] - cam_0_base[1]) < half_image_left \
                and target.in_range == "ok" and not target.fake_grape:
            g_param.masks_image = cv.circle(g_param.masks_image, (int(target.x_p), int(target.y_p)),
                                            radius=4, color=(0, 0, 255), thickness=4)
        # mark orange dot on grapes that are too high/low.
        if target.in_range == "high" or target.in_range == "low":
            g_param.masks_image = cv.circle(g_param.masks_image, (int(target.x_p), int(target.y_p)),
                                            radius=4, color=(255, 165, 0), thickness=4)
        if target.sprayed and abs(target.grape_world[1] - cam_0_base[1]) < half_image_left \
                and target.in_range == "ok" and target.fake_grape:
            g_param.masks_image = cv.circle(g_param.masks_image, (int(target.x_p), int(target.y_p)),
                                            radius=4, color=(255, 0, 0), thickness=4)
    if g_param.show_images:
        show_in_moved_window("Checking status", g_param.masks_image, None)
        cv.waitKey()
        cv.destroyAllWindows()


def check_end_program_time():
    eleven = "\033[1m" + "11" + "\033[0m"
    enter = "\033[1m" + "Enter" + "\033[0m"
    temp_input = input(f"Press {eleven} to end the program. Press {enter} to continue.")
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
        real_grape = input(colored("Yes: ", "cyan") + "press " + zero +
                           colored(" No: ", "red") + "Press " + one + " \n")
        if real_grape == '0' or real_grape == '1':
            break
    if real_grape == '0':
        update_database_no_grape(grape_index_to_check)
        return True
    return False


def log_statistics():
    """
    write down some descriptive statics of what just recorded
    :return:
    """
    # take_manual_image()
    amount_of_grapes = len(g_param.TB)
    amount_of_fake = sum(g.fake_grape is True for g in g_param.TB)
    amount_sprayed = sum((g.sprayed is False and g.fake_grape is False) for g in g_param.TB)
    print('-' * 50, '\n', '-' * 50, '\n', '-' * 25 + ' summary ' + '-' * 25)
    print(f'Total grapes detected: {amount_of_grapes}')
    print(f'Total "false" grapes detected: {amount_of_fake}')
    print(f'Precision : {round(((amount_of_grapes - amount_of_fake) / amount_of_grapes + 0.0000001), 3)}')
    print(f'percentage of grapes that were not reachable: {round((amount_sprayed / amount_of_grapes + 0.0000001), 3)}'
          f' \n')
    print('-' * 50, '\n', '-' * 50)
    pass


def update_database_visualization():
    """
    Update location of the center point of all grapes that could still appear in the image
    """
    for j in range(len(g_param.TB)):
        TB_class.update_grape_center(j)
        grape_temp = [g_param.TB[j].x_center, g_param.TB[j].y_center]
        x_temp, y_temp = point_meter_2_pixel(d=g_param.TB[j].distance, point=[grape_temp[0], grape_temp[1]])
        if -2000 < x_temp < 2000:  # Update all grapes in database that still should appear in the image.
            g_param.TB[j].x_p = int(x_temp)
            g_param.TB[j].y_p = int(y_temp)
        else:
            break  # check if it's necessary


def print_image_details(step_dir):
    """
    prints image direction
    :param step_dir: direction of movement
    """
    im_numb = g_param.image_number
    image_det = f"Picture number {im_numb}, next moving direction: {step_dir}"
    print(colored(image_det, 'green'))


def switch_materiel():
    """
    if it's the 10th' treatment for this materiel, print a reminder massage.
    """
    if len(g_param.TB) % 10 == 0 and len(g_param.TB) > 0:
        input("10 grapes were sprayed using this materiel. press enter after replacing the tank")


def display_image_with_mask(image_path, mask, alpha = 0.7):
    # image = cv.imread(image_path)
    # image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    image = image_path
    im = image.copy()
    print(image.shape)
    print(mask.shape)
    non_zeros = []
    for i in range(len(mask[0][0] + 1)):
        non_zeros.append(np.count_nonzero(mask[:,:,i]))
    min_index = non_zeros.index(max(non_zeros))
    for i in range(len(mask[0][0] + 1)):
        # if i != min_index:
        rgb = (randint(0,255), randint(0,255), randint(0,255))
        mask_temp = mask[:,:,i].copy()
        image[mask_temp==1] = rgb
    plt.figure(figsize=(12,8))
    plt.imshow(im)
    plt.imshow(image, 'gray', interpolation='none', alpha=alpha, vmin = 1) # alpha: [0-1], 0 is 100% transperancy, 1 is 0%
    plt.show()
    time.sleep(5)
    print("watched image?")


def calc_gt_box(image_number):
    mask_count = g_param.read_write_object.count_masks_in_image(image_number)
    print(mask_count)
    # for i in range(mask_count):
    #     masks = g_param.read_write_object.load_mask()


if __name__ == '__main__':
    init_program()
    print(">>> Start position: ")
    print_current_location(g_param.trans.capture_pos)
    # The main loop:
    while not_finished:
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
            # move_const(g_param.step_sizegit , "right", current_location)  # try to move 1 step size
            # if g_param.time_to_move_platform:
            #     print('issue moving const ')
            #     print_current_location(g_param.trans.capture_pos)
            #     break

            if direction == "right":
                steps_counter += 1
        else:
            first_run = False
        # input("Press Enter to take picture")
        print(fg.yellow + "wait" + fg.rs, "\n")
        # take_manual_image() # TODO: uncomment for exp!!!
        capture_update_TB()  # 5 + 7-12 inside #
        update_database_visualization()  # FIXME
        print_image_details(step_direction[(g_param.image_number + 1) % 4])
        print(fg.green + "continue" + fg.rs, "\n", "TB after detecting first grape:", "\n", g_param.TB[-6:])
        grape_ready_to_spray = TB_class.sort_by_and_check_for_grapes('leftest_first')  # 6
        # input("press enter for continue to spraying")
        # g_param.masks_image = cv.cvtColor(g_param.masks_image, cv.COLOR_RGB2BGR)
        if not first_run and g_param.image_number > 0:
            mark_sprayed_and_display()
        if grape_ready_to_spray:  # 15- yes (change to sorting according to 14, rap the next lines in a function)
            # update_wait_another_round()  # for future work- details inside.
            amount_of_grapes_to_spray = count_un_sprayed()
            for i in range(amount_of_grapes_to_spray):
                # TODO- add evaluation mode which mark unlabeled masks as false detection,
                #  and assign masks accordignly to the GT.
                eval_mode = True
                if eval_mode:
                    gt_box = calc_gt_box(g_param.image_number)
                    # display_differences(g_param.masks_image, gt_box, gt_class_id, gt_mask,
                    #                     pred_box, pred_class_id, pred_score, pred_mask,
                    #                     class_names, title="", ax=None,
                    #                     show_mask=True, show_box=True,
                    #                     iou_threshold=0.5, score_threshold=0.5)
                    # compute_overlaps_masks(gt_mask, pred_masks)
                # visualization
                g_param.masks_image = cv.putText(g_param.masks_image, str(g_param.TB[i].index),
                                                 org=(int(g_param.TB[i].x_p), int(g_param.TB[i].y_p)),
                                                 fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
                                                 color=(255, 255, 255), thickness=1, lineType=2)
                display_points(g_to_display=g_param.TB[i])
                if g_param.show_images:
                    show_in_moved_window("Next target to be sprayed", g_param.masks_image, i)
                    cv.waitKey()
                    cv.destroyAllWindows()
                grape = g_param.TB[i]  # 16 grape is the most to the left in the list, not sprayed
                not_grape = remove_by_visual(i)  # FIXME- add option- if manually confirmed, it's a grape
                if not_grape:  # if it's not a grape, skip next part
                    continue
                move2sonar(grape)  # 17
                if g_param.time_to_move_platform:  # 18
                    break  # 19
                # TODO: uncomment when working with sonar
                g_param.last_grape_dist, is_grape = activate_sonar(grape.index, not_grape)  # 20
                # g_param.last_grape_dist, is_grape = 0.55, 1  # without sonar usage
                # maskk = g_param.read_write_object.load_mask(grape.index) #TODO uncomment 2 lines to validate mask
                # display_image_with_mask(g_param.masks_image, maskk)
                # switch_materiel()  # check if it is time to switch materiel

                print("distance :", g_param.last_grape_dist, "is_grape :", is_grape)
                if is_grape and g_param.TB[i].in_range == "ok":  # 21 - yes
                    TB_class.update_by_real_distance(g_param.TB[i], i)  # 23,24 TODO: check
                    # spray_process(grape)
                    if g_param.time_to_move_platform:  # 27
                        break  #
                    move2capture()
                    update_database_sprayed(i)  # 28
                    mark_sprayed_and_display()
                else:
                    update_database_no_grape(i)  # 22
        else:  # 15- no grapes to spray_procedure
            print(print_line_sep_time(), "No more targets to spray_procedure. take another picture")
            if external_signal_all_done:  # 20- yes # every 4 movements and press 11.
                not_finished = False
                print("Finished- external signal all done")
                break  # 23
        if steps_counter >= number_of_steps and step_direction[(g_param.plat_position_step_number + 1) % 4] == "right":
            g_param.time_to_move_platform = True
            print(print_line_sep_time(), '\n', " move platform", '\n')
            external_signal_all_done = check_end_program_time()
            # break
            # restart_target_bank()  # option to restart without initialize


##### todo:
"""
check IoU between previously sprayed masks to new masks to be added (or between OBBs)
"""