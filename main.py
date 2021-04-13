import sys
import g_param
import numpy as np
import cv2 as cv
import DAQ_BG
from test_one_record import test_spec
from preprocessing_and_adding import preprocess_one_record
from distance import distance2
import tensorflow as tf
import time
import Target_bank as TB_class
import transform
import read_write
from sty import fg, Style, RgbFg

# from Target_bank import print_grape
# uncomment this line and comment next for field exp
# from mask_rcnn import take_picture_and_run as capture_update_TB, show_in_moved_window
from masks_for_lab import take_picture_and_run as capture_update_TB, show_in_moved_window, \
    point_meter_2_pixel, image_resize
import write_to_socket
import read_from_socket
import math

########################################################################################################################
# parameters ###########################################################################################################
########################################################################################################################

fg.orange = Style(RgbFg(255, 150, 50))
fg.red = Style(RgbFg(247, 31, 0))
fg.green = Style(RgbFg(31, 177, 31))
fg.yellow = Style(RgbFg(255, 255, 70))
g_param.read_write_object = read_write.ReadWrite()
rs_rob = read_from_socket.ReadFromRobot()  # FIXME: make it possible to run on "load" when robot is turned off
ws_rob = write_to_socket.Send2Robot()
weight_file_name = r'\saved_CNN_clasifier_noise0.03_learn123_test4_3classes_77_2classes_92.1_try2_class_w0.350.350.3.h5'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sys.path.append("C:/Users/omerdad/Desktop/RFR/")
# start_pos = np.array([-0.31741425, -0.26198481, 0.47430055, -0.67481487, -1.51019764, 0.5783255 ])  # Left pos
start_pos = np.array([-0.252, -0.24198481, 0.52430055, -0.6474185, -1.44296026, 0.59665296])  # check pos
# start_pos = np.array([-0.31745283, -0.03241247,  0.43269234, -0.69831852, -1.50455224,  0.60859664]) # Middle pos
# start_pos = np.array([-0.31741425, 0.04, 0.47430055, -0.69831206, -1.50444873, 0.60875449])  # right pos
step_size = 0.25
# TODO: think of better way to calculate the amount of steps based on range of the movement
number_of_steps = 1  # amount of steps before plat move
steps_counter = 0
moving_direction = "right"  # right/left
sleep_time = 3.05
step_direction = ["right", "up", "right", "down"]  # the order of movement
direction = None
g_param.init()
first_run = True
g_param.show_images = True  # g.show_images: if true, it is visualizes the process
time_to_move_platform, external_signal_all_done = False, False
not_finished = True


########################################################################################################################
# ---------- controlling the robot ----------

def init_variables():
    """
    init Trans, read_write_object
    """
    g_param.trans = transform.Trans()
    read_write_init()


def read_write_init():
    g_param.read_write_object.create_directory()
    if g_param.process_type == "load":
        g_param.read_write_object.create_simulation_config_file()


def line_division(p1, p2, ratio):
    x = p1[0] + ratio * (p2[0] - p1[0])
    y = p1[1] + ratio * (p2[1] - p1[1])
    return np.array([x, y])


def move_and_spray(start, end, target_grape):
    if g_param.process_type != "load":
        if possible_move(start, target_grape) and possible_move(end, target_grape):
            ws_rob.move_command(True, start, 5)
            ws_rob.spray_command(True)
            ws_rob.move_command(True, end, 5)
            ws_rob.spray_command(False)
        else:
            pass
    return


def calc_path_points(pos, x_c, y_c, p):
    pos[1] = pos[1] + (p[0] - x_c)
    pos[2] = pos[2] + (y_c - p[1])
    return pos


def calc_spray_diameter(x_max):
    # the function will calculate how close the sprayer can get to the target.
    # By knowing that we will calculate the spray diameter
    pass


# tell me which input you want. I don't think that I need any output,
# maybe only if there is a problem such as no more spraying material.
def spray_procedure(g, d, k):
    """
    :param g: The target grape for spraying
    :param d: The nozzle diameter
    :param k:
    """
    # s = g.w_meter * g.h_meter
    # a = k * (g.mask/s)
    a = 0.8
    center = read_position()
    p1 = g.corners[0]
    p2 = g.corners[1]
    p3 = g.corners[2]
    p4 = g.corners[3]
    x_c = g.x_center
    y_c = g.y_center
    print("target's width: ", g.w_meter)
    N = math.floor(g.w_meter/d + 0.5)
    print("The number of path is: ", N)
    r = (g.w_meter / 2 + d) / g.w_meter
    print("ratio", r)
    if r < 1:
        bottom_point = line_division(p1, p2, 0.5)
        top_point = line_division(p3, p4, 0.5)
        pixel_path = point_meter_2_pixel(g.distance, [bottom_point, top_point])
        display_path_of_spray(g, pixel_path)

        end_p = calc_path_points(np.copy(center), x_c, y_c, bottom_point)
        start_p = calc_path_points(np.copy(center), x_c, y_c, top_point)
        print("end_p", end_p)
        print("start_p", start_p)
        # input("press enter for check spray procedure")
        print(fg.yellow + "wait" + fg.rs, "\n")
        move_and_spray(start_p, end_p, g)
        print(fg.green + "continue" + fg.rs, "\n")
    else:
        r = 0.33

    # secondary paths
    bottom_point = line_division(p1, p2, r)
    top_point = line_division(p4, p3, r)
    bottom_point = line_division(top_point, bottom_point, a)
    pixel_path = point_meter_2_pixel(g.distance, [bottom_point, top_point])
    display_path_of_spray(g, pixel_path)
    end_p = calc_path_points(np.copy(center), x_c, y_c, bottom_point)
    start_p = calc_path_points(np.copy(center), x_c, y_c, top_point)
    print("end_p", end_p)
    print("start_p", start_p)
    input("press enter for check spray procedure")
    print(fg.yellow + "wait" + fg.rs, "\n")
    move_and_spray(start_p, end_p, g)
    print(fg.green + "continue" + fg.rs, "\n")
    bottom_point = line_division(p1, p2, 1 - r)
    top_point = line_division(p4, p3, 1 - r)
    bottom_point = line_division(top_point, bottom_point, a)
    pixel_path = point_meter_2_pixel(g.distance, [bottom_point, top_point])
    display_path_of_spray(g, pixel_path)
    end_p = calc_path_points(np.copy(center), x_c, y_c, bottom_point)
    start_p = calc_path_points(np.copy(center), x_c, y_c, top_point)
    print("end_p", end_p)
    print("start_p", start_p)
    input("press enter for check spray procedure")
    print(fg.yellow + "wait" + fg.rs, "\n")
    move_and_spray(start_p, end_p, g)
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


def update_distance(*args):
    pass


# will include creating connection between the different elements (end effectors)
def read_position():
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
            ws_rob.move_command(False, new_location, 5)
            # check_update_move(new_location)
    else:
        pass  # TODO: Sigal - do we need to save the spray_procedure location and sonar location??
    print(fg.green + "continue" + fg.rs, "\n")


def move2spray(grape_1):
    tcp = g_param.trans.aim_spray(grape_1.x_center, grape_1.y_center, grape_1.distance)
    print("grape dist: ", grape_1.distance)
    new_location = g_param.trans.tcp_base(tcp)
    print("spray_procedure location: ", new_location)
    location = read_position()
    print("old location ", location)
    yz_move = np.concatenate([location[0:1], new_location[1:6]])
    print("y_movbe", yz_move)
    input("press enter to move to spray_procedure location")
    print(fg.yellow + "wait" + fg.rs, "\n")
    if g_param.process_type != "load":
        if possible_move(new_location, grape_1):
            ws_rob.move_command(False, yz_move, 4)
            ws_rob.move_command(True, new_location, 2)
            # check_update_move(new_location)
    else:
        pass  # TODO: Sigal - do we need to save the spray_procedure location and spray_procedure location??
    print(fg.green + "continue" + fg.rs, "\n")


def move2capture():
    print(g_param.trans.capture_pos)
    location = read_position()
    print("old location ", location)
    x_move = np.concatenate([g_param.trans.capture_pos[0:1], location[1:3], g_param.trans.capture_pos[3:6]])
    print("x_move", x_move)
    input("Press Enter to move to capture location")
    print(fg.yellow + "wait" + fg.rs, "\n")
    if g_param.process_type != "load":
        ws_rob.move_command(True, x_move, 5)
        ws_rob.move_command(False, g_param.trans.capture_pos, 5)
    else:
        pass  # TODO: Sigal - do we need to save the temp location and spray_procedure location??
    print(fg.green + "continue" + fg.rs, "\n")


# I switched between the names
def check_update_move(goal_pos):
    time.sleep(1.5)
    curr_pos = np.around(read_position(), 2)
    if not np.array_equal(curr_pos, np.around(goal_pos, 2)):
        print("The arm can not reach the goal position!!!")
        input("press enter to move back to capture position")
        move2capture()


def possible_move(goal_pos, grape_1):
    """
    :param goal_pos: goal position for to move to.
    :param grape_1: The grape according to the parameters are checked
    :return: True if goal position is in reach of the robot, False else
    """
    z_max = 0.82
    z_min = 0.35
    y_max = 0.6
    euclid_max = 0.95
    euclid_dist = np.linalg.norm(np.array([0, 0, 0]) - goal_pos[0:3])
    print("euclid_dist ", euclid_dist)
    if abs(goal_pos[1]) > y_max:
        print("Target too right, move platform")
        g_param.TB[grape_1.index].in_range = "right"
        g_param.time_to_move_platform = True
        return False
    elif goal_pos[2] > z_max:
        print("Target too high!")
        g_param.TB[grape_1.index].in_range = "high"
        return False
    elif goal_pos[2] < z_min:
        print("Target too low!")
        g_param.TB[grape_1.index].in_range = "low"
        return False
    elif euclid_dist > euclid_max:
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
    size_of_step = calc_step_size(size_of_step)
    print("move const ", step_size, " start at:", location)
    if g_param.process_type != "load":
        if direction_to_move == "right":
            location[1] = location[1] + size_of_step
            ws_rob.move_command(True, location, sleep_time)
        elif direction_to_move == "up":
            location[2] = location[2] + size_of_step
            ws_rob.move_command(True, location, sleep_time)
        elif direction_to_move == "down":
            location[2] = location[2] - size_of_step
            ws_rob.move_command(True, location, sleep_time)
        elif direction_to_move == "stay":
            pass
        else:
            location[1] = location[1] - size_of_step
            ws_rob.move_command(True, location, sleep_time)
        # check_update_move(location) # FIXME: omer
        pos = read_position()
        if g_param.process_type == "record":
            g_param.read_write_object.write_location_to_csv(pos=pos)
    else:
        pos = g_param.read_write_object.read_location_from_csv()
    g_param.trans.set_capture_pos(pos)
    if not time_to_move_platform:
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
        ws_rob.move_command(True, start_pos, 4)
        move_platform()  # TODO:Edo - record and load platform movements,platform_step_size
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


# ---------- visualizations ----------

def display_path_of_spray(grape_spray, path_points):
    """
    :param grape_spray: grape to draw the spraying path on top of it
    :param path_points:  points that are the edges of the spraying route
    :return: display image if g_param.show_images is True: show the image and zoomed in image of the path
    """
    for index in range(0, len(path_points) - 1):
        start = path_points[index]
        end = path_points[index + 1]
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
        numpy_horizontal_concat = image_resize(numpy_horizontal_concat, height=945)
        show_in_moved_window("masks, mask zoomed in", numpy_horizontal_concat)
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


def write_distances(mask_id, dist_from_sonar, dist_confirm):
    if g_param.process_type != "record":
        return
    if dist_confirm == "":
        distances = [dist_from_sonar, dist_from_sonar]
    else:
        distances = [dist_from_sonar, float(dist_confirm)]
    g_param.read_write_object.write_sonar_distances(mask_id, distances)


def get_distances(mask_id, dist_from_sonar, real_dist):
    if g_param.process_type == "record":
        write_distances(mask_id, dist_from_sonar, real_dist)
    if g_param.process_type == "load":
        meas, real = g_param.read_write_object.read_sonar_distances(mask_id)
        print(f'Real distance to grape {real} will be used. value measured by sonar is {meas}')
        return real
    return dist_from_sonar


# calling the 2 sonar methods to get distance and validate existence of grapes


def activate_sonar(mask_id):
    """
    activate sonar- read distance and grape/no grape.
    :return: distance to grape, grape (yes=1,no =0)
    """
    counter = 0  # initiate flag
    record = get_class_record(mask_id)
    # if counter == 1 means new acquisition # adding (way of pre-process) - gets 'noise' or 'zero_cols'
    counter, x_test = preprocess_one_record(record, counter, adding='noise')
    # there are 2 weights files, each one suits to other pre-process way (adding)
    preds_3classes, no_grapes = test_spec(x_test, weight_file_name)
    # pred[0] - no grape, pred[1] - 1-5 bunches, pred[2] - 6-10 bunches
    # no grapes - True if there are no grapes (pred[0]> 0.5)
    preds_2classes = [preds_3classes[0][0], preds_3classes[0][1] + preds_3classes[0][2]]
    print("classes", preds_2classes)
    # Sonar distance prediction
    transmition_Chirp = DAQ_BG.chirp_gen(DAQ_BG.chirpAmp, DAQ_BG.chirpTime, DAQ_BG.f0, DAQ_BG.f_end,
                                         DAQ_BG.update_freq, DAQ_BG.trapRel)
    # D = correlation_dist(transmition_Chirp, record)
    dist_from_sonar = distance2(record, mask_id)
    # real dist, real distance measured
    real_dist = input(f'Distance measured by the sonar is :{dist_from_sonar}meters.'
                      f' press enter to confirm, or enter real distance (in meters)')
    if real_dist != "":
        real_dist = float(real_dist)
    else:
        real_dist = dist_from_sonar
        # make it more efficent in 1 line ?
    # real_dist = float(real_dist) if real_dist != "" else dist_from_sonar
    distance = get_distances(mask_id, dist_from_sonar, real_dist)
    return distance, preds_2classes[1]  # make sure it converts to True/ False


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
    temp_step_size = g_param.platform_step_size
    if g_param.process_type == "record" or g_param.process_type == "work":
        print("Current platform step size: ", g_param.platform_step_size, '\n',
              "insert number in CM to change it or press Enter to continue")
        # temp_step_size = input("Enter platform step size") #TODO : UNCOMMENT
        if temp_step_size == 'end':
            external_signal_all_done = True
            return
        temp_step_size = ""
        if temp_step_size != "":
            temp_step_size = int(temp_step_size) / 100
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


# mark that the grape is fake and remove mask (to save space in memory)
def update_database_no_grape(index):
    """
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
    print(g_param.TB)
    print_current_location(current_location)
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
    """
    print_line_sep_time()
    cam_0 = np.array([0, 0, 0, 1])
    cam_0_base = np.matmul(g_param.trans.t_cam2base, cam_0)
    print("cam 0 base >>>>>>>>>>>>>>>>>>>>>", cam_0_base)
    print("TB after sorting \n", g_param.TB)
    half_image_left = g_param.half_width_meter
    for a in range(len(g_param.TB)):
        target = g_param.TB[a]
        # sprayed and steel should appear in the image #
        if target.sprayed and abs(target.grape_world[1] - cam_0_base[1]) < half_image_left \
                and target.in_range == "ok":
            print("distance from center of image : ", target.grape_world[1] - cam_0_base[1])
            g_param.masks_image = cv.circle(g_param.masks_image, (int(target.x_p), int(target.y_p)),
                                            radius=4, color=(0, 0, 255), thickness=4)
        # mark orange dot on grapes that are too high/low.
        if target.in_range == "high" or target.in_range == "low":
            g_param.masks_image = cv.circle(g_param.masks_image, (int(target.x_p), int(target.y_p)),
                                            radius=4, color=(255, 165, 0), thickness=4)
    if g_param.show_images:
        show_in_moved_window("Checking status", g_param.masks_image)
        cv.waitKey()
        cv.destroyAllWindows()


def check_end_program_time():
    temp_input = input("Enter 11 to end the program")
    if temp_input == "11":
        return True
    return False


def display_points(g_to_display):
    """
    display 4 points
    :param g_to_display: grape to display
    :return:
    """
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


if __name__ == '__main__':
    init_program()
    print(">>> Start position: ")
    current_location = g_param.trans.capture_pos
    print_current_location(g_param.trans.capture_pos)
    print(np.around(read_position(), 2))
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
            move_const(g_param.step_size, direction, current_location)
            # move_const(g_param.step_size, "right", current_location)  # try to move 1 step size
            if g_param.time_to_move_platform:  # TODO- update manually step size for world location to the right.
                print('issue moving const ')
                print_current_location(current_location)
                break
            current_location = read_position()  # 1 + 2  establish connection with the robot
            g_param.trans.set_capture_pos(current_location)
            if direction == "right":
                steps_counter += 1
        else:
            first_run = False
        # input("Press Enter to take picture")
        print(fg.yellow + "wait" + fg.rs, "\n")
        capture_update_TB()  # 5 + 7-12 inside
        print(fg.green + "continue" + fg.rs, "\n", "TB after detecting first grape:", "\n", g_param.TB)
        grape_ready_to_spray = TB_class.sort_by_and_check_for_grapes('leftest_first')  # 6
        input("press enter for continue to spraying")
        g_param.masks_image = cv.cvtColor(g_param.masks_image, cv.COLOR_RGB2BGR)
        if not first_run:
            mark_sprayed_and_display()
        print(g_param.TB)
        if grape_ready_to_spray:  # 15- yes (change to sorting according to 14, rap the next lines in a function)
            # update_wait_another_round()  # for future work- details inside.
            amount_of_grapes_to_spray = count_un_sprayed()
            for i in range(amount_of_grapes_to_spray):
                # visualization
                g_param.masks_image = cv.putText(g_param.masks_image, str(g_param.TB[i].index), org=(g_param.TB[i].x_p,
                                                                                                     g_param.TB[i].y_p),
                                                 fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
                                                 color=(255, 255, 255), thickness=1, lineType=2)
                display_points(g_to_display=g_param.TB[i])
                if g_param.show_images:
                    show_in_moved_window("Checking", g_param.masks_image)
                    cv.waitKey()
                    cv.destroyAllWindows()

                grape = g_param.TB[i]  # 16 grape is the the most __ in the least, not sprayed
                move2sonar(grape)  # 17
                if g_param.time_to_move_platform:  # 18
                    break  # 19
                # g_param.last_grape_dist, is_grape = activate_sonar(grape.index)  # 20 # FIXME- WORKING WITH SONAR
                g_param.last_grape_dist, is_grape = g_param.avg_dist, True # without sonar usage
                print("distance :", g_param.last_grape_dist, "is_grape :", is_grape)
                if is_grape and g_param.TB[i].in_range == "ok":  # 21 - yes
                    TB_class.update_distance(g_param.TB[i], read_position())  # 23,24
                    print("current location before spray_procedure", current_location)
                    move2spray(grape)  # 25+26
                    if time_to_move_platform:  # 27
                        break  #
                    # spray_procedure_pixels(grape)
                    spray_procedure(grape, d=0.05, k=1)  # 28 # TODO: Omer, generate safety movement
                    move2capture()
                    update_database_sprayed(i)  # 28
                    mark_sprayed_and_display()
                else:
                    update_database_no_grape(i)  # 22
        else:  # 15- no grapes to spray_procedure
            print(print_line_sep_time(), "No more targets to spray_procedure. take another picture")
            if external_signal_all_done:  # 20- yes #TODO: change external signal- keyboard press.
                not_finished = False
                print("Finished- external signal all done")
                break  # 23
        if steps_counter >= number_of_steps and step_direction[(g_param.plat_position_step_number + 1) % 4] == "right":
            g_param.time_to_move_platform = True
            print(print_line_sep_time(), '\n', " move platform", '\n')
            external_signal_all_done = check_end_program_time()
            # break
            # restart_target_bank()  # option to restart without initialize
