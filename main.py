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
# TODO: uncomment this line and comment next for field exp
# from mask_RCNN import take_picture_and_run as capture_update_TB, pixel_2_meter
from masks_for_lab import take_picture_and_run as capture_update_TB, show_in_moved_window
import write_to_socket
import read_from_socket

########################################################################################################################
# parameters ###########################################################################################################
########################################################################################################################

fg.orange = Style(RgbFg(255, 150, 50))
fg.red = Style(RgbFg(247, 31, 0))
fg.green = Style(RgbFg(31, 177, 31))
fg.yellow = Style(RgbFg(255, 255, 70))
g_param.read_write_object = read_write.ReadWrite()
rs_rob = read_from_socket.ReadFromRobot()
ws_rob = write_to_socket.Send2Robot()
weight_file_name = r'\saved_CNN_clasifier_noise0.03_learn123_test4_3classes_77_2classes_92.1_try2_class_w0.350.350.3.h5'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sys.path.append("C:/Users/omerdad/Desktop/RFR/")
# start_pos = np.array([-0.31741425, -0.26198481, 0.47430055, -0.67481487, -1.51019764, 0.5783255 ])  # Left pos
start_pos = np.array([-0.252, -0.24198481, 0.52430055, -0.6474185, -1.44296026,  0.59665296])  # check pos
# start_pos = np.array([-0.31745283, -0.03241247,  0.43269234, -0.69831852, -1.50455224,  0.60859664]) # Middle pos
# start_pos = np.array([-0.31741425, 0.04, 0.47430055, -0.69831206, -1.50444873, 0.60875449])  # right pos
step_size = 0.25
platform_step_size = 0.40
# the next two variables are just for the lab
number_of_steps = int(platform_step_size/step_size)
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
g_param.trans = transform.Trans()
g_param.read_write_object.create_directories()

########################################################################################################################


def update_distance(*args):
    pass


# will include creating connection between the different elements (end effectors)
def read_position():
    cl = rs_rob.read_tcp_pos()  # cl = current_location, type: tuple
    cur_location = np.asarray(cl)
    return cur_location


# calling the 2 sonar methods to get distance and validate existence of grapes
def activate_sonar():
    counter = 0  # intiate flag
    record = DAQ_BG.rec()
    counter, x_test = preprocess_one_record(record, counter,
                                            adding='noise')  # if counter == 1 means new acquisition # adding (way of preprocess) - gets 'noise' or 'zero_cols'
    preds_3classes, no_grapes = test_spec(x_test,
                                          weight_file_name)  # there are 2 weights files, each one suits to other preprocess way (adding)
    # pred[0] - no grape, pred[1] - 1-5 bunches, pred[2] - 6-10 bunches
    # no grapes - True if there are no grapes (pred[0]> 0.5)
    preds_2classes = [preds_3classes[0][0], preds_3classes[0][1] + preds_3classes[0][2]]
    print("classes", preds_2classes)
    # Sonar distance prediction
    transmition_Chirp = DAQ_BG.chirp_gen(DAQ_BG.chirpAmp, DAQ_BG.chirpTime, DAQ_BG.f0, DAQ_BG.f_end,
                                         DAQ_BG.update_freq, DAQ_BG.trapRel)
    # D = correlation_dist(transmition_Chirp, record)
    dist_from_sonar = distance2(record)
    return dist_from_sonar, preds_2classes[1]  # make sure it converts to True/ False


def restart_TB():
    val = input("Enter 1 to restart the program, or Enter to continue: ")
    if val == 1:
        g_param.TB = []
        print("restarted")
    else:
        print("continue the program")


def print_line_sep_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print('-' * 40, current_time, '-' * 40, '\n')


def move_platform():
    pass


def move2sonar(grape_1):
    input("Press Enter to move to sonar")
    print(fg.yellow + "wait" + fg.rs, "\n")
    x_cam, y_cam = grape_1.x_meter, grape_1.y_meter
    tcp = g_param.trans.aim_sonar(x_cam, y_cam)
    new_location = g_param.trans.tcp_base(tcp)
    print(">>>>>new location", new_location)
    if g_param.process_type != "load":
        if possible_move(new_location, grape_1):
            ws_rob.move_command(False, new_location, 5)
            check_update_move(new_location)
    else:
        pass # TODO: Sigal - do we need to save the spray_procedure location and sonar location??
    print(fg.green + "continue" + fg.rs, "\n")


def move2spray(grape_1):
    tcp = g_param.trans.aim_spray(grape_1.x_meter, grape_1.y_meter, grape_1.distance)
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
            ws_rob.move_command(False, yz_move, 2)
            ws_rob.move_command(True, new_location, 2)
            check_update_move(new_location)
    else:
        pass # TODO: Sigal - do we need to save the spray_procedure location and spray_procedure location??
    print(fg.green + "continue" + fg.rs, "\n")


def move2capture():
    """
    :param capture_location_1: 1 is to make it clear that it is a local variable.
    """
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
        pass # TODO: Sigal - do we need to save the temp location and spray_procedure location??
    print(fg.green + "continue" + fg.rs, "\n")


# I switched between the names
def check_update_move(goal_pos):
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
    if abs(goal_pos[1])> y_max:
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





def calc_step_size(m):
    """
    :param m: step size
    :return: step size- move alpha
    """
    direction_of_move = g_param.direction
    if direction_of_move == "up" or direction_of_move == "down":
        m = m * g_param.height_step_size
    return m


def move_const(size_of_step, direction, location):
    """
    calculate if the new position of the arm is in reach. return the position and boolean.
    grape1 = the TB object of the grape
    I assumed that
    type_of_move = move/ spray_procedure/ sonar/ take_picture
    move = advence X cm to the right (arbitrary), not receiving grape as input (not relevant)
    spray_procedure - move to spraying position
    sonar - move to sonar position
    take_picture - move to take_picture from centered position #
    :param size_of_step: step size
    :param direction:
    :param location:
    :return:
    """
    size_of_step = calc_step_size(size_of_step)
    print("move const ", step_size, " start at:", location)
    if g_param.process_type != "load":
        if direction == "right":
            location[1] = location[1] + size_of_step
            ws_rob.move_command(True, location, sleep_time)
        elif direction == "up":
            location[2] = location[2] + size_of_step
            ws_rob.move_command(True, location, sleep_time)
        elif direction == "down":
            location[2] = location[2] - size_of_step
            ws_rob.move_command(True, location, sleep_time)
        else:
            location[1] = location[1] - size_of_step
            ws_rob.move_command(True, location, sleep_time)
        check_update_move(location) # FIXME: omer
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
    :return: void, prints current location
    """
    x_base = "x base: " + str(round(cur_location[0], 3)) + " "
    y_base = "y base: " + str(round(cur_location[1], 3)) + " "
    z_base = "z base: " + str(round(cur_location[2], 3)) + " "
    print("current location : ", x_base + y_base + z_base)


# mark that the grape is fake and remove mask (to save space in memory)
def update_database_no_grape(index):
    g_param.TB[index].fake_grape = True
    g_param.TB[index].mask = None
    g_param.TB[index].sprayed = True
    # check with Sigal if I want to count amount of grapes sprayed
    # any way, I could just subtract the amount of grapes that are sprayed and not fake_grape


# Change the grape to sprayed
def update_database_sprayed(index_of_grape):
    g_param.TB[index_of_grape].sprayed = True
    print(g_param.TB)
    print_current_location(current_location)
    g_param.TB[index_of_grape].mask = None  # (to save space in memory)
    g_param.masks_image = cv.circle(g_param.masks_image,
                                    (g_param.TB[index_of_grape].x_p, g_param.TB[index_of_grape].y_p),
                                    radius=4, color=(0, 0, 255), thickness=4)

    # cv.putText(g_param.masks_image, str(g_param.TB[i].index), org=(g_param.TB[i].x_p + 30,
    #                                                                g_param.TB[i].y_p + 30),
    #            fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=4,
    #            color=(255, 50, 255), thickness=1, lineType=2)


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
    amount = 0
    if len(g_param.TB) > 0:
        for ind in range(len(g_param.TB)):
            if g_param.TB[ind].sprayed is False and not g_param.TB[ind].wait_another_step:
                amount += 1
    print(amount)
    return amount


def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]
            for i in range(1, nb_points+1)]


def move_and_spray(start, end):
    ws_rob.move_command(True, start, 5)
    ws_rob.spray_command(True)
    ws_rob.move_command(True, end, 5)
    ws_rob.spray_command(False)


# tell me which input you want. I don't think that I need any output,
# maybe only if there is a problem such as no more spraying material.
def spray_procedure(g):
    curr_location = read_position()
    end_p = np.copy(curr_location)
    start_p = np.copy(curr_location)
    print("Robot current location ", start_p)
    # print(curr_location)
    p1 = g.corners[0]
    p2 = g.corners[1]
    p3 = g.corners[2]
    p4 = g.corners[3]
    print("p1", p1)
    print("p2", p2)

    x_c = g.x_meter
    y_c = g.y_meter

    bottom_points_1 = intermediates(p1, p2, 1)
    bottom_points_2 = intermediates(p3, p4, 1)
    delta_x_1 = x_c - bottom_points_1[0][0]
    delta_y_1 = y_c - bottom_points_1[0][1]
    delta_x_2 = x_c - bottom_points_2[0][0]
    delta_y_2 = y_c - bottom_points_2[0][1]

    end_p[1] = end_p[1] + delta_x_1
    end_p[2] = end_p[2] + delta_y_1
    start_p[1] = start_p[1] + delta_x_2
    start_p[2] = start_p[2] + delta_y_2
    print("end_p", end_p)
    print("start_p", start_p)
    input("press enter for check spray procedure")
    print(fg.yellow + "wait" + fg.rs, "\n")
    move_and_spray(start_p, end_p)
    print(fg.green + "continue" + fg.rs, "\n")

    # left_path_points = intermediates(top_points[0], bottom_points[0], 3)
    # right_path_points = intermediates(top_points[2], bottom_points[2], 3)


def mark_sprayed_and_display():
    print_line_sep_time()
    cam_0 = np.array([0, 0, 0, 1])
    cam_0_base = np.matmul(g_param.trans.t_cam2base, cam_0)
    print("cam 0 base >>>>>>>>>>>>>>>>>>>>>", cam_0_base)
    print("TB after sorting \n", g_param.TB)
    for a in range(len(g_param.TB)):
        half_image_left = g_param.half_width_meter
        # sprayed and steel should appear in the image #
        if g_param.TB[a].sprayed and abs(g_param.TB[a].grape_world[1] -
                                         cam_0_base[1]) < half_image_left:
            print("distance from center of image : ", g_param.TB[a].grape_world[1] - cam_0_base[1])
            g_param.masks_image = cv.circle(g_param.masks_image, (int(g_param.TB[a].x_p), int(g_param.TB[a].y_p)),
                                            radius=4, color=(0, 0, 255), thickness=4)
    if g_param.show_images:
        show_in_moved_window("Checking status", g_param.masks_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


def init_arm_and_platform():
    """
    move arm before platform movement
    platform movement step size
    move arm after platform movement for first picture position
    """
    # move_command(False, start_pos, 5) #TODO: Omer
    # move_platform(platform_step_size) #TODO write the function
    # TODO, check if change to True, or more complex movement (it's moving in a dangerous way)

    print_line_sep_time()

    if g_param.process_type != "load":
        ws_rob.move_command(True, start_pos, 4)
        if g_param.process_type == "record":
            g_param.read_write_object.write_location_to_csv(pos=read_position())
        current_location = read_position()  # 1 + 2  establish connection with the robot
        g_param.trans.set_capture_pos(current_location)
        g_param.trans.update_cam2base(current_location)  # 4

    else:
        current_location = g_param.read_write_object.read_location_from_csv()
        g_param.trans.set_capture_pos(current_location)
        g_param.trans.update_cam2base(current_location)


def check_more_than_half_away(x_meter, half_step_size):
    """
    :param x_meter: location of middle of the grape in meter
    :param half_step_size: half step size in meters
    :return:
    true if  x_base > half_step_size, then spray_procedure only after next image (when grape will be captured when
    it is closer to the center of the point, which in high probability produce more accurate mask.
    else, return False, meaning that the grape wen't get an image when it is closer to the center.
    """
    return x_meter > half_step_size



if __name__ == '__main__':
    init_arm_and_platform()
    print(">>> Start position: ")
    current_location = g_param.trans.capture_pos
    print_current_location(g_param.trans.capture_pos)
    print(np.around(read_position(), 2))
    # The main loop:
    while not_finished:
        if not first_run:
            g_param.image_number += 1
            print_line_sep_time()
            if g_param.time_to_move_platform:
                init_arm_and_platform()  # 3
                steps_counter = 0
            direction = step_direction[g_param.image_number % 4]
            g_param.direction = direction
            move_const(step_size, direction, current_location)
            # move_const(step_size, "right", current_location)  # try to move 1 step size
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
        input("Press Enter to take picture")  # TODO: add base world location to the input (or inside)
        print(fg.yellow + "wait" + fg.rs, "\n")
        capture_update_TB(current_location, g_param.image_number)  # 5 + 7-12 inside
        print(fg.green + "continue" + fg.rs, "\n",  "TB after detecting first grape:", "\n", g_param.TB)
        grape_ready_to_spray = TB_class.sort_by_and_check_for_grapes('leftest_first')  # 6
        input("press enter for continue to spraying")
        if not first_run:
            mark_sprayed_and_display()
        print(g_param.TB)
        if grape_ready_to_spray:  # 15- yes (change to sorting according to 14, rap the next lines in a function)
            # update_wait_another_round()  # for future work- details inside.
            amount_of_grapes_to_spray = count_un_sprayed()
            for i in range(amount_of_grapes_to_spray):
                # visualization
                g_param.masks_image = cv.putText(g_param.masks_image, str(g_param.TB[i].index), org=(g_param.TB[i].x_p,
                                                 g_param.TB[i].y_p), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
                                                 color=(255, 255, 255), thickness=1, lineType=2)
                if g_param.show_images:
                    show_in_moved_window("Checking", g_param.masks_image)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                grape = g_param.TB[i]  # 16 grape is the the most __ in the least, not sprayed
                print("check>>>>>>>>>>>>", grape.corners)
                move2sonar(grape)  # 17
                if g_param.time_to_move_platform:  # 18
                    break  # 19
                # distance, is_grape = activate_sonar()  # 20 FIXME: Yossi + Edo
                g_param.last_grape_dist, is_grape = g_param.avg_dist, True
                print("distance :", g_param.last_grape_dist, "is_grape :", is_grape)
                if is_grape and g_param.TB[i].in_range == "ok":  # 21 - yes
                    TB_class.update_distance(g_param.TB[i], read_position())  # 23,24
                    print("current location before spray_procedure", current_location)
                    move2spray(grape)  # 25+26
                    if time_to_move_platform:  # 27
                        break  #
                    spray_procedure(grape)  # 28
                    move2capture()  # TODO: Omer, generate safety movement
                    update_database_sprayed(i)  # 28 # TODO Edo: show the image and mark grape sprayed
                    mark_sprayed_and_display()
                else:
                    update_database_no_grape(i)  # 22
        else:  # 15- no grapes to spray_procedure
            print(print_line_sep_time(), "No more targets to spray_procedure. take another picture")
            if external_signal_all_done:  # 20- yes #TODO: Create function to check external signal- keyboard press
                not_finished = False
                print("Finished- external signal all done")
                break  # 23
        if steps_counter >= number_of_steps and step_direction[(g_param.image_number + 1) % 4] == "right":
            g_param.time_to_move_platform = True
            print(print_line_sep_time(), '\n', " Safety STOP!", '\n')
            break
            # restart_TB()  # option to restart without initialize
