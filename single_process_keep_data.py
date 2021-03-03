# grape = ID, mask, pixel center, world center, angle, pixel width
# , pixel length , sprayed, dist_to_center
#
# dist_to_center = euclidean distance between pixel_center to center of image.
# ID = unique ID per grape cluster.
# grapes = grapes detected in image
# target_bank = list of grapes, sorted first by sprayed/ not sprayed, second by a chosen parameter
# in our case for example always the grape that is in the opposite direction of advancement
# import packages and the ReadFromRobot class- do not change

# list of parameters to tune:
#     step_size = 0.45
#     sleep_time = 2.9
#     safety_dist = 0.30
#     distance = 680 # change
#     same_grape_distance_threshold = 9 cm (0.09m)
#     show_images = True (default) show images

import sys
import numpy as np
import cv2 as cv
import tensorflow as tf
import time
import logging

logging.getLogger('tensorflow').disabled = True


# from Target_bank import print_grape
# TODO: uncomment this line and comment next for field exp
# from mask_rcnn import take_picture_and_run as capture_update_TB, pixel_2_meter
from masks_for_lab import take_picture_and_run as capture_update_TB, pixel_2_meter, showInMovedWindow, meter_2_pixel
import write_to_socket
import read_from_socket
import Target_bank as TBK
import transform
import read_write
import DAQ_BG
from test_one_record import test_spec
from preprocessing_and_adding import preprocess_one_record
from distance import distance2
import g_param



g_param.read_write_object = read_write.ReadWrite()
rs_rob = read_from_socket.ReadFromRobot()
ws_rob = write_to_socket.Send2Robot()

weight_file_name = r'\saved_CNN_clasifier_noise0.03_learn123_test4_3classes_77_2classes_92.1_try2_class_w0.350.350.3.h5'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sys.path.append("C:/Users/omerdad/Desktop/RFR/")

########################################################################################################################
# parameters ###########################################################################################################
########################################################################################################################
start_pos = np.array([-0.30741425, -0.24198481, 0.52430055, -0.67481487, -1.51019764, 0.5783255])  # check pos
# start_pos = np.array([-0.38741425, -0.26198481, 0.47430055, -0.67481487, -1.51019764, 0.5783255])  # check pos
# start_pos = np.array([-0.31741425, -0.26198481, 0.47430055, -0.67481487, -1.51019764, 0.5783255 ])  # Left pos
# start_pos = np.array([-0.31741425, -0.26198481, 0.47430055, -0.67481487, -1.51019764, 0.5783255])  # check pos # TODO ORIGINAL
# start_pos = np.array([-0.31745283, -0.03241247,  0.43269234, -0.69831852, -1.50455224,  0.60859664]) # Middle pos
# start_pos = np.array([-0.31741425, 0.04, 0.47430055, -0.69831206, -1.50444873, 0.60875449])  # right pos
step_size = 0.25
# the next two variables are just for the lab
number_of_steps = 3
steps_counter = 0
moving_direction = "right"  # right/left
platform_step_size = 0.5
sleep_time = 3.1
g_param.init()
first_run, g_param.show_images = True, True  # g.show_images: if true, it is visualizes the process
time_to_move_platform, external_signal_all_done = False, False
not_finished, no_tech_problem = True, True  # no_tech_problem will be checked as part of init
g_param.trans = transform.Trans()
g_param.read_write_object.create_directories()
########################################################################################################################


def update_distance(*args):
    pass


# will include creating connection between the different elements (end effectors)
def read_position():
    cl = rs_rob.read_tcp_pos()  # cl = current_location, type: tuple
    # no_tech_problem = check_connected_succsesfully()
    cur_location = np.asarray(cl)
    return cur_location, True  # TODO: check_connected_succsesfully


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


def check_connected_successfully():
    pass


def move2sonar(grape_1):
    input("Press Enter to to sonar")
    x_cam, y_cam = grape_1.x_meter, grape_1.y_meter
    tcp = g_param.trans.aim_sonar(x_cam, y_cam)
    new_location = g_param.trans.tcp_base(tcp)
    print(">>>>>new location", new_location)
    ws_rob.move_command(False, new_location, 5)
    check_update_move(new_location)


def move2spray(grape_1):
    tcp = g_param.trans.aim_spray(grape_1.x_meter, grape_1.y_meter, grape_1.distance)
    print("grape dist: ", grape_1.distance)
    new_location = g_param.trans.tcp_base(tcp)
    print("spray location: ", new_location)
    location = read_position()[0]
    print("old location ", location)
    yz_move = np.concatenate([location[0:1], new_location[1:6]])
    print("y_movbe", yz_move)
    input("press enter to move to spray location")
    ws_rob.move_command(False, yz_move, 4)
    ws_rob.move_command(True, new_location, 5)
    check_update_move(new_location)


def move2temp(temp_location_1):
    print(temp_location_1)
    location = read_position()[0]
    print("old location ", location)
    x_move = np.concatenate([temp_location_1[0:1], location[1:3], temp_location_1[3:6]])
    print("x_movbe", x_move)
    input("Press Enter to move to temp location")
    ws_rob.move_command(True, x_move, 5)
    ws_rob.move_command(False, temp_location_1, 5)
    check_update_move(temp_location_1)


def check_update_move(location):
    new_tcp = read_position()[0]  # in case the can not reach the next position.
    if np.array_equal(new_tcp[3:], location):
        g_param.time_to_move_platform = True


def move_const(m, direction, location):
    """
    calculate if the new position of the arm is in reach. return the position and boolean.
    grape1 = the TB object of the grape
    I assumed that
    type_of_move = move/ spray/ sonar/ take_picture
    move = advence X cm to the right (arbitrary), not receiving grape as input (not relevant)
    spray - move to spraying position
    sonar - move to sonar position
    take_picture - move to take_picture from centered position #TODO: not for now
    :param m:
    :param direction:
    :param location:
    :return:
    """
    print("move const ", step_size, " start at:", location)
    if direction == "right":
        location[1] = location[1] + m
        ws_rob.move_command(True, location, sleep_time)
    else:
        location[1] = location[1] - m
        ws_rob.move_command(True, location, sleep_time)
    check_update_move(location)
    if not time_to_move_platform:
        g_param.trans.update_cam2base(read_position()[0])


def print_current_location():
    x_base = "x base: " + str(round(current_location[0], 3)) + " "
    y_base = "y base: " + str(round(current_location[1], 3)) + " "
    z_base = "z base: " + str(round(current_location[2], 3)) + " "
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
    print_current_location()
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


# tell me which input you want. I don't think that I need any output,
# maybe only if there is a problem such as no more spraying material.
def spray():
    ws_rob.spray_command(True)
    time.sleep(0.2)
    ws_rob.spray_command(False)


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
        showInMovedWindow("Checking status", g_param.masks_image)
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
    ws_rob.move_command(False, start_pos, 4)


def check_more_than_half_away(x_meter, half_step_size):
    """
    :param x_meter: location of middle of the grape in meter
    :param half_step_size: half step size in meters
    :return:
    true if  x_meter > half_step_size, then spray only after next image (when grape will be captured when
    it is closer to the center of the point, which in high probability produce more accurate mask.
    else, return False, meaning that the grape wen't get an image when it is closer to the center.
    """
    return x_meter > half_step_size


if __name__ == '__main__':

    init_arm_and_platform()
    current_location, no_tech_problem = read_position()  # 1 + 2  establish connection with the robot
    temp_location = current_location
    g_param.trans.set_pic_tcp(temp_location)
    g_param.trans.update_cam2base(current_location)  # 4

    print(">>> Start position: ")
    print_current_location()

    # The main loop:
    while not_finished and no_tech_problem:
        g_param.image_number += 1
        if not first_run:
            print_line_sep_time()
            if g_param.time_to_move_platform:
                init_arm_and_platform()  # 3
                steps_counter = 0
            move_const(step_size, "right", current_location)  # try to move 1 step size
            if g_param.time_to_move_platform:
                print('issue moving const ')
                print_current_location()
                break
            current_location, no_tech_problem = read_position()  # 1 + 2  establish connection with the robot
            temp_location = current_location
            g_param.trans.set_pic_tcp(temp_location)
            g_param.trans.update_cam2base(current_location)
            g_param.read_write_object.write_transformations_to_csv()
            steps_counter += 1
        else:
            first_run = False

        input("Press Enter to take picture")
        capture_update_TB(current_location, g_param.image_number)  # 5 + 7-12 inside # TODO: add base world location to the input (or inside)
        g_param.read_write_object.write_target_bank_to_csv(g_param.TB)
        print("TB after detecting first grape:", "\n", g_param.TB)
        grape_ready_to_spray = TBK.sort_by_and_check_for_grapes('leftest_first')  # 6
        input("press enter for continue to spraying")
        if not first_run:
            mark_sprayed_and_display()
        print(g_param.TB)
        if grape_ready_to_spray:  # 15- yes (change to sorting according to 14, rap the next lines in a function)
            # update_wait_another_round()  # TODO comment it
            amount_of_grapes_to_spray = count_un_sprayed()
            for i in range(amount_of_grapes_to_spray):

                g_param.masks_image = cv.putText(g_param.masks_image, str(g_param.TB[i].index), org=(g_param.TB[i].x_p,
                                                                                                     g_param.TB[i].y_p),
                                                 fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
                                                 color=(255, 255, 255), thickness=1, lineType=2)
                if g_param.show_images:
                    showInMovedWindow("Checking", g_param.masks_image)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                grape = g_param.TB[i]  # 16 grape is the the most __ in the least, not sprayed
                move2sonar(grape)  # 17
                if g_param.time_to_move_platform:  # 18
                    break  # 19
                # distance, is_grape = activate_sonar()  # 20 FIXME: Yossi  + Edo
                sonar_dist, is_grape = g_param.const_dist, True
                print("distance :", sonar_dist, "is_grape :", is_grape)
                # move2temp(temp_location) # TODO: Omer, later generate direct movement from sonar to sprayer
                if is_grape:  # 21 - yes
                    update_distance(i, sonar_dist)  # 23,24 TODO: omer ,Edo- update grape parameters
                    g_param.trans.update_distance(read_position()[0])
                    print("current location before spray", current_location)
                    move2spray(grape)  # 25+26
                    if time_to_move_platform:  # 27
                        break  #
                    spray()  # 28
                    move2temp(temp_location)  # TODO: Omer, generate safety movement
                    update_database_sprayed(i)  # 28 # TODO Edo: show the image and mark grape sprayed
                    mark_sprayed_and_display()
                else:
                    update_database_no_grape(i)  # 22
        else:  # 15- no grapes to spray

            print(print_line_sep_time(), "No more targets to spray. take another picture")
            if external_signal_all_done:  # 20- yes #TODO: Create function to check external signal- keyboard press
                not_finished = False
                print("Finished- external signal all done")
                break  # 23
            # else:  # 20 - no
            #     if g_param.time_to_move_platform:  # 21- yes
            #         input("Time to move platform, Press Enter to move platform")
            #         # init_arm_and_platform() #TODO: check this function
            #         # steps_counter = 0
            #         # g_param.trans.update_base2world()
            #     else:  # 21- no
            #         print("Not time to move platform, move arm to take another image")
        if steps_counter > number_of_steps:
            g_param.time_to_move_platform = True
            # restart_TB()  # option to restart without initialize
