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
import g
import numpy as np
import cv2 as cv
import math
import DAQ_BG
from test_one_record import test_spec
from preprocessing_and_adding import preprocess_one_record
from distance import distance2
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
import tensorflow as tf
weight_file_name = r'\saved_CNN_clasifier_noise0.03_learn123_test4_3classes_77_2classes_92.1_try2_class_w0.350.350.3.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sys.path.append("C:/Users/omerdad/Desktop/RFR/")
import socket
import time
import Target_bank as TBK
# from Target_bank import print_grape
# TODO: uncomment this line and comment next for field exp
# from mask_rcnn import take_picture_and_run as capture_update_TB, pixel_2_meter
from masks_for_lab import take_picture_and_run as TPAR, pixel_2_meter, showInMovedWindow, meter_2_pixel
from Send2UR5 import read_tcp_pos, move_command, spray_command

g.init()

# T_TCP_C = np.array([[0.7071, -0.7071, 0, 0.065], [0.7071, 0.7071, 0, -0.075], [0, 0, 1, 0], [0, 0, 0, 1]])
T_C_TCP = np.array([[0.7071, -0.7071, 0, 0.08216], [0.7071, 0.7071, 0, -0.060677], [0, 0, 1, 0], [0, 0, 0, 1]]) # FIXME: Old values
#T_C_TCP = np.array([[0, 0, -1, 0], [0.7071, -0.7071, 0, 0.01], [0.7071, 0.7071, 0, 0.005], [0, 0, 0, 1]]) # FIXME Omer! we changed the matrix, it's not the values you used.

#Vm_TCP = np.array([-0.045, -0.1, 0.12])
#Vs_TCP = np.array([0.11, 0.07, 0])
Vs_TCP = np.array([0.152735, -0.00282843, 0])
Vm_TCP = np.array([-0.0311127, -0.10323759, 0.12])

# will include creating connection between the diefferent elemnts (end effectors)
def read_position():
    cl = read_tcp_pos()  # cl = current_location, type: tuple
    # no_tech_problem = check_connected_succsesfully()
    current_location = np.asarray(cl)
    return current_location, True  # TODO: check_connected_succsesfully


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
    transmition_Chirp = DAQ_BG.chirp_gen(DAQ_BG.chirpAmp, DAQ_BG.chirpTime, DAQ_BG.f0, DAQ_BG.f_end, DAQ_BG.update_freq,
                                         DAQ_BG.trapRel)
    # D = correlation_dist(transmition_Chirp, record)
    dist_from_sonar = distance2(record)
    return dist_from_sonar, preds_2classes[1]  # make sure it converts to True/ False


def restart_TB():
    val = input("Enter 1 to restart the program, or Enter to continue: ")
    if val == 1:
        g.TB = []
        print("restarted")
    else:
        print("continue the program")


def move_platform():
    pass


def check_time_to_move_platform(current_location):
    return False


def check_connected_successfully():
    pass


# gets the current_position and the next_pos. moves the arm
# (I don't know what are the limitations you have to consider
def move_arm(current_location, next_pos):
    pass


# calculate rotation matrix
def rotation_matrix(k, ct, st):
    R = np.array(
        [[pow(k[0], 2) * (1 - ct) + ct, k[0] * k[1] * (1 - ct) - k[2] * st, k[0] * k[2] * (1 - ct) + k[1] * st],
         [k[0] * k[1] * (1 - ct) + k[2] * st, pow(k[1], 2) * (1 - ct) + ct, k[1] * k[2] * (1 - ct) - k[0] * st],
         [k[0] * k[2] * (1 - ct) - k[1] * st, k[1] * k[2] * (1 - ct) + k[0] * st,
          pow(k[2], 2) * (1 - ct) + ct]])  # rotation matrix
    return R


# Turns axis angles  into rotation matrix
def angVec2Rot(RV):
    # print ("RV ", RV)
    t = np.linalg.norm(RV)
    # print("t ", t)
    k = RV / t
    # print("k ", k)
    ct = math.cos(t)
    st = math.sin(t)
    R = rotation_matrix(k, ct, st)
    # print("Rotation matrix:\n", R)
    return R


# move the component in front of the center of the cluster
def move2Cluster(Vg_TCP, V, T_TCP_B, rTCP):
    V = Vg_TCP - V
    # print("V_tag: ", V)
    V_tag = np.append(V, [1], axis=0)
    V_2tag = np.matmul(T_TCP_B, V_tag)
    # print("V_2tag: ", V_2tag)
    newPos = np.concatenate((V_2tag[:3], rTCP), axis=0)
    print("new position: ", newPos)
    move_command(False, newPos, 5)


def move_component(grape_to_spray, cur_location, component):
    x_cam, y_cam = grape_to_spray.x_world_meter, grape_to_spray.y_world_meter
    print("x_cam and y_cam: ",  x_cam, y_cam)
    print("Current location: ", cur_location)
    R = angVec2Rot(cur_location[3:])

    T_TCP_B = np.append(R, [[0, 0, 0]], axis=0)
    T_TCP_B = np.append(T_TCP_B, [[cur_location[0]], [cur_location[1]], [cur_location[2]], [1]], axis=1)
    V_cam = np.array([float(x_cam), float(y_cam), 0, 1])
    Vg_TCP = np.matmul(T_C_TCP, V_cam)
    Vg_TCP = Vg_TCP[:-1]
    if component == "sonar":
        print("move_component, cur_location[3:]", cur_location[3:])
        input("Press enter for move to sonar pos")
        move2Cluster(Vg_TCP, Vs_TCP, T_TCP_B, cur_location[3:]) #TODO Omer: commented to save checking time
    else:  # spray
        print("Vm_TCP: ", Vm_TCP)
        input("Press enter for spray")
        move2Cluster(Vg_TCP, Vm_TCP, T_TCP_B, cur_location[3:]) #TODO Omer: commented to save checking time

# FIXME Omer: I changed the input from grape to x,y. also d was in mm and the other units are meters
def cam_2_base (x_world_meter, y_world_meter, cur_location, d):
    d = d*0.001 # convert from mm to m
    x_cam, y_cam = x_world_meter, y_world_meter
    R = angVec2Rot(cur_location[3:])
    T_TCP_B = np.append(R, [[0, 0, 0]], axis=0)
    T_TCP_B = np.append(T_TCP_B, [[cur_location[0] - d], [cur_location[1]], [cur_location[2]], [1]], axis=1) # FIXME: Omer- I changed to list
    V_cam = np.array([float(x_cam), float(y_cam), 0, 1])
    Vg_TCP = np.matmul(T_C_TCP, V_cam)
    print("V_cam", V_cam)
    print("Vg_TCP", Vg_TCP)
    Vg_B = np.matmul(T_TCP_B, Vg_TCP)
    Vg_B = Vg_B[:-1]
    return Vg_B


def move_const(m, direction, pos):
    """
    :param m: step distance in meters
    :param direction: motion direction
    :param pos: current position
    Move the robot a fixed distance left or right, the motion is linear on the Y axis.
    """
    # input("Press Enter to move arm")
    print("move const 15 start at: ", pos)
    if direction == "right":
        pos[1] = pos[1] + m
        move_command(True, pos, sleep_time) # FIXME: this one uncommented
    else:
        pos[1] = pos[1] - m
        move_command(True, pos, sleep_time) # FIXME: this one uncommented
    # TODO Omer: add limitation to the robot's movement
    new_pTCP = read_position()[:-3]  # in case the can not reach the next position.
    if np.array_equal(new_pTCP, pos[:-3]):
        return True
    else:
        return False


# calculate if the new position of the arm is in reach. return the position and boolean.
# grape1 = the TB object of the grape
# I assumed that
# type_of_move = move/ spray/ sonar/ take_picture
# move = advence X cm to the right (arbitrary), not receiving grape as input (not relevant)
# spray - move to spraying position
# sonar - move to sonar position
# take_picture - move to take_picture from centered position #TODO: not for now

def move_to_next_pos(current_location, grape_to_spray, type_of_move):
    if type_of_move == "move":
        return move_const(step_size, "right", current_location)
    elif type_of_move == "spray":  # TODO: and return
        move_component(grape_to_spray, current_location, "spray")
    elif type_of_move == "sonar":
        move_component(grape_to_spray, current_location, "sonar")
    elif type_of_move == "take_picture":
        pass
    return False
    # return True/False, [x,y,z,Rx,Ry,Rz]
    # The return should be the boolean value if the new pos is within reach from the current position
    # if the value is True, then the new coordinates will be returned. else current position is returned.

def update_Vm_tcp (distance):
    Vm_TCP[2] = Vm_TCP[2] - (distance / 1000) + safety_dist

# tell me which input you want. I don't think that I need any output,
# maybe only if there is a problem such as no more spraying matirel.
def spray():
    pass


def update_cam2world_transform(a):
    return 2

# calculates the DPI to cm conversion. based on 100 dpi https://www.pixelto.net/cm-to-px-converter
def update_database_remove_items(current_location):
    pass


# mark that the grape is fake and remove mask (to save space in memory)
def update_database_no_grape(index):
    g.TB[i].fake_grape = True
    g.TB[i].mask = None
    g.TB[i].sprayed = True
    # check with Sigal if I want to count amount of grapes sprayed
    # any way, I could just subtract the amount of grapes that are sprayed and not fake_grape


# Change the grape to sprayed
def update_database_sprayed(index_of_grape):
    print(g.TB)
    g.TB[index_of_grape].sprayed = True
    print(g.TB)
    g.TB[index_of_grape].mask = None  # (to save space in memory)
    g.masks_image = cv.circle(g.masks_image, (g.TB[index_of_grape].x_p, g.TB[index_of_grape].y_p),
                              radius=4, color=(0, 0, 255), thickness=4)


def count_un_sprayed():
    amount = 0
    if len(g.TB) > 0:
        for ind in range(len(g.TB)):
            if g.TB[ind].sprayed is False:
                amount += 1
    return amount

# update in the database that the grape was sprayed
def update_database_sonard(index, real_distance):
    g.TB[index].distance = real_distance


def update_world_transform():
    for i in range(len(g.TB)):
        if g.TB[i].x_world_meter < -0.5:  # change to parameter
            continue
        else:
            g.TB[i].x_world_meter = round((g.TB[i].x_world_meter - 0.15), 3)
            update_pixel_location(i)


# Not in used for now
def update_pixel_location(i):  # fix the conversion (it's not 37.795)
    pass
    # meter_2_pixel(distance, i)
    # conversion_ratio = 37.795
    # conversion_ratio = 30.795
    # conversion_ratio = 25.795
    # print("old xp before update: ", g.TB[i].x_p)
    # g.TB[i].x_p -= step_size * 100 * conversion_ratio
    # print("new xp after update: " , g.TB[i].x_p )


def spray():
    pass


def mark_sprayed_and_display():

    print("TB after sorting \n", g.TB)
    for i in range(len(g.TB)):
        half_image_left = -0.35  # TODO calc it based on distance
        if g.TB[i].sprayed and g.TB[i].x_world_meter > half_image_left: # sprayed and steel should appear in the image #
            g.masks_image = cv.circle(g.masks_image, (int(g.TB[i].x_p), int(g.TB[i].y_p)),
                                      radius=4, color=(0, 0, 255), thickness=4)
    if g.show_images:
        showInMovedWindow("Checking status", g.masks_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


# TODO: add another parameter of location before all spraying and sonar positions. save it.
if __name__ == '__main__':
    start_pos = np.array([-0.31741425, -0.26198481,  0.47430055, -0.69831206, -1.50444873,  0.60875449])  # Left pos
    # start_pos = np.array([-0.31745283, -0.03241247,  0.43269234, -0.69831852, -1.50455224,  0.60859664]) # Middle pos
    # start_pos = np.array([-0.31741425, 0.04, 0.47430055, -0.69831206, -1.50444873, 0.60875449])  # right pos
    move_command(False, start_pos, 5)
    time.sleep(0.3)
    first_run, g.show_images = True, True  # g.show_images: if true, it is visualizes the process
    current_location, no_tech_problem = read_position()  # 1 + 2  establish connection with the robot
    temp_location = current_location
    time_to_move_platform, external_signal_all_done = False, False
    not_finished, no_tech_problem = True, True  # no_tech_problem will be checked as part of init
    step_size = 0.45
    sleep_time = 3.2
    safety_dist = 0.30
    distance = 680 # change
    update_Vm_tcp(distance)
    print(">>> Start position: ", current_location)
    # The main algorithm:
    # filming_position is the next position to go after taking a picture
    while not_finished and no_tech_problem:
        print(g.TB)
        # filming_position = current_location
        if first_run:
            first_run = False
        else:
            print("I moved 15 cm!!!!!!!!!!!!!!")
            print("current location befor moving 15", current_location)
            time_to_move_platform = move_to_next_pos(current_location, None, 'move')
            current_location, no_tech_problem = read_position()
            temp_location = current_location
            print("current location after moving 15: ", current_location)
            print("temp location after moving 15: ", temp_location)
            update_world_transform()
        # TODO: calc x,y locations of all new points
        distance = 680
        # distance = input("Press Enter distance in mm before taking a picture")
        input("Press Enter to take picture")
        TPAR(distance, current_location)  # 5 + 7-12 inside # TODO: add base world location to the input (or inside)
        print("TB after detecting first grape:", "\n", g.TB)
        grape_ready_to_spray = TBK.sort_by_and_check_for_grapes('leftest_first')  # 6
        input("press enter for continue to spraying")
        if not first_run:
            mark_sprayed_and_display()
        # print("TB after sorting \n", g.TB)
        # for i in range(len(g.TB)):
        #     if g.TB[i].sprayed and g.TB[i].x_meter > -0.5: # sprayed and steel should appear in the image
        #         g.masks_image = cv.circle(g.masks_image, (int(g.TB[i].x_p), int(g.TB[i].y_p)),
        #                                   radius=4, color=(0, 0, 255), thickness=4)
        # showInMovedWindow("Checking status", g.masks_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        if grape_ready_to_spray is None:  # 13- no
            print("No more targets to spray. take another picture")
            if external_signal_all_done:  # 20- yes #TODO: Create function to check external signal
                not_finished = False
                print("Finished- external signal all done")
                break  # 23
            else:  # 20 - no
                if time_to_move_platform:  # 21- yes
                    input("Time to move platform, Press Enter to move platform")
                    move_platform() #TODO: write this function
                else:  # 21- no
                    print("Not time to move platform, move arm to take another image")

        else:  # 13- yes (change to sorting according to 14, rap the next lines in a function)
            amount_of_grapes_to_spray = count_un_sprayed()
            for i in range(amount_of_grapes_to_spray):
                g.masks_image = cv.putText(g.masks_image, str(g.TB[i].index), org=(g.TB[i].x_p, g.TB[i].y_p),
                                           fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
                                           color=(255, 255, 255), thickness=1, lineType=2)
                if g.show_images:
                    showInMovedWindow("Checking", g.masks_image)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                print(g.TB)
                grape = g.TB[i]  # grape is the the most __ in the least, unsprayed
                time_to_move_platform = move_to_next_pos(current_location, grape, 'sonar') # TODO Omer: check time to move platform
                print("temp location", temp_location)
                # input("Press enter for moving the robot back to temp location")
                move_command(False, temp_location, sleep_time)
                if time_to_move_platform:  # 15.1
                    continue  # move to 13, select next grape
                current_location, no_tech_problem = read_position()
                # distance, is_grape = activate_sonar()  # FIXME: Yossi  + Edo
                distance, is_grape = 680, True
                print("distance :", distance, "is_grape :", is_grape)
                if is_grape:  # 16 - yes
                    update_database_sonard(i, distance)  # 17
                    # if time_to_move_platform or g.TB[i].x_meter > (step_size / 2):  # 17. TODO: more complicated logic for later
                    #     continue  # move to 13, select next grape
                    print("Distance: ", distance)
                    time_to_move_platform = move_to_next_pos(current_location, grape, 'spray') # TODO Omer: check time to move platform
                    print("temp location", temp_location)
                    input("Press enter for move the robot back to temp location, before spraying")
                    move_command(False, temp_location, sleep_time)  # TODO Omer: moves to much to the right

                    # the or is for the case where in the next image the cluster will get closer to center
                    # spray()  # TODO: omer
                    print("current TB:")
                    update_database_sprayed(i)  # 18 # TODO Edo: show the image and mark grape sprayed
                else:  # 16 - no
                    update_database_no_grape(i)  # 19
        restart_TB()  # option to restart without initialize
