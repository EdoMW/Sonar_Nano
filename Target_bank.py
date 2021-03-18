import numpy as np
import math
from operator import itemgetter, attrgetter
import g_param
from math import cos, sin, pi,  radians
import cv2 as cv

# Define colors for printing
from sty import fg, Style, RgbFg
fg.orange = Style(RgbFg(255, 150, 50))
fg.red = Style(RgbFg(247, 31, 0))
fg.green = Style(RgbFg(31, 177, 31))
fg.yellow = Style(RgbFg(255, 255, 70))

######################################################
################# Parameters to tune #################
######################################################
"""
same_grape_distance_threshold: min distance to distinguish between to grapes
edge_distance_threshold: if distance from right edge of the grape to the edge of the image (when moving right),
                        don't add the grape to TB (it will get inside at the next iteration) 
"""
same_grape_distance_threshold = 0.07
edge_distance_threshold = 0.04


# prints the TB by grapes index
def print_by_id():
    print("TB by id: ", "\n")
    for i in range(len(g_param.TB)):
        print(g_param.TB[i].index, end=" ")

class Target_bank:
    grape_index = 0

    def target_to_string(self):
        ind = self.index
        ind = " ID : " + str(ind) + " "
        ind = fg.orange + ind + fg.rs
        a = self.x_meter
        b = self.y_meter
        c = self.x_p
        d = self.y_p
        x = " x:" + str(a) + " "
        y = " y:" + str(b) + " "
        x_c = " x_p:" + str(c) + " "
        y_c = " y_p:" + str(d)
        w = f" w: {self.w_meter}"
        h = f" h: {self.h_meter}"
        if self.sprayed:
            e = fg.green + str(self.sprayed) + fg.rs
        else:
            e = fg.red + str(self.sprayed) + fg.rs
        f = " area: " + str(self.rect_area) + " "
        sp = " sprayed :" + str(e) + " "
        wr = "wait_round: " + str(self.wait_another_step)
        world_data = " x world " + str(round(self.x_meter, 3)) + " y world " + str(round(self.y_meter, 3)) + " "
        x_base = "x base: " + str(round(self.grape_world[0], 3)) + " "
        y_base = "y base: " + str(round(self.grape_world[1], 3)) + " "
        z_base = "z base: " + str(round(self.grape_world[2], 3)) + " "
        base_data = x_base + y_base + z_base + '\n'
        angle = f" angle: {self.angle}"
        return ind + sp + wr + world_data + base_data + w + h + angle + '\n'
        # return ind + sp + wr + x_c + y_c + world_data + base_data + w + h + angle + '\n'

    def __repr__(self):
        params = self.target_to_string()
        corners = f"corner_1: {self.corners[0]} corner_2: {self.corners[1]} " \
                  f"corner_3: {self.corners[2]} corner_4: {self.corners[3]}"
        return params + corners + '\n' + '\n'
        # return ind + x + y + x_c + y_c + f + sp + world_data + base_data

    def calc_corners_by_gemotry(self):
        """
        Doesn't work when extra process is done to the image.
        :return:
        """
        x_cen, y_cen = self.x_meter, self.y_meter
        w, h = self.w_meter, self.h_meter
        angle = self.angle
        ninety = pi/2
        alpha = radians(angle)
        beta = ninety - radians(angle)
        # beta = radians(angle)
        # alpha = ninety - radians(angle)
        w = w/2
        h = h/2
        cor_1 = [round(x_cen + (-cos(beta) * h + cos(alpha)*w), 3), round(y_cen + (-sin(beta)*h - w*sin(alpha)), 3)]
        cor_2 = [round(x_cen + (cos(beta) * h + cos(alpha) * w), 3), round(y_cen + (sin(beta) * h - w * sin(alpha)), 3)]
        cor_3 = [round(x_cen + (cos(beta) * h - cos(alpha) * w), 3), round(y_cen + (sin(beta) * h + w * sin(alpha)), 3)]
        cor_4 = [round(x_cen + (-cos(beta) * h - cos(alpha) * w), 3), round(y_cen + (-sin(beta)*h + w * sin(alpha)), 3)]
        # auto_corners = cv.boxPoints(box)
        return [cor_1, cor_2, cor_3, cor_4]

    def __init__(self, x, y, w, h, angle, mask, pixels_data, grape_world, corners):
        """
        :param x: x center coordinate in meters
        :param y: y center coordinate in meters
        :param w: width in meters
        :param h: height in meters
        :param angle: in degrees from the horizontal line
        :param mask: a 2D bitmap of the grape.
        :param pixels_data: all data in pixels
        :param grape_world: data in meter #TODO- CHECK if it has a use.
        :param corners: list of lists. [[x,y],..,[x,y]]. corners[0] is the the lowest one, then it goes clockwise.

        """
        self.index = Target_bank.grape_index
        self.grape_world = grape_world
        # self.y_base = str(round(self.grape_world[1], 3))
        self.x_p = int(pixels_data[0])
        self.y_p = int(pixels_data[1])
        self.w_p = int(pixels_data[2])
        self.h_p = int(pixels_data[3])
        self.x_meter = round(x, 3)  # TODO: check how to update this 4 parameters as world
        self.y_meter = round(y, 3)
        self.w_meter = round(w, 3)
        self.h_meter = round(h, 3)
        self.dist_from_center = Target_bank.calc_dist_from_center(self.x_p, self.y_p)
        self.angle = angle
        self.rect_area = self.w_p * self.h_p
        self.sprayed = False
        self.mask = mask
        self.distance = 0.71  # 0:default distance value, 1:from sonar
        self.fake_grape = False
        self.wait_another_step = False
        self.corners = simlifay_corners(corners)
        # amount of updates, what iteration was the last update

    def calc_dist_from_center(x, y):
        return math.sqrt(x * x + y * y)


# if distance between centers is smaller than the treshhold
def check_if_in_TB(grape_world, target):
    """
    :param grape_world: The grape coordinates in world parameters
    :param target: The grape coordinates in pixels
    :return: True,the updated pixel values for already in the TB grape.
             False, None- the grapes does not exist in TB. it will be added.
    """
    if len(g_param.TB) > 0:
        for i in range(len(g_param.TB)):  # TODO: make it only for possible grapes in reach of the image
            point_b = g_param.TB[i].grape_world
            # print("grape_world", grape_world)
            # print("distance : ", np.linalg.norm(grape_world - point_b))
            distance_between_grapes = np.linalg.norm(grape_world - point_b)
            if distance_between_grapes < same_grape_distance_threshold:

                # print("distance between old and new cluster", distance_between_grapes)
                g_param.TB[i].x_p = int(target[6][0])
                g_param.TB[i].y_p = int(target[6][1])
                g_param.TB[i].w_p = int(target[6][2])
                g_param.TB[i].h_p = int(target[6][3])
                g_param.TB[i].x_meter = target[0]
                g_param.TB[i].y_meter = target[1]
                g_param.TB[i].w_meter = target[2]
                g_param.TB[i].h_meter = target[3]
                # decide if to update world
                return True, i
    return False, None


def simlifay_corners(corners):
    """
    convert corners from np.array to list, round to 3 decimel points.
    :param corners: corners list of np.arrays
    :return: corner list in meters
    """
    corners_simple = []
    for i in range(4):
        corner_list = corners[i][:2]
        corn = corner_list.tolist()
        corn = [round(corn[0], 3), round(corn[1], 3)]
        corners_simple.append(corn)
    return corners_simple


# add a new detected target to the TB
# TODO: get as input current_location and use it to calc x,y locaiions
def check_close_to_edge(target):
    """
    checks if target (grape) is too close to one of the edges:
    upper, lower or the one on the right (assuming moving only to the right).
    :param target: grape cluster
    :return: True if the grape is too close to the right edge
    """
    x_m = target[0]
    y_m = target[1]
    w_m = target[2] / 2
    h_m = target[3] / 2
    angle = target[4]
    right = check_close_to_right_edge(x_m, w_m, h_m, angle)
    lower = check_close_to_lower_edge(y_m, w_m, h_m, angle)
    upper = check_close_to_upper_edge(y_m, w_m, h_m, angle)
    return True if True in [right, lower, upper] else False  # True if at least one of them is True


def check_close_to_right_edge(x_m, w_m, h_m, angle):
    """
    :param x_m: x center coordinate of the grape
    :param w_m: width of the Bounding box
    :param h_m: height of the Bounding box
    :param angle: angle of rotation of the bounding box
    :return: True if too close to the right edge
    """
    dist_on_x_from_center_1 = w_m * math.cos(math.radians(angle))
    dist_on_x_from_center_2 = h_m * math.cos(math.radians(angle))
    dist_to_edge_1 = abs(g_param.half_width_meter - (x_m + dist_on_x_from_center_1))
    dist_to_edge_2 = abs(g_param.half_width_meter - (x_m + dist_on_x_from_center_2))
    return dist_to_edge_1 < edge_distance_threshold or dist_to_edge_2 < edge_distance_threshold


def check_close_to_lower_edge(y_m, w_m, h_m, angle):
    """
    :param y_m: x center coordinate of the grape
    :param w_m: width of the Bounding box
    :param h_m: height of the Bounding box
    :param angle: angle of rotation of the bounding box
    :return: True if too close to the lower edge
    """
    dist_on_y_from_center_1 = w_m * math.sin(math.radians(angle))
    dist_on_y_from_center_2 = h_m * math.sin(math.radians(angle))
    dist_to_edge_1 = abs(g_param.half_height_meter - (y_m + dist_on_y_from_center_1))
    dist_to_edge_2 = abs(g_param.half_height_meter - (y_m + dist_on_y_from_center_2))
    return dist_to_edge_1 < edge_distance_threshold or dist_to_edge_2 < edge_distance_threshold


def check_close_to_upper_edge(y_m, w_m, h_m, angle):
    """
    :param y_m: x center coordinate of the grape
    :param w_m: width of the Bounding box
    :param h_m: height of the Bounding box
    :param angle: angle of rotation of the bounding box
    :return: True if too close to the lower edge
    """
    dist_on_y_from_center_1 = w_m * math.sin(math.radians(angle))
    dist_on_y_from_center_2 = h_m * math.sin(math.radians(angle))
    dist_to_edge_1 = abs(g_param.half_height_meter + (y_m + dist_on_y_from_center_1))
    dist_to_edge_2 = abs(g_param.half_height_meter + (y_m + dist_on_y_from_center_2))
    return dist_to_edge_1 < edge_distance_threshold or dist_to_edge_2 < edge_distance_threshold


def round_to_three(arr):
    for i in range(0,4):
        arr[i] = round(arr[i], 3)
    return arr


def add_to_TB(target):
    target = round_to_three(target)
    too_close = check_close_to_edge(target)  # FIXME Edo
    temp_grape_world = g_param.trans.grape_world(target[0], target[1])
    ans, temp_target_index = check_if_in_TB(temp_grape_world, target)
    if ans:
        # print("the grape already in TB")
        # print("true") # TODO check the end condition
        closer_to_center = g_param.TB[temp_target_index].dist_from_center < Target_bank.calc_dist_from_center(target[0],
                                                                                                              target[1])
        if closer_to_center or too_close:  # not sprayed and closer to center
            g_param.TB[temp_target_index].grape_world = temp_grape_world


        # if not temp_target.sprayed:
        #     print("why update?: ", g.TB)
        #     g.TB.append(Target_bank(target[0], target[1], target[2], target[3], target[4], target[5], target[6]))
        #     print("updated :", g.TB)
    else:

        if not too_close:
            # print("the grape not in TB yet")
            g_param.TB.append(Target_bank(target[0], target[1], target[2], target[3], target[4],
                                          target[5], target[6], temp_grape_world, target[8]))
            Target_bank.grape_index += 1
        # print("not in TB yet but too close to edge")


def sort_by_and_check_for_grapes(sorting_type):
    """
    Sort the array and return if there are targets to spray_procedure
    :param sorting_type: How to sort the grapes
    :return: False if the TB is empty or there are no more grapes to spray_procedure
             True otherwise.
    """
    if sorting_type == 'rect_size':
        sort_by_rect_size()
    if sorting_type == 'dist_from_current_pos':
        sort_by_dist_from_current_pos()
    if sorting_type == 'mask_size':
        sort_by_mask_size()
    if sorting_type == 'leftest_first':
        sort_by_leftest_first()
    if len(g_param.TB) > 0:
        if g_param.TB[0].sprayed:
            return False
        else:
            return True
    return False


def sort_by_leftest_first():
    g_param.TB = sorted(g_param.TB, key=attrgetter('sprayed', 'x_meter'))
    # g_param.TB = sorted(g_param.TB, key=attrgetter('sprayed', 'y_base'))


def sort_by_rect_size():
    g_param.TB = sorted(g_param.TB, key=attrgetter('sprayed', 'rect_area'))


def sort_by_dist_from_current_pos():
    pass


def sort_by_mask_size():
    pass