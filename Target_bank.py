import numpy as np
np.set_printoptions(precision=3)
import math
from operator import itemgetter, attrgetter
import g_param
from math import cos, sin, pi, radians
import scipy
import cv2 as cv
from transform import rotation_coordinate_sys
# Define colors for printing
from sty import fg, Style, RgbFg
fg.orange = Style(RgbFg(255, 150, 50))
fg.red = Style(RgbFg(247, 31, 0))
fg.green = Style(RgbFg(31, 177, 31))
fg.yellow = Style(RgbFg(255, 255, 70))

# ----------------------------------------------------------------
# ---------------------- Parameters to tune ----------------------
# ----------------------------------------------------------------

"""
same_grape_distance_threshold: min distance to distinguish between to grapes
edge_distance_threshold: if distance from right edge of the grape to the edge of the image (when moving right),
                        don't add the grape to TB (it will get inside at the next iteration) 
"""
same_grape_distance_threshold = 0.09
edge_distance_threshold = 0.01


# prints the TB by grapes index
def print_by_id():
    print("TB by id: ", "\n")
    for i in range(len(g_param.TB)):
        print(g_param.TB[i].index, end=" ")


class Target_bank:

    def __init__(self, x, y, w, h, angle, mask, pixels_data, grape_world,
                 corners, p_corners, grape_base, pixels_count, com, mask_score):
        """
        13 variables (for now)
        :param x: x center coordinate in meters
        :param y: y center coordinate in meters
        :param w: width in meters
        :param h: height in meters
        :param angle: in degrees from the horizontal line
        :param mask: a 2D bitmap of the grape.
        :param pixels_data: all data in pixels
        :param grape_world: data in meter #
        :param corners: list of lists. [[x,y],..,[x,y]]. corners[0] is the the lowest one, then it goes clockwise.

        """
        self.index = Target_bank.grape_index
        self.grape_world = grape_world
        self.grape_base = grape_base
        self.last_updated = g_param.image_number
        self.x_p = int(pixels_data[0])  # p are in pixels. 0,0 is the center of the image.
        self.y_p = int(pixels_data[1])
        self.w_p = int(pixels_data[2])
        self.h_p = int(pixels_data[3])
        self.x_center = round(x, 3)  # base are in meters, relative to the center of the image
        self.y_center = round(y, 3)
        self.w_meter = round(w, 3)
        self.h_meter = round(h, 3)
        self.dist_from_center = self.calc_dist_from_center()  # in pixels
        self.angle = angle
        self.rect_area = self.w_p * self.h_p  # in pixels
        self.sprayed = False
        self.mask = mask
        self.mask_score = mask_score
        self.center_of_mass = com
        self.pixels_count = pixels_count  # in pixels
        self.distance = 0.75  # 0:default distance value, 1:from sonar
        self.fake_grape = False
        self.in_range = "ok"
        self.wait_another_step = False
        self.p_corners = p_corners
        self.corners = simplify_corners(corners)
        # amount of updates, what iteration was the last update

    grape_index = 0

    def target_to_string(self):
        ind = self.index
        ind = " ID : " + str(ind) + " "
        ind = fg.orange + ind + fg.rs
        a = self.x_center
        b = self.y_center
        c = self.x_p
        d = self.y_p
        fake_grape = " fake grape: " + str(self.fake_grape) + " "
        dist_from_center = " dist from center in pixels:" + str(round(self.dist_from_center, 1))
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
        # wr = "wait_round: " + str(self.wait_another_step)
        base = " x base " + str(round(self.x_center, 3)) + " y base " + str(round(self.y_center, 3))
        x_world = "x world: " + str(round(self.grape_world[0], 3)) + " "
        y_world = "y world: " + str(round(self.grape_world[1], 3)) + " "
        z_world = "z world: " + str(round(self.grape_world[2], 3)) + " "
        world_data = x_world + y_world + z_world + '\n'
        angle = f" angle: {self.angle}"
        distance = f'distance: {self.distance}'
        return ind + sp + fake_grape + world_data + base + w + h + angle + '\n'\
               + distance + dist_from_center + '\n'
        # return ind + sp + wr + x_c + y_c + world_data + base_data + w + h + angle + '\n'

    def __repr__(self):
        params = self.target_to_string()
        corners = f"corner_1: {self.corners[0]} corner_2: {self.corners[1]} " \
                  f"corner_3: {self.corners[2]} corner_4: {self.corners[3]}"
        return params + corners + '\n'
        # return ind + x + y + x_c + y_c + f + sp + world_data + base_data

    def calc_corners_by_gemotry(self):
        """
        Doesn't work when extra process is done to the image.
        :return:
        """
        x_cen, y_cen = self.x_center, self.y_center
        w, h = self.w_meter, self.h_meter
        angle = self.angle
        ninety = pi / 2
        alpha = radians(angle)
        beta = ninety - radians(angle)
        # beta = radians(angle)
        # alpha = ninety - radians(angle)
        w = w / 2
        h = h / 2
        cor_1 = [round(x_cen + (-cos(beta) * h + cos(alpha) * w), 3),
                 round(y_cen + (-sin(beta) * h - w * sin(alpha)), 3)]
        cor_2 = [round(x_cen + (cos(beta) * h + cos(alpha) * w), 3), round(y_cen + (sin(beta) * h - w * sin(alpha)), 3)]
        cor_3 = [round(x_cen + (cos(beta) * h - cos(alpha) * w), 3), round(y_cen + (sin(beta) * h + w * sin(alpha)), 3)]
        cor_4 = [round(x_cen + (-cos(beta) * h - cos(alpha) * w), 3),
                 round(y_cen + (-sin(beta) * h + w * sin(alpha)), 3)]
        # auto_corners = cv.boxPoints(box)
        return [cor_1, cor_2, cor_3, cor_4]

    def calc_center_of_mass(self):
        if self.mask is not None:
            return scipy.ndimage.measurements.center_of_mass(self.mask)
        return None

    def calc_dist_from_center(self):
        if type(self) is list:
            x = 512 - self[0]
            y = 512 - self[1]
        else:
            x = 512 - self.x_p
            y = 512 - self.y_p
        return round(math.sqrt((x * x) + (y * y)), 2)


def calc_z_move():
    a = g_param.image_number % 4
    if a == 0 or a == 3:
        return 0
    return g_param.height_step_size * g_param.step_size


def update_grape_center(index):
    """
    To update camera coordinates of each grape cluster center points in meters, relative to the center of the image
    in the TB after end effector movement (next capturing position).
    y of base is equal to camera x (up to -/+)
    z of base is equal to camera y (up to -/+)
    :param index: index of the grape in TB
    """

    if g_param.image_number > 0:
        if g_param.TB[index].last_updated < g_param.image_number:
            capture_pos_r = rotation_coordinate_sys(g_param.trans.capture_pos[0],
                                                    g_param.trans.capture_pos[1], -g_param.base_rotation_ang)[1]
            prev_capture_pos_r = rotation_coordinate_sys(g_param.trans.prev_capture_pos[0],
                                                         g_param.trans.prev_capture_pos[1],
                                                         -g_param.base_rotation_ang)[1]
            delta_x = capture_pos_r-prev_capture_pos_r
            # print("????????????????",delta_x)
            delta_y = g_param.trans.capture_pos[2] - g_param.trans.prev_capture_pos[2]
            # print("!!!!!!!!!!!!!!!!", delta_y)
            g_param.TB[index].x_center += delta_x
            g_param.TB[index].y_center += delta_y
        # if g_param.image_number > 0:
        #     if (math.floor(g_param.image_number / 2) - math.floor((g_param.image_number - 1) / 2)) > 0:
        #         g_param.TB[index].x_center += (math.floor(g_param.image_number / 2)
        #                                        - math.floor((g_param.image_number - 1) / 2)) * g_param.step_size
        # g_param.TB[index].y_center += calc_z_move()


# if distance between centers is smaller than the treshhold
def check_if_in_TB(grape_world, target):
    """
    :param grape_world: The grape coordinates in world parameters
    :param target: The grape coordinates in pixels
    :return: True,the updated pixel values for already in the TB grape.
             False, None- the grapes does not exist in TB. it will be added.
    """
    if len(g_param.TB) > 0:
        for i in range(len(g_param.TB)):  # TODO (after exp): make it only for possible grapes in reach of the image
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
                g_param.TB[i].x_center = target[0]
                g_param.TB[i].y_center = target[1]
                g_param.TB[i].w_meter = target[2]
                g_param.TB[i].h_meter = target[3]
                g_param.TB[i].last_updated = g_param.image_number
                # decide if to update world
                return True, i
    return False, None


def simplify_corners(corners):
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
    # x_m = target[0]
    y_m = target[1]
    w_m = target[2] / 2
    h_m = target[3] / 2
    angle = target[4]
    right = check_close_to_right_edge(target[8])
    if right:
        g_param.masks_image = cv.circle(g_param.masks_image, (target[6][0],  target[6][1]),
                                        radius=2, color=(0, 0, 255), thickness=2)
    lower = check_close_to_lower_edge(y_m, w_m, h_m, angle, target[8])
    upper = check_close_to_upper_edge(target[8])
    # print(f'right: {right}, up: {upper}, down: {lower}')
    return True if True in [right, lower, upper] else False  # True if at least one of them is True


def check_close_to_right_edge(corners_in_m):
    """
    :param corners_in_m:
    :return: True if too close to the right edge
    """
    edt = edge_distance_threshold
    p1, p2, p3, p4 = corners_in_m[0][0], corners_in_m[1][0], corners_in_m[2][0], corners_in_m[3][0]
    dist_to_edge_1 = g_param.half_width_meter - p1
    dist_to_edge_2 = g_param.half_width_meter - p2
    dist_to_edge_3 = g_param.half_width_meter - p3
    dist_to_edge_4 = g_param.half_width_meter - p4
    # print("Right edge: ", dist_to_edge_1," ", dist_to_edge_2," ", dist_to_edge_3," ", dist_to_edge_4)
    if dist_to_edge_1 < edt or dist_to_edge_2 < edt or dist_to_edge_3 < edt or dist_to_edge_4 < edt:
        print("Right to close too edge")
    return dist_to_edge_1 < edt or dist_to_edge_2 < edt or dist_to_edge_3 < edt or dist_to_edge_4 < edt


#  FIXME: not working.
def check_close_to_lower_edge(y_m, w_m, h_m, angle, corners_in_m):
    """
    :param corners_in_m:
    :param y_m: x center coordinate of the grape
    :param w_m: width of the Bounding box
    :param h_m: height of the Bounding box
    :param angle: angle of rotation of the bounding box
    :return: True if too close to the lower edge
    """
    edt = edge_distance_threshold
    p1 = corners_in_m[0][1]
    dist_to_lower_edge = g_param.half_height_meter - p1
    if dist_to_lower_edge < edt:
        print("Lower edge too close to edge", dist_to_lower_edge, "p1", p1)
        return dist_to_lower_edge < edt
    return False
    # if g_param.direction == "down" or g_param.direction == "right":
    #     dist_on_y_from_center_1 = w_m * math.sin(math.radians(angle))
    #     dist_on_y_from_center_2 = h_m * math.sin(math.radians(angle))
    #     dist_to_edge_1 = abs(g_param.half_height_meter - (y_m + dist_on_y_from_center_1))
    #     dist_to_edge_2 = abs(g_param.half_height_meter - (y_m + dist_on_y_from_center_2))
    #     if dist_to_edge_1 < edge_distance_threshold or dist_to_edge_2 < edge_distance_threshold:
    #         print("Lower to close too edge")
    #     return dist_to_edge_1 < edge_distance_threshold or dist_to_edge_2 < edge_distance_threshold
    # return False


def check_close_to_upper_edge(corners_in_m):
    """
    :return: True if too close to the lower edge
    """
    edt = edge_distance_threshold
    p3 = corners_in_m[2][1]
    dist_to_upper_edge = g_param.half_height_meter + p3
    if dist_to_upper_edge < edt:
        print("Upper edge too close", p3)
        return dist_to_upper_edge < edt
    return False
    # dist_on_y_from_center_1 = w_m * math.sin(math.radians(angle))
    # dist_on_y_from_center_2 = h_m * math.sin(math.radians(angle))
    # dist_to_edge_1 = abs(g_param.half_height_meter + (y_m + dist_on_y_from_center_1))
    # dist_to_edge_2 = abs(g_param.half_height_meter + (y_m + dist_on_y_from_center_2))
    # if dist_to_edge_1 < edge_distance_threshold or dist_to_edge_2 < edge_distance_threshold:
    #     print("Upper to close too edge")
    # return dist_to_edge_1 < edge_distance_threshold or dist_to_edge_2 < edge_distance_threshold


def round_to_three(arr):
    for i in range(0, 4):
        arr[i] = round(arr[i], 3)
    return arr


def add_to_target_bank(target):
    """
    Checks if a new target (grape) already exits.
    compares the center of the new target to all exiting targets in the last 1m of the Y axis (the one the platform goes
    along) TO BE DONE!
    if the center of the new target is lower than the threshold to one of the exiting targets, treat it as the same one.
    if the target center in this image is closer to the center, update it's world coordinates.
    else- add new target to the target bank.
    :param target:
    :return:
    """
    target = round_to_three(target)
    too_close = check_close_to_edge(target)
    temp_grape_world = g_param.trans.grape_world(target[0], target[1])
    grape_base = g_param.trans.grape_base(target[0], target[1])
    grape_in_TB, temp_target_index = check_if_in_TB(temp_grape_world, target)
    if g_param.eval_mode:
        pass
    if grape_in_TB:
        pass
        # closer_to_center = g_param.TB[temp_target_index].dist_from_center < Target_bank.calc_dist_from_center(target)
        # if closer_to_center or too_close:  # not sprayed and closer to center
        #     g_param.TB[temp_target_index].grape_world = temp_grape_world
        #     g_param.TB[temp_target_index].grape_base = grape_base
    else:
        if not too_close:
            # print("the grape not in TB yet")
            g_param.TB.append(Target_bank(target[0], target[1], target[2], target[3], target[4],
                                          target[5], target[6], temp_grape_world, target[8], target[9],
                                          grape_base, target[10], target[11]))
            if g_param.work_place == 'field':
                g_param.read_write_object.save_mask(target[5], Target_bank.grape_index)
                mask_path = f'npzs/{g_param.image_number}_{Target_bank.grape_index}.npz'
                np.savez_compressed(mask_path, target[5])
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
    g_param.TB = sorted(g_param.TB, key=attrgetter('sprayed', 'x_center'), reverse=False)


def sort_by_rect_size():
    g_param.TB = sorted(g_param.TB, key=attrgetter('sprayed', 'rect_area'))


def sort_by_dist_from_current_pos():
    pass


def sort_by_mask_size():
    pass


def point_pixels_2_meter(d, point):
    """
    :param d: distance to grape
    :param point: point to convert to meter
    :return: [x, y] in meters relative to the center
    """
    cen_poi_x_0 = int(point[0])
    cen_poi_y_0 = int(point[1])
    cen_poi_x_0 = cen_poi_x_0 - int(1024 / 2)
    cen_poi_y_0 = cen_poi_y_0 - int(1024 / 2)
    x_point = d * (cen_poi_x_0 / 1024) * (7.11 / 8)
    y_point = d * (cen_poi_y_0 / 1024) * (5.33 / 8) * 1.33
    return [x_point, y_point]


def box_points_to_np_array(d, corner):
    p1 = point_pixels_2_meter(d, corner)
    p1 = np.array(p1)
    p1 = np.insert(arr=p1, obj=2, values=1, axis=0)
    return p1


def calculate_w_h(d, box_points):
    """
    calculates the width and height of the box.
    I used the same method as the cv.minAreaRect way of calculating H,W
    :param d: distance to grape
    :param box_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :return: w,h in meters
    """
    p1 = box_points_to_np_array(d, box_points[0])
    p2 = box_points_to_np_array(d, box_points[1])
    p3 = box_points_to_np_array(d, box_points[2])
    p4 = box_points_to_np_array(d, box_points[3])
    w = round(np.linalg.norm(p1 - p2), 3)
    h = round(np.linalg.norm(p2 - p3), 3)
    if w > h:
        h, w = w, h
        p1, p2, p3, p4 = p1, p4, p3, p2
    return w, h, [p1, p2, p3, p4]


def update_by_real_distance(grape):
    """
    call the function ONLY if Sonar was activated
    :param ind: index of the grape
    """
    # TODO (after exp): call the function that updates g_param.avg_dist with sonar_location
    grape.distance = g_param.last_grape_dist + g_param.sonar_x_length
    x, y = point_pixels_2_meter(grape.distance, [grape.x_p, grape.y_p])
    w, h, corners = calculate_w_h(grape.distance, grape.p_corners)
    grape.x_center, grape.y_center, grape.w_meter, grape.h_meter, grape.corners = x, y, w, h, corners
    grape.grape_world = g_param.trans.grape_world(x, y)
    grape.grape_base = g_param.trans.grape_base(x, y)
