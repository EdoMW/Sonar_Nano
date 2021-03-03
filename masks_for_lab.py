from __future__ import print_function
import cv2 as cv
import argparse
import random as rng
import matplotlib.pyplot as plt
from termcolor import colored

from Target_bank import check_if_in_TB, add_to_TB, sort_by_and_check_for_grapes, sort_by_rect_size
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from numpy import *
import math
from math import pi, cos, sin
import sys
from pyueye import ueye
import numpy as np
import g_param
import time
import os

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized





def showInMovedWindow(winname, img, x = 0, y = 0):
    cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)        # Create a named window
    cv.moveWindow(winname, x, y)   # Move it to (x,y)
    # cv.resizeWindow(winname, 400, 512)
    # img = cv.resize(img, (800, 1024))
    cv.imshow(winname,img)


def masks_to_convex_hulls(list_of_masks):
    list_of_npas = []
    list_of_CH = []
    if len(list_of_masks) > 0:
        for i in range(0, len(list_of_masks)):
            if len(list_of_masks[i]) > 0:
                ch = mask_to_convex_hull(list_of_masks[i])[0]
                npa = mask_to_convex_hull(list_of_masks[i])[1]
                list_of_CH.append(ch)
                list_of_npas.append(npa)
    return list_of_CH, list_of_npas


def mask_to_convex_hull(mask):
    arr_to_list = []
    for i in range(0, 1024):  # change it that it won't check
        for j in range(0, 1024):  # the padded by zeros area
            if mask[i, j] == 1:
                point = [j, i]
                arr_to_list.append(point)
    hull = ConvexHull(arr_to_list)
    # [2]
    npa = np.asarray(arr_to_list, dtype=np.float32)
    # [3]
    hull = ConvexHull(npa)
    return hull, npa


# function to change from float vec to int vec
def two_value_to_int_vec(vec):
    corner_1 = vec
    corner_1_1 = int(corner_1[0])
    corner_1_2 = int(corner_1[1])
    corner_1_new = list()
    corner_1_new.append(corner_1_1)
    corner_1_new.append(corner_1_2)
    return corner_1_new


def minBoundingRect(hull_points_2d):
    # print "Input convex hull points: "
    # print hull_points_2d

    # Compute edges (x2-x1,y2-y1)
    edges = zeros((len(hull_points_2d) - 1, 2))  # empty 2 column array
    for i in range(len(edges)):
        edge_x = hull_points_2d[i + 1, 0] - hull_points_2d[i, 0]
        edge_y = hull_points_2d[i + 1, 1] - hull_points_2d[i, 1]
        edges[i] = [edge_x, edge_y]
    # print "Edges: \n", edges
    # Calculate edge angles   atan2(y/x)
    edge_angles = zeros((len(edges)))  # empty 1 column array
    for i in range(len(edge_angles)):
        edge_angles[i] = math.atan2(edges[i, 1], edges[i, 0])
    # print "Edge angles: \n", edge_angles

    # Check for angles in 1st quadrant
    for i in range(len(edge_angles)):
        edge_angles[i] = abs(edge_angles[i] % (math.pi / 2))  # want strictly positive answers
    # print "Edge angles in 1st Quadrant: \n", edge_angles

    # Remove duplicate angles
    edge_angles = unique(edge_angles)
    # print "Unique edge angles: \n", edge_angles

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, sys.maxsize, 0, 0, 0, 0, 0, 0)  # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range(len(edge_angles)):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = array([[math.cos(edge_angles[i]), math.cos(edge_angles[i] - (math.pi / 2))],
                   [math.cos(edge_angles[i] + (math.pi / 2)), math.cos(edge_angles[i])]])
        # print "Rotation matrix for ", edge_angles[i], " is \n", R

        # Apply this rotation to convex hull points
        rot_points = dot(R, transpose(hull_points_2d))  # 2x2 * 2xn
        # print "Rotated hull points are \n", rot_points

        # Find min/max x,y points
        min_x = nanmin(rot_points[0], axis=0)
        max_x = nanmax(rot_points[0], axis=0)
        min_y = nanmin(rot_points[1], axis=0)
        max_y = nanmax(rot_points[1], axis=0)
        # print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        # print "Potential bounding box ", i, ":  width: ", width, " height: ", height, "  area: ", area

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = (edge_angles[i], area, width, height, min_x, max_x, min_y, max_y)
        # Bypass, return the last found rect
        # min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]
    R = array([[math.cos(angle), math.cos(angle - (math.pi / 2))], [math.cos(angle + (math.pi / 2)), math.cos(angle)]])
    # print "Projection matrix: \n", R

    # Project convex hull points onto rotated frame
    proj_points = dot(R, transpose(hull_points_2d))  # 2x2 * 2xn
    # print "Project hull points are \n", proj_points

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]
    # print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_point = dot([center_x, center_y], R)
    # print "Bounding box center point: \n", center_point

    # Calculate corner points and project onto rotated frame
    corner_points = zeros((4, 2))  # empty 2 column array
    corner_points[0] = dot([max_x, min_y], R)
    corner_points[1] = dot([min_x, min_y], R)
    corner_points[2] = dot([min_x, max_y], R)
    corner_points[3] = dot([max_x, max_y], R)
    # print "Bounding box corner points: \n", corner_points

    # print "Angle of rotation: ", angle, "rad  ", angle * (180/math.pi), "deg"

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point,
            corner_points)  # rot_angle, area, width, height, center_point, corner_points


'''
mrbbs (minimum rodated bounding box s ) is a list of dicitnoray 
that contains the next parameters:
rot_angle, area, width, height, center_point, corner_points
'''


def output_dict(npas):
    mrbbs = []
    for i in range(0, len(npas)):
        mrbb = minBoundingRect(npas[i])
        center_point = two_value_to_int_vec(mrbb[4])
        first_corner = two_value_to_int_vec(mrbb[5][0])
        second_corner = two_value_to_int_vec(mrbb[5][1])
        third_corner = two_value_to_int_vec(mrbb[5][2])
        forth_corner = two_value_to_int_vec(mrbb[5][3])
        corners = []
        corners.append(first_corner)
        corners.append(second_corner)
        corners.append(third_corner)
        corners.append(forth_corner)
        fixed_angle = (mrbb[0] * 180) / pi
        mmrns_dic = {"rot_angle": fixed_angle, "area": mrbb[1],
                     "width": mrbb[2], "height": mrbb[3],
                     "center_point": center_point,
                     "corner_points": corners}
        mrbbs.append(mmrns_dic)
    return mrbbs


# computing intersaction between rotated bounding boxes


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x * v.y - self.y * v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a * p.x + self.b * p.y + self.c

    def intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a * other.b - self.b * other.a
        return Vector(
            (self.b * other.c - self.c * other.b) / w,
            (self.c * other.a - self.a * other.c) / w
        )


def rectangle_vertices(cx, cy, w, h, r):
    angle = pi * r / 180
    dx = w / 2
    dy = h / 2
    dxcos = dx * cos(angle)
    dxsin = dx * sin(angle)
    dycos = dy * cos(angle)
    dysin = dy * sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - -dysin, dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - dysin, dxsin + dycos),
        Vector(cx, cy) + Vector(-dxcos - dysin, -dxsin + dycos)
    )


def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
                intersection, intersection[1:] + intersection[:1],
                line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x * q.y - p.y * q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))


# GT_rotated_boxes
# det_rotated_boxes
# create new compute_overlaps to match rotated Bounding boxes
class calcs():

    def calc_angle_between_boxes(matched_gt_boxes, positive_ids,
                                 GT_rotated_boxes, det_rotated_boxes):
        angles = []
        for i in range(0, len(matched_gt_boxes)):
            a = matched_gt_boxes[i]
            b = positive_ids[i]
            det_angle = det_rotated_boxes[a][4]
            GT_angle = GT_rotated_boxes[b][4]
            angle = abs(det_angle - GT_angle)
            angles.append(angle)
        return angles

    def mean_of_list(list_to_calc):
        if len(list_to_calc) < 1:
            return None
        sum = 0
        for i in range(0, len(list_to_calc)):
            sum += list_to_calc[i]
        x_avg = sum / len(list_to_calc)
        return x_avg

    def sd_of_list(list_to_calc):
        if len(list_to_calc) < 2:
            return 0
        sum = 0
        sum_of_squers = 0
        for i in range(0, len(list_to_calc)):
            sum += list_to_calc[i]
        x_avg = sum / len(list_to_calc)
        for i in range(0, len(list_to_calc)):
            sum_of_squers += ((list_to_calc[i] - x_avg) * (list_to_calc[i] - x_avg))
        sigma = sqrt((1 / (len(list_to_calc) - 1)) * sum_of_squers)
        return sigma


###### TO DO: check hit for second biggest mask and rotated box
# Libraries


def ueye_take_picture_2(image_number):
    # Variables
    hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
    sInfo = ueye.SENSORINFO()
    cInfo = ueye.CAMINFO()
    pcImageMemory = ueye.c_mem_p()
    MemID = ueye.int()
    rectAOI = ueye.IS_RECT()
    pitch = ueye.INT()
    nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
    channels = 3  # 3: channels for color mode(RGB); take 1 channel for monochrome
    m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
    bytes_per_pixel = int(nBitsPerPixel / 8)

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # print("START")
    # print()

    # Starts the driver and establishes the connection to the camera
    nRet = ueye.is_InitCamera(hCam, None)
    # if nRet != ueye.IS_SUCCESS:
    #     print("is_InitCamera ERROR")

    # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    # if nRet != ueye.IS_SUCCESS:
    #     print("is_GetCameraInfo ERROR")

    # You can query additional information about the sensor type used in the camera
    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    # if nRet != ueye.IS_SUCCESS:
    #     print("is_GetSensorInfo ERROR")

    # Set display mode to DIB
    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    # nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_OPENGL)

    # Set the right color mode
    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        # print("IS_COLORMODE_BAYER: ", )
        # print("\tm_nColorMode: \t\t", m_nColorMode)
        # print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        # print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        # print()

    # Can be used to set the size and position of an "area of interest"(AOI) within an image
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    # if nRet != ueye.IS_SUCCESS:
    #     print("is_AOI ERROR")

    width = rectAOI.s32Width
    height = rectAOI.s32Height

    # Prints out some information about the camera and the sensor
    # print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
    # print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
    # print("Maximum image width:\t", width)
    # print("Maximum image height:\t", height)
    # print()

    # ---------------------------------------------------------------------------------------------------------------------------------------

    # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
    nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_AllocImageMem ERROR")
    else:
        # Makes the specified image memory the active memory
        nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetImageMem ERROR")
        else:
            # Set the desired color mode
            nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

    # Activates the camera's live video mode (free run mode)
    nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")

    # nRet = ueye.is_FreezeVideo(hCam, ueye.IS_DONT_WAIT)
    # if nRet != ueye.IS_SUCCESS:
    #    print("is_CaptureVideo ERROR")

    # Enables the queue mode for existing image memory sequences
    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
    # if nRet != ueye.IS_SUCCESS:
    #     print("is_InquireImageMem ERROR")
    # else:
    #     print("Press q to leave the programm")

    # ---------------------------------------------------------------------------------------------------------------------------------------

    # Continuous image display
    if (nRet == ueye.IS_SUCCESS):

        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        pic_array_1 = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        time.sleep(1.1)
        pic_array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

        # ...reshape it in an numpy array...
        frame = np.reshape(pic_array, (height.value, width.value, bytes_per_pixel))
        frame = np.pad(frame, pad_width=[(506, 506), (0, 0), (0, 0)], mode='constant') # pad with zeros above and under
        # ...resize the image by a half
        frame = cv.resize(frame, (0, 0), fx=0.331606, fy=0.331606)

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # ...and finally display it
        # cv.imshow("SimpleLive_Python_uEye_OpenCV", frame)
        time.sleep(0.5)
        showInMovedWindow("SimpleLive_Python_uEye_OpenCV", frame)
        time.sleep(0.5)

        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)

        # Press q if you want to end the loop
        #if cv.waitKey(1) & 0xFF == ord('q'):
        # if cv.waitKey(100) or 0xFF == ord('q'):
        frame = frame[:,:,0:3]
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # print(frame.shape)
        folder_path_for_images = r'D:\Users\NanoProject\SonarNano\exp_data_16_2_21\rgb_images'
        img_name = 'num_dt.jpeg'
        img_name = img_name.replace("num", str(image_number))
        img_name = img_name.replace("dt", str(current_time))
        image_path = os.path.join(folder_path_for_images, img_name)

        plt.imsave(image_path, frame)
        g_param.read_write_object.save_rgb_image(frame)
        cv.destroyAllWindows()

        # print("image_path!!!!!!:", image_path)
        # a_string = "C:/Drive/25_11_20/num_dt.jpeg"
        # a_string = a_string.replace("num", str(image_number))
        # a_string = a_string.replace("dt", str(current_time))
        # a_string = a_string.replace("num", str(i))
        # im = Image.fromarray(frame)
        # plt.imsave(a_string, frame)
        # break
    # ---------------------------------------------------------------------------------------------------------------------------------------

    # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
    ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

    # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
    ueye.is_ExitCamera(hCam)

    # Destroys the OpenCv windows
    cv.destroyAllWindows()
    #
    # print()
    # print("END")
    return image_path


def biggest_box(det_rotated_boxes):
    areas = []
    for i in range(len(det_rotated_boxes)):
        area = det_rotated_boxes[i][2] * det_rotated_boxes[i][3]
        areas.append(area)
    return areas.index(max(areas))


def calc_biggest_box(boxes):
    biggest = boxes[0][2] * boxes[0][3]
    biggest_index = 0
    for i in range(len(boxes)):
        if boxes[i][2] * boxes[i][3] > biggest:
            biggest = boxes[i][2] * boxes[i][3]
            biggest_index = i
    return biggest_index


def pixel_2_meter(d, box):
    cen_poi_x_0 = box[0]
    cen_poi_y_0 = box[1]
    cen_poi_x_0 = cen_poi_x_0 - int(1024 / 2)
    cen_poi_y_0 = cen_poi_y_0 - int(1024 / 2)
    cen_poi_x_0 = d * (cen_poi_x_0 / 1024) * (7.11 / 8)
    cen_poi_y_0 = d * (cen_poi_y_0 / 1024) * (5.33 / 8) * 1.33  # FIXME talk to Sigal
    w = d * (box[2] / 1024) * (7.11 / 8)
    h = d * (box[3] / 1024) * (5.33 / 8)
    return [cen_poi_x_0, cen_poi_y_0, w, h, box[4]]


def meter_2_pixel(d, i):
    cen_poi_x_0 = g_param.TB[i].x_world_meter
    cen_x = (cen_poi_x_0 * 16 * 1024) / (1.2 * d)
    g_param.TB[i].x_world_meter = cen_x
    # cen_poi_y_0 = g.TB[i].y_p
    # y_m = d * (cen_poi_y_0 / 1024) * (1.6 / 16)


# im1 = 'path to captured image indside cv2.imageread'
def take_picture_and_run(current_location, image_number):
    d = g_param.const_dist
    box = [0, 0, 0, 0, 0]
    plt.clf()  # clean the canvas
    image_details = f"Picure number {image_number}"
    print(colored(image_details, 'green'))
    image_path = ueye_take_picture_2(image_number)
    # img = cv.imread(image_path)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #
    # cv.destroyAllWindows()
    rng.seed(12345)

    def fix_angle_to_0_180(w, h, a):
        if w <= h:
            a += 180
        else:
            a += 90
        return a

    def thresh_callback(val):
        threshold = val
        ret, thresh = cv.threshold(src_gray, 50, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        thresh = cv.dilate(thresh, kernel, iterations=2)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        thresh = cv.erode(thresh, kernel, iterations=2)
        canny_output = cv.Canny(thresh, threshold, threshold * 2)
        contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Find the rotated rectangles and ellipses for each contour
        minRect = [None] * len(contours)
        corners_rect = [None] * len(contours)
        # rect = ((center_x,center_y),(width,height),angle)
        for i, c in enumerate(contours):
            minRect[i] = cv.minAreaRect(c)
            corners_rect[i] = cv.boxPoints(minRect[i])
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

        for i, c in enumerate(contours):
            # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            box = cv.boxPoints(minRect[i])
            box = np.intp(box)
            print("points ", corners_rect[i]) # TODO- continue from here
            # width_from_points = np.linalg.norm(a-b)
            width_a = int(minRect[i][1][0])
            height_a = int(minRect[i][1][1])
            area = width_a * height_a
            # print(minRect[i]) # for Debugging
            # print(int(minRect[i][1][0]), int(minRect[i][1][1])) # for Debugging
            color_index = 50
            tresh_size = 6000
            if area > tresh_size and (width_a/height_a > 0.15) and (height_a/width_a > 0.15):
                # print("area: ", area)
                # TODO: להכפיל את הזוית ב-1
                boxes.append(minRect[i])
                cv.drawContours(green, [box], 0, (255-color_index,255-color_index*2,255-color_index*3))
                color_index += 20
        # cv.imshow('mask__', green)

    parser = argparse.ArgumentParser(
        description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
    args = parser.parse_args()
    img = cv.imread(image_path)
    # img = cv.resize(img, (1024, 692))  # Resize image
    RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ## mask of green (36,25,25) ~ (86, 255,255)q
    mask = cv.inRange(RGB, (40,40,50), (255, 255, 255))
    ## slice the green
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    src_gray = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    # na = np.array(src_gray) # TODO fixme- next 4 lines if only black image, take another image.
    # f = np.dot(na.astype(np.uint32), [1])
    # nColours = len(np.unique(f))
    # print("nColours", nColours)


    source_window = 'Source'
    cv.namedWindow(source_window)
    max_thresh = 255
    thresh = 100  # initial threshold
    cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
    boxes = []
    thresh_callback(thresh)
    # print(len(boxes), boxes)
    boxes = set(boxes) # remove duplicates
    boxes = list(boxes)
    print(boxes)
    cv.putText(green, 'Grapes detected', org=(500, 85),
                       fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
                       color=(255, 255, 255), thickness=1,
                       lineType=2)

    cv.waitKey(0)
    cv.destroyAllWindows()


##################### commented ******************
    predicted_masks_to_mrbb = []
    amount_of_mask_detacted = len(boxes)
    for i in range(0, amount_of_mask_detacted):
        x = int(boxes[i][0][0])
        y = int(boxes[i][0][1])
        w = int(boxes[i][1][0])
        h = int(boxes[i][1][1])
        a = int(boxes[i][2])
        box = [x, y, w, h, a]
        predicted_masks_to_mrbb.append(box)

    det_rotated_boxes = []
    # for the new function
    for b in range(0, len(predicted_masks_to_mrbb)):
        cen_poi_x_0 = predicted_masks_to_mrbb[b][0]
        cen_poi_y_0 = predicted_masks_to_mrbb[b][1]
        width_0 = predicted_masks_to_mrbb[b][2]
        height_0 = predicted_masks_to_mrbb[b][3]
        angle_0 = predicted_masks_to_mrbb[b][4] * -1
        # angle_0 = fix_angle_to_0_180(w=width_0, h=height_0, a=angle_0)
        # angle_0 = (angle_0*180)/pi
        det_box = [int(cen_poi_x_0), int(cen_poi_y_0), int(width_0), int(height_0), angle_0, None]
        box_in_meter = pixel_2_meter(d, det_box)
        det_rotated_boxes.append(box_in_meter)
        grape = [box_in_meter[0], box_in_meter[1], box_in_meter[2], box_in_meter[3], box_in_meter[4], None,  det_box]
        add_to_TB(grape)


    print("boxes", boxes)
    # using the TB
    box_index = sort_by_and_check_for_grapes('rect_size')
    print("box_index", box_index)
    img = cv.circle(img, (int(1024/2), int(1024/2)), radius=5, color=(0, 255, 0), thickness=3)
    img = cv.putText(img, 'Green = Center point, Red = already sprayed', org = (175, 85),
                     fontFace = cv.FONT_HERSHEY_COMPLEX, fontScale = 1,
                     color = (255, 255, 255), thickness = 1,  lineType = 2)
    numpy_horizontal_concat = np.concatenate((img, green), axis=1)
    numpy_horizontal_concat = image_resize(numpy_horizontal_concat, height=950)
    # cv.imshow("Masks and first Chosen grape cluster to spray", numpy_horizontal_concat)
    showInMovedWindow("Masks and first Chosen grape cluster to spray", numpy_horizontal_concat)
    g_param.masks_image = img
    cv.waitKey(0) # TODO: uncomment
    cv.destroyAllWindows()


if __name__ == '__main__':
    g_param.masks_image = None
    take_picture_and_run()
