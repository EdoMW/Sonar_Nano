from __future__ import print_function
import cv2 as cv
# import argparse
import random as rng
import matplotlib.pyplot as plt
# import matplotlib
import scipy
from termcolor import colored
# import itertools
from Target_bank import check_if_in_TB, add_to_target_bank, sort_by_and_check_for_grapes, sort_by_rect_size
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from numpy import *
import math
from math import pi, cos, sin
import sys
from pyueye import ueye
import numpy as np
from operator import itemgetter
np.set_printoptions(precision=3)
import g_param
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import imutils
from PIL import Image, ImageDraw, ImageDraw
import scipy.misc
from add_grapes import add_grapes
from sty import fg, Style, RgbFg

fg.green = Style(RgbFg(31, 177, 31))
fg.yellow = Style(RgbFg(255, 255, 70))

# parameters #
####################################
amount_of_tries = 3
num_of_pixels = 100
####################################


if g_param.work_place == "field":
    # Root directory of the project
    ROOT_DIR = os.path.abspath("C:/Drive/Mask_RCNN-master")

    import warnings

    warnings.filterwarnings("ignore")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from self_utils.utils import *
    import self_utils.model as modellib
    from self_utils.config import Config
    from self_utils.visualize import *
    from self_utils.model import log
    from self_utils import visualize

    # from mrcnn import utils
    # from mrcnn.config import Config
    # import mrcnn.model as modellib
    # from mrcnn import visualize
    # from mrcnn.model import log
    import imgaug

    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, r"samples\coco"))  # To find local versio
    print(sys.path)
    from pycocotools import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Root directory of the project
    ROOT_DIR = os.getcwd()
    if ROOT_DIR.endswith("C:\Drive\Mask_RCNN-master"):
        # Go up two levels to the repo root
        ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)

    # Path to trained weights file
    COCO_WEIGHTS_PATH = os.path.join("C:\Drive\Mask_RCNN-master", "mask_rcnn_coco.h5")

    # Directory to save logs and model checkpoints, if not provided
    # through the command line argument --logs
    DEFAULT_LOGS_DIR = MODEL_DIR


    ############################################################
    #  Configurations
    ############################################################

    class GrapeConfig(Config):
        """Configuration for training on the toy dataset.
        Derives from the base Config class and overrides some values.
        """
        # Give the configuration a recognizable name
        NAME = "grape"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 2

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # Background + grape

        # Number of training steps per epoch
        STEPS_PER_EPOCH = 30

        # Skip detections with < 90% confidence
        DETECTION_MIN_CONFIDENCE = 0.95

        DETECTION_NMS_THRESHOLD = 0.1

        DETECTION_MAX_INSTANCES = 40

        MASK_SHAPE = [28, 28]
        # very important to define correctly!!!!!
        BACKBONE = "resnet101"

        BACKBONE_STRIDES = [4, 8, 16, 32, 64]

        RPN_ANCHOR_RATIOS = [0.5, 1, 2]


    ############################################################
    #  Dataset
    ############################################################

    class GrapeDataset(utils.Dataset):
        def load_grape(self, dataset_dir, subset):
            """Load a subset of the grape dataset.
            dataset_dir: Root directory of the dataset.
            subset: Subset to load: train or val
            """
            # Add classes. We have only one class to add.
            self.add_class("grape", 1, "grape_cluster")

            # Train or validation dataset?
            assert subset in ["train", "val", "test"]
            dataset_dir = os.path.join("C:\Drive\Mask_RCNN-master\samples\grape\dataset", subset)
            dataset = os.listdir(dataset_dir)

            # Add images
            import ntpath
            for image_id in dataset:
                img_path = os.path.join(dataset_dir, image_id)
                img_name = os.path.splitext(ntpath.basename(img_path))[0]
                self.add_image(
                    "grape",
                    image_id=str(img_name),
                    path=img_path)

        def load_mask(self, image_id):
            import numpy as np
            """Generate instance masks for an image.
           Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
            """
            # get mask directory from image id
            mask_dir = self.mask_reference(image_id)
            mask = np.load(mask_dir)
            # Return mask, and array of class IDs of each instance. Since we have
            # one class ID only, we return an array of 1s
            return mask, np.ones([mask.shape[-1]], dtype=np.int32)

        def image_reference(self, image_id):
            """Return the path of the image."""
            info = self.image_info[image_id]
            if info["source"] == "grape":
                return info["path"]
            else:
                super(self.__class__, self).image_reference(image_id)

        def mask_reference(self, image_id):
            """Return the mask directory of the image."""
            info = self.image_info[image_id]
            image_name = info["id"]
            mask_name = image_name + ".npy"
            mask_dir = os.path.join("C:/Drive/Mask_RCNN-master/samples/grape/anew_masks/" + mask_name)
            return mask_dir


    ############################ trainig  1  #######################################
    def train(model):
        """Train the model."""
        # Training dataset.
        dataset_train = GrapeDataset()
        dataset_train.load_grape('C:/Drive/Mask_RCNN-master/samples/grape/dataset', "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = GrapeDataset()
        dataset_val.load_grape('C:/Drive/Mask_RCNN-master/samples/grape/dataset', "val")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        # augmentation_1 = imgaug.augmenters.Sequential([
        # imgaug.augmenters.Fliplr(0.5),
        # imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0)),
        # imgaug.augmenters.Crop(percent=(0, 0.1)),
        # imgaug.augmenters.LinearContrast((0.75, 1.5))],
        #                                          random_order=True)
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=70,
                    # augmentation = augmentation_1,
                    layers='all')


    ############################################################
    # Training 2
    ############################################################

    # if __name__ == '__main__':
    #     config = GrapeConfig()
    #     config.display()
    #     logs_dir = 'C:\Drive\Mask_RCNN-master/logs'
    #     model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=logs_dir)
    #     # weights_path = '/content/gdrive/My Drive/grapes data/Mask_RCNN-master/mask_rcnn_coco.h5'
    #     # weights_path = model.find_last()
    #     # print(weights_path)
    #     model.load_weights('C:\Drive\Mask_RCNN-master\logs_to_import\exp_7\mask_rcnn_grape_0080.h5')
    #     # train (model)

    config = GrapeConfig()
    config.display()
    logs_dir = 'C:\Drive\Mask_RCNN-master/logs'
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=logs_dir)
    # weights_path = '/content/gdrive/My Drive/grapes data/Mask_RCNN-master/mask_rcnn_coco.h5'
    # weights_path = model.find_last()
    # print(weights_path)
    model.load_weights(r'C:\Drive\Mask_RCNN-master\logs_to_import\exp_7\mask_rcnn_grape_0080.h5')


    # train (model)

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = InferenceConfig()
    config.display()

    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    # DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"


    def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax


    classes = []
    classes.append('BG')
    classes.append('grape_cluster')

    import tensorflow as tf

    with tf.device("/gpu:0"):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    weights_path = r'D:\Users\NanoProject\Sonar_Nano\weights\2021_weights.h5'
    print("Loading weights ", weights_path)
    g_param.cnn_config = g_param.get_cnn_config(GrapeConfig())
    model.load_weights(weights_path, by_name=True)

    dataset_test = GrapeDataset()
    dataset_test.load_grape('C:\Drive\Mask_RCNN-master\samples\grape\dataset', "test")
    dataset_test.prepare()


def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    resize image
    :param image: image
    :param width: width after resizing
    :param height: height after resizing
    :param inter: interpolation method for resizing
    :return: resized image
    """
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
    resized = cv.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


# def show_in_moved_window(win_name, img, i=None, x=(-1090), y=35): # lab
def show_in_moved_window(win_name, img, i=None, x=0, y=0):  # lab
    """
    show image
    :param win_name: name of the window
    :param img: image to display
    :param i: index of the grape
    :param x: x coordinate of end left corner of the window
    :param y: y coordinate of end left corner of the window
    """
    if img is not None:
        target_bolded = img.copy()
        if i is not None:
            # if not g_param.TB[i].sprayed:
            #     print("grape to display: ", g_param.TB[i])
            cv.drawContours(target_bolded, [np.asarray(g_param.TB[i].p_corners)], 0, (15, 25, 253), thickness=3)
        cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)  # Create a named window
        cv.moveWindow(win_name, x, y)  # Move it to (x,y)
        # cv.resizeWindow(win_name, 400, 512)
        cv.imshow(win_name, target_bolded)
        cv.waitKey()
        cv.destroyAllWindows()


def masks_to_convex_hulls(list_of_masks):
    """
    convert masks (list) into convex hull (list of points that construct the convex hull per each image
    :param list_of_masks: list of masks to convert
    :return: two lists
    """
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
    """
    convert singal mask to convex hull.
    not efficent!!! find better way.
    :param mask: mask to convert
    :return: convex hull
    """
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


def two_value_to_int_vec(vec):
    """
    :param vec:
    :return: change from float vec to int vec
    """
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


def output_dict(npas):
    """
    mrbbs (minimum rodated bounding box s ) is a list of dicitnoray
    that contains the next parameters:
    rot_angle, area, width, height, center_point, corner_points
    """
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
    # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    # You can query additional information about the sensor type used in the camera
    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    # Set display mode to DIB
    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    # Set the right color mode
    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
        bytes_per_pixel = int(nBitsPerPixel / 8)
    # Can be used to set the size and position of an "area of interest"(AOI) within an image
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    # if nRet != ueye.IS_SUCCESS:
    #     print("is_AOI ERROR")
    width = rectAOI.s32Width
    height = rectAOI.s32Height

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

    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Continuous image display
    if nRet == ueye.IS_SUCCESS:
        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        pic_array_1 = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        # time.sleep(0.1)
        time.sleep(0.5)
        pic_array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

        # ...reshape it in an numpy array...
        frame = np.reshape(pic_array, (height.value, width.value, bytes_per_pixel))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_original_quality = frame[:, :, 0:3].copy()
        frame = np.pad(frame, pad_width=[(506, 506), (0, 0), (0, 0)], mode='constant')  # pad with zeros above and under
        # ...resize the image by a half
        frame = cv.resize(frame, (0, 0), fx=0.331606, fy=0.331606)

        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)

        frame = frame[:, :, 0:3]

        # TODO make folder_path_for_images a parameter
        folder_path_for_images = r'D:\Users\NanoProject\Images_for_work'
        img_name = 'num_dt.jpeg'
        img_name = img_name.replace("num", str(image_number))
        img_name = img_name.replace("dt", str(current_time))
        image_path = os.path.join(folder_path_for_images, img_name)
        plt.imsave(image_path, frame)
        # if g_param.process_type == "record":
        #     g_param.read_write_object.save_rgb_image(frame, frame_original_quality)
        cv.destroyAllWindows()

    # ---------------------------------------------------------------------------------------------------------------------------------------
    else:
        print("no image was taken")
    # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
    ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

    # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
    ueye.is_ExitCamera(hCam)

    # Destroys the OpenCv windows
    cv.destroyAllWindows()
    return image_path, frame, frame_original_quality


def biggest_box(det_rotated_boxes):
    """
    calc the biggest rotated box (w * h)
    :param det_rotated_boxes: detected rotated box
    :return: index of the biggest box
    """
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
    cen_poi_y_0 = d * (cen_poi_y_0 / 1024) * (5.33 / 8) * 1.33
    return [cen_poi_x_0, cen_poi_y_0]


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


def point_meter_2_pixel(d, point):
    """
    :param d: distance to grape
    :param point: point to convert to pixel
    :return: [x, y] in pixels relative to the top left corner
    """
    x_meter = point[0]
    y_meter = point[1]
    cen_poi_x_0 = 1024 * (x_meter / d) * (8 / 7.11)
    cen_poi_y_0 = 1024 * (y_meter / d) * (8 / 5.33) * (1 / 1.33)
    cen_poi_x_0 = cen_poi_x_0 + int(1024 / 2)
    cen_poi_y_0 = cen_poi_y_0 + int(1024 / 2)
    return np.array([cen_poi_x_0, cen_poi_y_0])


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
    w = np.linalg.norm(p1 - p2)
    h = np.linalg.norm(p2 - p3)
    if w > h:
        h, w = w, h
        p1, p2, p3, p4 = p1, p4, p3, p2
    return w, h, [p1, p2, p3, p4]


def meter_2_pixel(d, i):
    cen_poi_x_0 = g_param.TB[i].x_world_meter
    cen_x = (cen_poi_x_0 * 16 * 1024) / (1.2 * d)
    g_param.TB[i].x_world_meter = cen_x
    # cen_poi_y_0 = g.TB[i].y_p
    # y_m = d * (cen_poi_y_0 / 1024) * (1.6 / 16)


def print_special_cam_error():
    """
    :return: print red warnning massege if no real image was taken in N iterations
    """
    print(colored("3 images in a row were not taken successfully. Check camera", 'red'))


def check_image(image_path):
    """
    it's not- rgb[combined[i]] is the all image. (0,0,0) is just (0,0,0)
    :param image_path: path to last image taken
    :return: True if real image was taken (not only black pixels)
    """
    x_val = np.random.choice(1024, num_of_pixels, replace=False)
    y_val = np.random.choice(1024, num_of_pixels, replace=False)
    combined = np.vstack((x_val, y_val)).T
    # image_path_1 = d'D:\Users\NanoProject\Images_for_work\black.jpg'
    img = cv.imread(image_path)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    for i in range(num_of_pixels):
        ans = np.array_equal(rgb[combined[i][0], combined[i][1]], (0, 0, 0))
        # ans = np.all(rgb[combined[i]] == (0, 0, 0))
        if not ans:
            return True
    else:
        return False


def add_circle_and_index(img_1, img_2):
    """
    :param img_1: original image
    :param img_2: image without anything
    :return: display two images.
    """
    img = img_1
    green = img_2
    img = cv.circle(img, (int(1024 / 2), int(1024 / 2)), radius=5, color=(0, 255, 0), thickness=3)
    img = cv.putText(img, 'Green = Center point, Red = already sprayed', org=(175, 85),
                     fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
                     color=(255, 255, 255), thickness=1, lineType=2)
    numpy_horizontal_concat = np.concatenate((img, green), axis=1)
    numpy_horizontal_concat = image_resize(numpy_horizontal_concat, height=950)
    # cv.imshow("Masks and first Chosen grape cluster to spray_procedure", numpy_horizontal_concat)
    show_in_moved_window("Masks and first Chosen grape cluster to spray_procedure", numpy_horizontal_concat)


def display_image_with_masks(image):
    """
    :param image: image with all the masks
    :return: display image with the masks
    """
    cv.putText(image, 'Grapes detected', org=(500, 85), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1,
               color=(255, 255, 255), thickness=1, lineType=2)
    # cv.waitKey(1)
    cv.destroyAllWindows()


def fix_angle_to_0_180(width, height, ang):
    """
    :param width: w
    :param height: h
    :param ang: (-90) - 90
    :return: 0 - 180
    """
    if width <= height:
        ang += 90
    else:
        ang += 180
    return ang


def arrange_data(width_0, height_0, corner_points):
    """
    :param width_0: w
    :param height_0: h
    :param corner_points: corner points
    :return: make w the shorter edge. replace point numbering accordingly
    """
    p1, p2, p3, p4 = corner_points
    if width_0 > height_0:
        height_0, width_0 = width_0, height_0
        p1, p2, p3, p4 = p1, p4, p3, p2
    return width_0, height_0, [p1, p2, p3, p4]


def check_if_in_list(temp_box, list_a):
    """
    Check if new element is already in the list (allow tolerance of up to 3 pixels from each corner separately)
    :param temp_box: new element to check
    :param list_a: list of element
    :return: True if it is a new point
    """
    list_size = len(list_a)
    if list_size == 0:
        return True
    temp_box = np.hstack(temp_box)
    for i in range(list_size):
        curr_box = np.hstack(list_a[i])
        check_diff = np.isclose(temp_box, curr_box, atol=10.01)
        ans = np.all(check_diff)
        if ans:
            return False
    return True


def sort_results(results):
    """
    sort the results of detection from left to right, and not by score.
    :param results: dict of the results
    :return: results (dict, same size), sorted.
    """
    bbox = utils.extract_bboxes(results['masks'])
    bbox = bbox.astype('int32')
    results['bbox'] = bbox
    res = []
    for i in range(len(results['scores'])):
        res.append({
            "rois": results['rois'][i], "class_ids": results['class_ids'][i], "scores": np.array(results['scores'][i]),
            "masks": results['masks'][:, :, i], "bbox": bbox[i], "bbox_left_x": bbox[i][1],
        })
    res = sorted(res, key=itemgetter('bbox_left_x'), reverse=False)
    a, b, c, d = res[0]['rois'], res[0]['bbox'], res[0]['scores'], res[0]['masks']
    # a, b, c, d = res[0]['rois'], res[0]['bbox'], res[0]['scores'], res[0]['masks'].reshape(1024, 1024, 1)
    if len(res) > 1:
        for i in range(1, len(res)):
            # a = np.dstack((a, res[i]['bbox']))
            # b = np.dstack((b, res[i]['bbox']))
            d = np.dstack((d, res[i]['masks']))
            a = np.vstack((a, res[i]['rois'].reshape(1, 4)))
            b = np.vstack((b, res[i]['bbox'].reshape(1, 4)))
            # d = np.vstack((d, res[i]['masks'].reshape(1024, 1024, 1)))
            c = np.append(c, res[i]['scores'])
    else:
        d = d.reshape(1024, 1024)
    results = {"rois": a, "class_ids": results['class_ids'], "scores": c, "masks": d, "bbox": b,
               }
    return results


def good_manual_image():
    """
    Check if the manual image taken is good
    """
    print("Image is good?")
    one = "\033[1m" + "0" + "\033[0m"
    zero = "\033[1m" + "1" + "\033[0m"
    while True:
        time.sleep(0.01)
        good_image = input(colored("Yes: ", "cyan") + "press " + zero +
                           colored(" No: ", "red") + "Press " + one + " \n")
        if good_image == '0' or good_image == '1':
            break
    if good_image == '1':
        return True
    return False


def take_manual_image():
    """
    Saves extra image during the experiment.
    validate that the image taken was good (press 1 to confirm or 0 to take another one).
    to use in exp, should be inserted after move to spray.
    """
    image_number = g_param.image_number
    plt.clf()  # clean the canvas
    amount_of_images = 1
    if g_param.process_type == "record":
        not_good_image = True
        while not_good_image:
            for i in range(amount_of_tries):
                image_path, frame, original_frame = ueye_take_picture_2(image_number)
                image_taken = check_image(image_path)
                print("try number: ", amount_of_images, " for manual image")
                if image_taken:
                    amount_of_images += 1
                    print(colored("Manual image taken successfully", 'green'))
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    show_in_moved_window(win_name="Chack manual image", img=frame)
                    good_image_taken = good_manual_image()
                    if good_image_taken:
                        g_param.read_write_object.save_manual_image(original_frame)
                        not_good_image = False
                        break
            if not image_taken:
                print_special_cam_error()


def take_picture_and_run():
    """
    :return:
    """
    image_number = g_param.image_number
    d = g_param.avg_dist
    plt.clf()  # clean the canvas
    if g_param.process_type == "record" or g_param.process_type == "work":
        for i in range(amount_of_tries):  # TODO: comment next 3 lines if working with camera
            # image_path = r'D:\Users\NanoProject\old_experiments\exp_data_11_10\rgb_images\0_11_10_46.jpeg'
            # frame = cv.imread(image_path)
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_path, frame, original_frame = ueye_take_picture_2(
                image_number)  # TODO: uncomment if working with camera

            image_taken = check_image(image_path)
            print("try number: ", i)
            if image_taken:
                print(colored("Image taken successfully", 'green'))
                if g_param.process_type == "record":
                    g_param.read_write_object.save_rgb_image(frame, original_frame)
                break
        if not image_taken:
            print_special_cam_error()
    else:
        image_path = g_param.read_write_object.load_image_path()

    # parser = argparse.ArgumentParser(
    #     description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
    # parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
    # args = parser.parse_args()

    # fOR FIELD simulation in lab
    if g_param.work_place == "field":
        dataset_images = os.listdir(r'C:\Drive\Mask_RCNN-master\samples\grape\dataset\test')
        img_path = random.choice(dataset_images)
        image_path = os.path.join(r'C:\Drive\Mask_RCNN-master\samples\grape\dataset\test', img_path)
        print(f"image_path : {image_path}")

        image_path = r'C:\Drive\Mask_RCNN-master\samples\grape\dataset\test\DSC_0280.JPG'
        # image_path = r'C:\Drive\Mask_RCNN-master\samples\grape\dataset\test\DSC_0363.JPG'

    img = cv.imread(image_path)
    # img = cv.resize(img, (1024, 692))  # Resize image if needed
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if g_param.work_place == "lab":
        # for lab
        rng.seed(12345)

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
                width_a = int(minRect[i][1][0])
                height_a = int(minRect[i][1][1])
                area = width_a * height_a
                color_index = 50
                tresh_size = 6000
                if tresh_size < area < 200_000 and (width_a / height_a > 0.15) and (height_a / width_a > 0.15):
                    # להכפיל את הזוית ב-1
                    boxes.append(minRect[i])
                    corner_points.append(box)
                    cv.drawContours(rgb, [box], 0, (255 - color_index, 255 - color_index * 2, 255 - color_index * 3))
                    color_index += 20
            show_in_moved_window("Check image", rgb, None)
            # cv.imshow()
            # cv.waitKey()
            # cv.destroyAllWindows()

        ## mask of green (36,25,25) ~ (86, 255,255)
        rgb = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        mask = cv.inRange(rgb, (40, 40, 50), (255, 255, 255))
        ## slice the green
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        src_gray = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))
        thresh = 100  # initial threshold
        boxes, corner_points = [], []
        thresh_callback(thresh)
        corner_points = [arr.tolist() for arr in corner_points]
        # corner_points = list(corner_points for corner_points,_ in itertools.groupby(corner_points))  # remove duplicates

        mask, obbs_list, corners_list, img_rgb = add_grapes(rgb)  # adding new grapes that were not recognized
        corner_points = corner_points + corners_list

        boxes = boxes + obbs_list

        new_corner_points = []
        for elem in corner_points:
            if check_if_in_list(elem, new_corner_points):
                new_corner_points.append(elem)
        corner_points = new_corner_points
        boxes = map(list, boxes)
        boxes = [list(elem) for elem in boxes]
        for i in range(0, len(boxes)):
            for j in range(0, 2):
                boxes[i][j] = list(boxes[i][j])
            boxes[i][2] = [boxes[i][2]]
        for i in range(0, len(boxes)):
            boxes[i] = [[np.round(float(i), 0) for i in nested] for nested in boxes[i]]
        new_boxes = []
        for elem in boxes:
            if check_if_in_list(elem, new_boxes):
                new_boxes.append(elem)
        boxes = new_boxes
        # boxes_with_corners = [list(itertools.chain(*i)) for i in zip(boxes, corner_points)]  # [box, corners]
        predicted_masks_to_mrbb, det_rotated_boxes = [], []

        amount_of_mask_detacted = len(boxes)
        for i in range(0, len(boxes)):
            x = int(boxes[i][0][0])
            y = int(boxes[i][0][1])
            w, h, corners_in_meter = calculate_w_h(d, corner_points[i])
            a = int(boxes[i][2][0])
            box = [x, y, w, h, a, corners_in_meter, corner_points[i]]
            predicted_masks_to_mrbb.append(box)
        # for the new function
        for b in range(0, len(predicted_masks_to_mrbb)):
            cen_poi_x_0 = predicted_masks_to_mrbb[b][0]
            cen_poi_y_0 = predicted_masks_to_mrbb[b][1]
            width_0 = predicted_masks_to_mrbb[b][2]
            height_0 = predicted_masks_to_mrbb[b][3]
            angle_0 = predicted_masks_to_mrbb[b][4] * -1
            corners_in_meter = predicted_masks_to_mrbb[b][5]
            corner_points = predicted_masks_to_mrbb[b][6]
            width_0, height_0, corner_points = arrange_data(width_0, height_0, corner_points)
            # angle_0 = fix_angle_to_0_180(w=width_0, h=height_0, a=angle_0)
            # angle_0 = (angle_0*180)/pi
            det_box = [int(cen_poi_x_0), int(cen_poi_y_0), int(boxes[b][1][0]), int(boxes[b][1][1]), angle_0, None]
            # det_box = [int(cen_poi_x_0), int(cen_poi_y_0), width_0, height_0, angle_0, None]
            x_center_meter, y_center_meter = point_pixels_2_meter(d, [det_box[0], det_box[1]])
            box_in_meter = [x_center_meter, y_center_meter, width_0, height_0, angle_0]
            det_rotated_boxes.append(box_in_meter)
            grape = [box_in_meter[0], box_in_meter[1], box_in_meter[2], box_in_meter[3], box_in_meter[4],
                     None, det_box, None, corners_in_meter, corner_points, None, None]
            add_to_target_bank(grape)

    if g_param.work_place == "lab_grapes":
        if not g_param.manual_work:  # for exp. only manual grape detection
        # for lab
            rng.seed(12345)

            def thresh_callback(val):
                threshold = val
                ret, thresh = cv.threshold(src_gray, 5, 255, cv.THRESH_BINARY)
                contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                # Find the rotated rectangles and ellipses for each contour
                minRect = [None] * len(contours)
                corners_rect = [None] * len(contours)
                # rect = ((center_x,center_y),(width,height),angle)
                for i, c in enumerate(contours):
                    minRect[i] = cv.minAreaRect(c)
                    corners_rect[i] = cv.boxPoints(minRect[i])
                drawing = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
                for i, c in enumerate(contours):
                    box = cv.boxPoints(minRect[i])
                    box = np.intp(box)
                    width_a = int(minRect[i][1][0])
                    height_a = int(minRect[i][1][1])
                    area = width_a * height_a
                    color_index = 30
                    tresh_size_min, tresh_size_max = 8_000, 200_000
                    if (width_a / (height_a + 0.001) > 0.2) and ( height_a / (width_a + 0.001) > 0.2) and \
                            tresh_size_min < area < tresh_size_max:  # and tresh_size < area < 200_000:
                        # להכפיל את הזוית ב-1
                        boxes.append(minRect[i])
                        corner_points.append(box)
                        cv.drawContours(rgb, [box], 0, (255 - color_index, 255 - color_index * 2, 255 - color_index * 3))
                        color_index += 20
                show_in_moved_window("check image", rgb, None)
                # cv.imshow("show boxes", rgb)
                # cv.waitKey()
                # cv.destroyAllWindows()
            rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
            hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
            cv.destroyAllWindows()
            mask = cv.inRange(hsv, (35, 38, 38), (68, 255, 255))
            imask = mask > 0
            green = np.zeros_like(rgb, np.uint8)
            green[imask] = rgb[imask]
            temp_bgr = cv.cvtColor(green, cv.COLOR_HSV2RGB)
            # rgb = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            src_gray = cv.cvtColor(temp_bgr, cv.COLOR_RGB2GRAY)
            thresh = 100  # initial threshold
            boxes, corner_points = [], []
            thresh_callback(thresh)
            corner_points = [arr.tolist() for arr in corner_points]
        else:
            rgb = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            show_in_moved_window("check images", rgb, None)
            boxes, corner_points = [], []
        # mask, obbs_list, corners_list, img_rgb = None,[],[], None
        # try:
        mask, obbs_list, corners_list, img_rgb = add_grapes(rgb)  # adding new grapes that were not recognized
        # except Exception as e:
        #     print("exception: ", e.__class__)
        corner_points = corner_points + corners_list
        boxes = boxes + obbs_list

        new_corner_points = []
        for elem in corner_points:
            if check_if_in_list(elem, new_corner_points):
                new_corner_points.append(elem)
        corner_points = new_corner_points
        boxes = map(list, boxes)
        boxes = [list(elem) for elem in boxes]
        for i in range(0, len(boxes)):
            for j in range(0, 2):
                boxes[i][j] = list(boxes[i][j])
            boxes[i][2] = [boxes[i][2]]
        for i in range(0, len(boxes)):
            boxes[i] = [[np.round(float(i), 0) for i in nested] for nested in boxes[i]]
        new_boxes = []
        for elem in boxes:
            if check_if_in_list(elem, new_boxes):
                new_boxes.append(elem)
        boxes = new_boxes
        # boxes_with_corners = [list(itertools.chain(*i)) for i in zip(boxes, corner_points)]  # [box, corners]
        predicted_masks_to_mrbb, det_rotated_boxes = [], []
        amount_of_mask_detacted = len(boxes)
        for i in range(0, len(boxes)):
            x = int(boxes[i][0][0])
            y = int(boxes[i][0][1])
            w, h, corners_in_meter = calculate_w_h(d, corner_points[i])
            a = int(boxes[i][2][0])
            box = [x, y, w, h, a, corners_in_meter, corner_points[i]]
            predicted_masks_to_mrbb.append(box)
        # for the new function
        for b in range(0, len(predicted_masks_to_mrbb)):
            cen_poi_x_0 = predicted_masks_to_mrbb[b][0]
            cen_poi_y_0 = predicted_masks_to_mrbb[b][1]
            width_0 = predicted_masks_to_mrbb[b][2]
            height_0 = predicted_masks_to_mrbb[b][3]
            angle_0 = predicted_masks_to_mrbb[b][4] * -1
            corners_in_meter = predicted_masks_to_mrbb[b][5]
            corner_points = predicted_masks_to_mrbb[b][6]
            width_0, height_0, corner_points = arrange_data(width_0, height_0, corner_points)
            # angle_0 = fix_angle_to_0_180(w=width_0, h=height_0, a=angle_0)
            # angle_0 = (angle_0*180)/pi
            det_box = [int(cen_poi_x_0), int(cen_poi_y_0), int(boxes[b][1][0]), int(boxes[b][1][1]), angle_0, None]
            # det_box = [int(cen_poi_x_0), int(cen_poi_y_0), width_0, height_0, angle_0, None]
            x_center_meter, y_center_meter = point_pixels_2_meter(d, [det_box[0], det_box[1]])
            box_in_meter = [x_center_meter, y_center_meter, width_0, height_0, angle_0]
            det_rotated_boxes.append(box_in_meter)
            # change mask[:,:,b] to None if not working on manual mode
            if g_param.manual_work:
                grape = [box_in_meter[0], box_in_meter[1], box_in_meter[2], box_in_meter[3], box_in_meter[4],
                         mask[:, :, b], det_box, None, corners_in_meter, corner_points, None, None]
            else:
                grape = [box_in_meter[0], box_in_meter[1], box_in_meter[2], box_in_meter[3], box_in_meter[4],
                         None, det_box, None, corners_in_meter, corner_points, None, None]
            add_to_target_bank(grape)

        # for field usage with CNN
    if g_param.work_place == "field":
        # TODO: same as lab_grapes: if not g_param.manual_work:  # for exp. only manual grape detection
        im0 = rgb
        im0 = utils.resize_image(im0, max_dim=1024, mode="square")
        im0, *_ = np.asarray(im0)
        # if g_param.process_type == "record":
        #     g_param.read_write_object.save_rgb_image(im0, rgb)
        img = im0
        img_with_masks = img.copy()
        arr = [im0]
        show_in_moved_window("check image taken", img, None)
        cv.waitKey()
        cv.destroyAllWindows()
        # cv.imshow("before detection", arr[0])
        # cv.waitKey()
        # use THE MASK R-CNN for real grapes: next 93 lines
        results = model.detect(arr, verbose=1)
        r = results[0]
        pred_masks = r['masks']
        print("amount of grapes :", len(pred_masks[0][0]))
        print(fg.yellow + "wait" + fg.rs, "\n")
        if len(pred_masks[0][0]) > 0:
            im1 = im0
            r = results[0]
            ax = get_ax(1)
            bbox = utils.extract_bboxes(r['masks']).astype('int32')
            r = sort_results(r) # WORKING!!!! Finaly
            if r['masks'].ndim < 3:
                r['masks'] = r['masks'].reshape((1024, 1024, 1))
            img_with_masks = visualize.display_instances(im1, bbox, r['masks'], r['class_ids'],
                                                         dataset_test.class_names, r['scores'], ax=ax,
                                                         title="Predictions", show_bbox=True)
        images, boxes, mini_boxes, boxes_min, pixels_count_arr, com_list = [], [], [], [], [], []
        pred_masks = r['masks']
        if len(pred_masks[0][0]) > 0:
            amount_of_mask_detacted = len(pred_masks[0][0])
            for i in range(amount_of_mask_detacted):
                com = np.asarray(scipy.ndimage.measurements.center_of_mass(pred_masks[:, :, i]))  # TODO- add to TB
                com = np.round(com)
                com_list.append(com)
                # load the image, convert it to grayscale, and blur it slightly
                im = Image.fromarray(pred_masks[:, :, i])
                im.save("your_file.jpeg")
                image = cv.imread("your_file.jpeg")
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                # threshold the image,
                thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)[1]
                # find contours in thresholded image, then grab the largest

                cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv.contourArea)

                hull = cv.convexHull(c)
                obb = cv.minAreaRect(hull)
                boxes.append(obb)
                box_min = cv.boxPoints(obb)
                box_min = np.int0(box_min)
                mini_boxes.append(box_min)
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv.fillPoly(mask, [c], [255, 255, 255])
                mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
                pixels_count = cv.countNonZero(mask)
                box_min = box_min.tolist()
                boxes_min.append(box_min)
                pixels_count_arr.append(pixels_count)
        amount_of_mask_detacted = len(pred_masks[0][0])
        print(fg.green + "continue" + fg.rs, "\n")
        masks, obbs_list, corners_list, img_rgb = add_grapes(img_with_masks)  # adding new grapes that were not recognized

        if len(obbs_list) > 0:
            pred_masks = np.dstack((pred_masks, masks))
            pixels_count_arr = pixels_count_arr + [10_000] * len(obbs_list)
            com_list = com_list + [(500, 500)] * len(obbs_list)
            boxes = boxes + obbs_list
            boxes_min = boxes_min + corners_list
            amount_of_mask_detacted += len(obbs_list)

        for i in range(amount_of_mask_detacted):
            pixels_count = pixels_count_arr[i]
            box_min = boxes_min[i]
            x, y = int(boxes[i][0][0]), int(boxes[i][0][1])
            w, h, corners_in_meter = calculate_w_h(d, box_min)
            a = int(boxes[i][2])
            box = [x, y, w, h, a, corners_in_meter, box_min]
            cen_poi_x_0, cen_poi_y_0 = box[0], box[1]
            width_0, height_0 = box[2], box[3]
            angle_0 = box[4] * -1
            corners_in_meter, corner_points = box[5], box[6]
            width_0, height_0, corner_points = arrange_data(width_0, height_0, corner_points)
            # det_box = [int(cen_poi_x_0), int(cen_poi_y_0), width_0, height_0, angle_0, None]
            det_box = [int(cen_poi_x_0), int(cen_poi_y_0), int(boxes[i][1][0]), int(boxes[i][1][1]), angle_0, None]
            x_center_meter, y_center_meter = point_pixels_2_meter(d, [det_box[0], det_box[1]])
            box_in_meter = [x_center_meter, y_center_meter, width_0, height_0, angle_0]
            # det_rotated_boxes.append(box_in_meter)
            grape = [box_in_meter[0], box_in_meter[1], box_in_meter[2], box_in_meter[3], box_in_meter[4],
                     pred_masks[:, :, i], det_box, None, corners_in_meter, corner_points, pixels_count, com_list[i]]
            add_to_target_bank(grape)

    if g_param.work_place == "lab":
        display_image_with_masks(green)
        add_circle_and_index(img, green)
        cv.waitKey(0)
        cv.destroyAllWindows()
        g_param.masks_image = rgb
    else:
        g_param.masks_image = img
    g_param.read_write_object.write_tb()


if __name__ == '__main__':
    g_param.masks_image = None
    take_picture_and_run()
