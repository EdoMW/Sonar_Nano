from __future__ import print_function
import g
import os
import sys
import cv2 as cv
import numpy as np
import argparse
import g_param
import random as rng
import matplotlib.pyplot as plt
from Target_bank import check_if_in_TB, add_to_TB, sort_by_and_check_for_grapes, sort_by_rect_size
# import random
# import math
# import argparse
# import numpy as np
# import skimage.io
# import cv2
# import time
# import re
# import matplotlib
# import skimage.draw
# from statistics import mean
# from statistics import stdev
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
# from PIL import Image, ImageDraw, ImageDraw
# import cv2
# import scipy.misc
# import matplotlib
# from distance import correlation_dist
# # from __future__ import print_function
# import cv2 as cv
# import numpy as np
# import argparse
# import random as rng
# from scipy.spatial import distance




# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Drive/Mask_RCNN-master/")

import warnings

warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco"))  # To find local version
import coco

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
    DETECTION_MIN_CONFIDENCE = 0.9

    DETECTION_MAX_INSTANCES = 40

    MASK_SHAPE = [28, 28]
    # very importent to defince corect!!!!!
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
    import imgaug
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
model.load_weights('C:\Drive\Mask_RCNN-master\logs_to_import\with_santos\la\mask_rcnn_grape_0080.h5')
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


dataset_val = GrapeDataset()
dataset_val.load_grape('C:\Drive\Mask_RCNN-master\samples\grape\dataset', "val")
dataset_val.prepare()

print("Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))
classes = []
classes.append('BG')
classes.append('grape_cluster')

import tensorflow as tf

with tf.device("/cpu:0"):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
# weights_path = model.find_last()
weights_path = 'C:\Drive\Mask_RCNN-master\logs_to_import\with_santos\la\mask_rcnn_grape_0080.h5'

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

dataset_test = GrapeDataset()
dataset_test.load_grape('C:\Drive\Mask_RCNN-master\samples\grape\dataset', "test")
dataset_test.prepare()

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from numpy import *
import sys
import math
from math import pi, cos, sin


# check if i can give up calc convex hulls

# [1] get each indivdul mask
# [2] run turogh nested for to get the positions of the mask and append them to list
# [3] convert list to np.array


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
    corner_1_new = []
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

    def compute_grape_iou(det_box, gt_boxes, dt_box_area, gt_boxes_area):
        """Calculates IoU of the given box with the array of the given boxes.
        det_box: 1D vector [y1, x1, y2, x2]
        gt_boxes: [boxes_count, (y1, x1, y2, x2)]
        dt_box_area: float. the area of 'det_box'
        gt_boxes_area: array of length boxes_count.
        Note: the areas are passed in rather than calculated here for
        efficiency. Calculate once in the caller to avoid duplicate work.
        """
        # Calculate intersection areas
        temp_ious = []
        for i in range(0, len(gt_boxes)):
            inter_area = intersection_area(det_box, gt_boxes[i])
            union = dt_box_area + gt_boxes_area[i] - inter_area
            iou = inter_area / union
            temp_ious.append(iou)
        return temp_ious

        # TO DO: check if it works with zero detections

    def compute_paper_measurements_grapes(pred_masks, gt_mask, iou):
        """Compute the recall at the given IoU threshold. It's an indication
        of how many GT boxes were found by the given prediction boxes.
        """
        # Measure overlaps
        overlaps = utils.compute_overlaps_masks(gt_mask, pred_masks)
        biggest_mask_iou = None
        main_mask_index = None
        if gt_mask.shape[-1] > 0:  # finding the mask with the biggest area
            areas = []
            for i in range(0, len(gt_mask[0][0])):
                mask = gt_mask[:, :, i]
                area = np.count_nonzero(mask)
                areas.append(sum(area))
            main_mask_index = areas.index(max(areas))

        iou_max = np.max(overlaps, axis=1)
        if main_mask_index != None:
            biggest_mask_iou = iou_max[main_mask_index]
        if biggest_mask_iou != None:
            if biggest_mask_iou >= 0.5:
                hit = 1
            else:
                hit = 0
        else:
            hit = 0
        iou_argmax = np.argmax(overlaps, axis=1)
        positive_ids = np.where(iou_max >= iou)[0]
        matched_gt_masks = iou_argmax[positive_ids]
        if len(gt_mask[0][0]) > 0:
            recall = len(set(matched_gt_masks)) / len(gt_mask[0][0])
        else:
            recall = None
        # else:
        #   if len(set(matched_gt_masks)) == 0:
        #     recall = 1
        #   else:
        #     recall = 0
        if len(pred_masks[0][0]) > 0:
            precision = len(set(matched_gt_masks)) / len(pred_masks[0][0])
        elif gt_mask[0][0] > 0:
            precision = 0
        else:
            precision = None
        if precision != None and recall != None:
            if precision > 0 and recall > 0:
                F1_score = (2 * recall * precision) / (precision + recall)
            else:
                F1_score = None
        else:
            F1_score = None
        return recall, precision, F1_score, positive_ids, biggest_mask_iou, hit

    def compute_overlaps_rotated(GT_rotated_boxes, det_rotated_boxes):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        For better performance, pass the largest set first and the smaller second.
        """
        # Areas of anchors and GT boxes
        area1 = []
        if len(GT_rotated_boxes) > 0:
            for i in range(0, len(GT_rotated_boxes)):
                area1.append((GT_rotated_boxes[i][2]) * (GT_rotated_boxes[i][3]))  # do it with for loops
            main_box_index = area1.index(max(area1))
        else:
            main_box_index = None

        area2 = []
        for i in range(0, len(det_rotated_boxes)):
            area2.append((det_rotated_boxes[i][2]) * (det_rotated_boxes[i][3]))  # do it with for loops
        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((amount_of_gt_masks, amount_of_mask_detacted))
        for i in range(overlaps.shape[1]):
            box2 = det_rotated_boxes[i]
            overlaps[:, i] = calcs.compute_grape_iou(box2, GT_rotated_boxes, area2[i], area1)

        main_overlaps = np.zeros((1, amount_of_mask_detacted))
        main_overlaps[0, :] = calcs.compute_grape_iou(GT_rotated_boxes[main_box_index], det_rotated_boxes,
                                                      area1[main_box_index], area2)
        return overlaps, main_overlaps, main_box_index

    def compute_measurements_grapes(det_rotated_boxes, GT_rotated_boxes, iou):
        """Compute the recall at the given IoU threshold. It's an indication
        of how many GT boxes were found by the given prediction boxes.
        pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
        gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
        """
        # Measure overlaps
        overlaps, main_overlaps, main_box_index = calcs.compute_overlaps_rotated(GT_rotated_boxes,
                                                                                 det_rotated_boxes)
        # the iou of the biggest rotated box from gt
        iou_main = np.max(main_overlaps, axis=1)
        iou_max = np.max(overlaps, axis=1)
        iou_argmax = np.argmax(overlaps, axis=1)
        positive_ids = np.where(iou_max >= iou)[0]
        matched_gt_boxes = iou_argmax[positive_ids]
        # for i in range (0,len(matched_gt_boxes)):
        #   print(matched_gt_boxes[i])
        # for i in range (0,len(positive_ids)):
        #   print(positive_ids[i])
        main_dist = None
        if iou > 0.2 or iou_main < 0.05:
            distances = calcs.calc_distances_between_centers(matched_gt_boxes,
                                                             positive_ids,
                                                             GT_rotated_boxes,
                                                             det_rotated_boxes)
        else:
            distances, main_dist = calcs.calc_distances_between_centers_with_main(matched_gt_boxes,
                                                                                  positive_ids,
                                                                                  GT_rotated_boxes,
                                                                                  det_rotated_boxes,
                                                                                  main_box_index)
        angles = calcs.calc_angle_between_boxes(matched_gt_boxes, positive_ids,
                                                GT_rotated_boxes, det_rotated_boxes)
        recall = len(set(matched_gt_boxes)) / len(GT_rotated_boxes)
        precision = len(set(matched_gt_boxes)) / len(det_rotated_boxes)
        if precision + recall > 0:
            F1_score = (2 * recall * precision) / (precision + recall)
        else:
            F1_score = 0
        if iou_main >= 0.5:
            hit = 1
        else:
            hit = 0
        return recall, positive_ids, distances, angles, precision, F1_score, iou_main, main_dist, hit

    def calc_distances_between_centers(matched_gt_boxes, positive_ids,
                                       GT_rotated_boxes, det_rotated_boxes):
        distances = []
        for i in range(0, len(matched_gt_boxes)):
            a = matched_gt_boxes[i]
            b = positive_ids[i]
            det_center_x = det_rotated_boxes[a][0]
            det_center_y = det_rotated_boxes[a][1]
            GT_center_x = GT_rotated_boxes[b][0]
            GT_center_y = GT_rotated_boxes[b][1]
            dist = math.hypot(det_center_x - GT_center_x, det_center_y - GT_center_y)
            distances.append(dist)
        return distances

    def calc_distances_between_centers_with_main(matched_gt_boxes, positive_ids,
                                                 GT_rotated_boxes, det_rotated_boxes,
                                                 main_box_index):
        distances = []
        for i in range(0, len(matched_gt_boxes)):
            a = matched_gt_boxes[i]
            b = positive_ids[i]
            det_center_x = det_rotated_boxes[a][0]
            det_center_y = det_rotated_boxes[a][1]
            GT_center_x = GT_rotated_boxes[b][0]
            GT_center_y = GT_rotated_boxes[b][1]
            dist = math.hypot(det_center_x - GT_center_x, det_center_y - GT_center_y)
            distances.append(dist)

        if main_box_index != None:
            main_cluster_index_arr = np.where(positive_ids == main_box_index)
            main_cluster_index = main_cluster_index_arr[0][0]
            aa = matched_gt_boxes[main_cluster_index]
            bb = positive_ids[main_cluster_index]
            det_center_x = det_rotated_boxes[aa][0]
            det_center_y = det_rotated_boxes[aa][1]
            GT_center_x = GT_rotated_boxes[bb][0]
            GT_center_y = GT_rotated_boxes[bb][1]
            main_dist = math.hypot(det_center_x - GT_center_x, det_center_y - GT_center_y)
        else:
            main_dist = None
        return distances, main_dist

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
from pyueye import ueye
import numpy as np
import time
import cv2
import sys
from PIL import Image

def ueye_take_picture_2(i):
    from pyueye import ueye
    import numpy as np

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

    # pParam = ueye.wchar_p()
    #
    # pParam.value = "/Desktop/tt.ini"
    #
    # ueye.is_ParameterSet(hCam, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, pParam, 0)

    # is_ParameterSet(m_hCam, IS_PARAMETERSET_CMD_LOAD_FILE, NULL, NULL)
    # ---------------------------------------------------------------------------------------------------------------------------------------
    print("START")
    print()

    # Starts the driver and establishes the connection to the camera
    nRet = ueye.is_InitCamera(hCam, None)
    if nRet != ueye.IS_SUCCESS:
        print("is_InitCamera ERROR")

    # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetCameraInfo ERROR")

    # You can query additional information about the sensor type used in the camera
    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetSensorInfo ERROR")

    # nRet = ueye.is_ResetToDefault(hCam)
    # if nRet != ueye.IS_SUCCESS:
    #    print("is_ResetToDefault ERROR")

    # Set display mode to DIB
    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    # nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_OPENGL)

    # Set the right color mode
    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_BAYER: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
        # for color camera models use RGB32 mode
        m_nColorMode = ueye.IS_CM_BGRA8_PACKED
        nBitsPerPixel = ueye.INT(32)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_CBYCRY: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
        # for color camera models use RGB32 mode
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("IS_COLORMODE_MONOCHROME: ", )
        print("\tm_nColorMode: \t\t", m_nColorMode)
        print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        print()

    else:
        # for monochrome camera models use Y8 mode
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        print("else")

    # Can be used to set the size and position of an "area of interest"(AOI) within an image
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        print("is_AOI ERROR")

    width = rectAOI.s32Width
    height = rectAOI.s32Height

    # Prints out some information about the camera and the sensor
    print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
    print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
    print("Maximum image width:\t", width)
    print("Maximum image height:\t", height)
    print()

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
    if nRet != ueye.IS_SUCCESS:
        print("is_InquireImageMem ERROR")
    else:
        print("Press q to leave the programm")

    # ---------------------------------------------------------------------------------------------------------------------------------------

    # Continuous image display
    while (nRet == ueye.IS_SUCCESS):

        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)

        # bytes_per_pixel = int(nBitsPerPixel / 8)

        # ...reshape it in an numpy array...
        frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
        frame = np.pad(frame, pad_width=[(506, 506), (0, 0), (0, 0)], mode='constant')
        # ...resize the image by a half
        frame = cv.resize(frame, (0, 0), fx=0.331606, fy=0.331606)
        # ---------------------------------------------------------------------------------------------------------------------------
        # Include image data processing here

        # ---------------------------------------------------------------------------------------------------------------------------------------

        # ...and finally display it
        cv.imshow("SimpleLive_Python_uEye_OpenCV", frame)

        # Press q if you want to end the loop
        # Press q if you want to end the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            frame = frame[:,:,0:3]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            print(frame.shape)
            from PIL import Image
            a_string = "C:/Drive/25_11_20/fake_grape_num.jpeg"
            a_string = a_string.replace("num", str(0))
            # a_string = a_string.replace("num", str(i))
            # import matplotlib.pyplot as plt
            im = Image.fromarray(frame)
            plt.imsave(a_string, frame)
            break
    # ---------------------------------------------------------------------------------------------------------------------------------------

    # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
    ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)

    # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
    ueye.is_ExitCamera(hCam)

    # Destroys the OpenCv windows
    cv.destroyAllWindows()

    print()
    print("END")

# change
max_number_of_pictures = 1


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


# def pixel_2_meter(d, box):
#     cen_poi_x_0 = box[0]
#     cen_poi_y_0 = box[1]
#     h = box[2]
#     w = box[3]
#     # cen_poi_x_0 = cen_poi_x_0 - int(config.IMAGE_MAX_DIM / 2)
#     # cen_poi_y_0 = cen_poi_y_0 - int(config.IMAGE_MAX_DIM / 2)
#     # cen_poi_x_0 = d * (cen_poi_x_0 / 1024) * (1.2 / 16)
#     # cen_poi_y_0 = d * (cen_poi_y_0 / 1024) * (1.6 / 16)
#     # w = d * (box[2] / 1024) * (1.2 / 16)
#     # h = d * (box[3] / 1024) * (1.2 / 16)
#     return [cen_poi_x_0/100, cen_poi_y_0/100, w/100, h/100, box[4]]

def pixel_2_meter(d, box):
    cen_poi_x_0 = box[0]
    cen_poi_y_0 = box[1]
    cen_poi_x_0 = cen_poi_x_0 - int(config.IMAGE_MAX_DIM / 2)
    cen_poi_y_0 = cen_poi_y_0 - int(config.IMAGE_MAX_DIM / 2)
    cen_poi_x_0 = d * (cen_poi_x_0 / 1024) * (7.11 / 8)
    cen_poi_y_0 = d * (cen_poi_y_0 / 1024) * (5.33 / 8)
    w = d * (box[2] / 1024) * (7.11 / 8)
    h = d * (box[3] / 1024) * (5.33 / 8)
    return [cen_poi_x_0, cen_poi_y_0, w, h, box[4]]


    # cen_poi_x_0 = box.x_p
    # cen_poi_y_0 = box.y_p
    # cen_poi_x_0 = cen_poi_x_0 - int(config.IMAGE_MAX_DIM / 2)
    # cen_poi_y_0 = cen_poi_y_0 - int(config.IMAGE_MAX_DIM / 2)
    # cen_poi_x_0 = d * (cen_poi_x_0 / 1024) * (1.2 / 16)
    # cen_poi_y_0 = d * (cen_poi_y_0 / 1024) * (1.6 / 16)
    # w = d * (box.w_p / 1024) * (1.2 / 16)
    # h = d * (box.h_p / 1024) * (1.2 / 16)
    # return [cen_poi_x_0/100, cen_poi_y_0/100, w/100, h/100, box.angle]

def take_from_omer():
    d = 520
    return d

# im1 = 'path to captured image indside cv2.imageread'
def take_picture_and_run(current_location, image_number):
    real_grapes = True
    d = g_param.const_dist
    box = [0,0,0,0,0]
    i = 0
    plt.clf()  # clean the canvas
    print(f"Picure number {i}")
    # ueye_take_picture_2(i)
    # take a picture by pressing a key
    # captured_image = take_picture()
    im1 = cv2.imread(r"C:\Drive\Mask_RCNN-master\samples\grape\dataset\train\DSC_0107.JPG") # change the path!!!!!!!!! that will fit the actual captured picture
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    img = im1


    # try_string = try_string.replace("num", str(i))
    # img = cv.imread('C:/Drive/28_7_20/grapes5.jpg')
    # img = cv.imread('C:/Drive/28_7_20/CDY_2015.jpg')

    # cv.imshow('bgrrr', img)
    # img = cv.imread('C:/Drive/28_7_20/try_2.jpeg')
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    ######## print to check if needed

    global TB
    cv.waitKey(0)
    cv.destroyAllWindows()
    im0 = img

    im0, window, scale, padding, crop = utils.resize_image(
        im0,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    img = im0
    if real_grapes: #TODO: add parameter "real_grapes" to know if working in lab/field
        arr = [im0]
        # use THE MASK R-CNN for real grapes: next 93 lines
        results = model.detect(arr, verbose=1)
        r = results[0]
        pred_masks = r["masks"]
        print("amount of grapes :", len(pred_masks[0][0]))
        if len(pred_masks[0][0]) > 0:
            im1 = im0
        else:
            print("There was no grapes detected in the capturd image.")
            im1 = cv.imread('C:/Drive/28_7_20/DSC_0210.JPG')
            im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)
            im1, window, scale, padding, crop = utils.resize_image(
                im1,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)
            arr = [im1]
            results = model.detect(arr, verbose=1)
        r = results[0]
        ax = get_ax(1)
        bbox = utils.extract_bboxes(r['masks'])
        bbox = bbox.astype('int32')
        pred_boxes = r["rois"]
        pred_class_ids = r["class_ids"]
        pred_scores = r["scores"]
        pred_masks = r["masks"]
        amount_of_mask_detacted = len(pred_masks[0][0])

        # display the picture with the masks, Bounding boxes, scores
        visualize.display_instances(im1, bbox, r['masks'], r['class_ids'],
                                    dataset_test.class_names, r['scores'], ax=ax,
                                    title="Predictions", show_bbox=False)

        # amount_of_mask_detacted = len(pred_masks[0][0])
        # print(amount_of_mask_detacted)
        # mask_index = calcs.biggest_mask_index(pred_masks)
        # print(mask_index)
        # if mask_index != None:
        predicted_masks_to_mrbb = []
        # Prediction
        for c in range(0, amount_of_mask_detacted):
            predicted_masks_to_mrbb.append(pred_masks[:, :, c])

        # calc the rotated bounding box for the biggest mask
        p_convex_hulls, p_npas = masks_to_convex_hulls(predicted_masks_to_mrbb)
        mrbbs_prediction = output_dict(p_npas)

        amount_of_mask_detacted = len(boxes)
        for i in range(0, amount_of_mask_detacted):
            x = int(boxes[i][0][0])
            y = int(boxes[i][0][1])
            w, h, corners_in_meter = calculate_w_h(d, corner_points[i])
            a = int(boxes[i][2])
            box = [x, y, w, h, a, corners_in_meter]
            predicted_masks_to_mrbb.append(box)


        det_rotated_boxes = []
        for b in range(0, len(mrbbs_prediction)):
            cen_poi_x_0 = mrbbs_prediction[b]["center_point"][0]
            cen_poi_y_0 = mrbbs_prediction[b]["center_point"][1]
            width_0 = mrbbs_prediction[b]["width"]
            height_0 = mrbbs_prediction[b]["height"]
            angle_0 = mrbbs_prediction[b]["rot_angle"]
            det_box = [cen_poi_x_0, cen_poi_y_0, int(width_0), int(height_0), angle_0, predicted_masks_to_mrbb[b]]
            box_in_meter = pixel_2_meter(d, det_box)
            det_rotated_boxes.append(box_in_meter)
            grape = [box_in_meter[0], box_in_meter[1], box_in_meter[2], box_in_meter[3], box_in_meter[4],
                     predicted_masks_to_mrbb[b],  det_box, pred_masks, None]
            add_to_TB(grape)


        image_det = im1
        for d in range(0, len(mrbbs_prediction)):
            cor_1 = mrbbs_prediction[d]["corner_points"][0] # list of [x,y] cordinates (int/float?)
            cor_2 = mrbbs_prediction[d]["corner_points"][1]
            cor_3 = mrbbs_prediction[d]["corner_points"][2]
            cor_4 = mrbbs_prediction[d]["corner_points"][3]

            image_det = cv.line(image_det, tuple(cor_1), tuple(cor_2), (250, (0), 0), thickness=3)
            image_det = cv.line(image_det, tuple(cor_2), tuple(cor_3), (250, (0), 0), thickness=3)
            image_det = cv.line(image_det, tuple(cor_3), tuple(cor_4), (250, (0), 0), thickness=3)
            image_det = cv.line(image_det, tuple(cor_4), tuple(cor_1), (250, (0), 0), thickness=3)
        plt.imshow(image_det)
        plt.show()
        ima = Image.fromarray(image_det)
        aa_string = "C:/Drive/13_8_20/maskrcnn_num.jpeg"
        ima.save(aa_string)
    else:
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
            # rect = ((center_x,center_y),(width,height),angle)
            for i, c in enumerate(contours):
                minRect[i] = cv.minAreaRect(c)
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

            for i, c in enumerate(contours):
                color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                box = cv.boxPoints(minRect[i])
                box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                area = minRect[i][-2][0] * minRect[i][-2][0]
                tresh_size = 25000
                if area > tresh_size:
                    # TODO: להכפיל את הזוית ב-1
                    boxes.append(minRect[i])
                    cv.drawContours(green, [box], 0, color)
            cv.imshow('mask__', green)

        parser = argparse.ArgumentParser(
            description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
        parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
        args = parser.parse_args()
        img = cv.imread('C:/Drive/25_11_20/fake_grape_0.jpeg')
        # img = cv.resize(img, (1024, 692))  # Resize image
        cv.imshow("source_window", img)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        ## mask of green (36,25,25) ~ (86, 255,255)q
        mask = cv.inRange(hsv, (18, 40, 60), (48, 240, 220))
        ## slice the green
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        src_gray = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))
        source_window = 'Source'
        cv.namedWindow(source_window)
        cv.imshow(source_window, green)
        max_thresh = 255
        thresh = 100  # initial threshold
        cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        boxes = []
        thresh_callback(thresh)
        # print(len(boxes), boxes)
        boxes = set(boxes) # remove duplicates
        boxes = list(boxes)
        print(boxes)

        cv.waitKey(0)
        cv.destroyAllWindows()


        # rng.seed(12345)
        #
        # def thresh_callback(val):
        #     threshold = val
        #     ret, thresh = cv.threshold(src_gray, 50, 255, cv.THRESH_BINARY)
        #     kernel = np.ones((5, 5), np.uint8)
        #     thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        #     thresh = cv.dilate(thresh, kernel, iterations=2)
        #     thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        #     thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        #     thresh = cv.erode(thresh, kernel, iterations=2)
        #     canny_output = cv.Canny(thresh, threshold, threshold * 2)
        #     contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #     # Find the rotated rectangles and ellipses for each contour
        #     minRect = [None] * len(contours)
        #     # rect = ((center_x,center_y),(width,height),angle)
        #     for i, c in enumerate(contours):
        #         minRect[i] = cv.minAreaRect(c)
        #     drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        #     for i, c in enumerate(contours):
        #         color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        #         box = cv.boxPoints(minRect[i])
        #         box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        #         area = minRect[i][-2][0] * minRect[i][-2][0]
        #         tresh_size = 25000
        #         if area > tresh_size:
        #             # TODO: להכפיל את הזוית ב-1
        #             boxes.append(minRect[i])
        #             cv.drawContours(green, [box], 0, color)
        #     cv.imshow('image after filters with bounding boxes', green)
        #
        # parser = argparse.ArgumentParser(
        #     description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
        # parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
        # args = parser.parse_args()
        # img = cv.imread('C:/Drive/28_7_20/try_2.jpeg')
        # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # # values to recognize all grapes
        # # mask = cv.inRange(hsv, (36, 0, 26), (70, 235, 255))
        # # values for only "fake" grapes"
        # mask = cv.inRange(hsv, (18, 20, 55), (48, 252, 235))
        # ## slice the green
        # imask = mask > 0
        # green = np.zeros_like(img, np.uint8)
        # green[imask] = img[imask]
        # src_gray = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
        # src_gray = cv.blur(src_gray, (3, 3))
        # source_window = 'Source'
        # cv.namedWindow(source_window)
        # max_thresh = 255
        # thresh = 100  # initial threshold
        # cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        # boxes = []
        # thresh_callback(thresh)
        # print(boxes)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    ##################### commented ******************
        predicted_masks_to_mrbb = []
        amount_of_mask_detacted = len(boxes)
        for i in range(0, amount_of_mask_detacted):
            x = boxes[i][0][0]
            y = boxes[i][0][1]
            w = boxes[i][1][0]
            h = boxes[i][1][0]
            a = boxes[i][2]
            box = [x,y,w,h,a]
            predicted_masks_to_mrbb.append(box)

        det_rotated_boxes = []
        # for the new function
        for b in range(0, len(predicted_masks_to_mrbb)):
            cen_poi_x_0 = box[0]
            cen_poi_y_0 = box[1]
            width_0 = box[2]
            height_0 = box[3]
            angle_0 = box[4] * (-1)
            angle_0 = (angle_0*180)/pi
            det_box = [cen_poi_x_0, cen_poi_y_0, int(width_0), int(height_0), angle_0, None]
            box_in_meter = pixel_2_meter(d, det_box)
            det_rotated_boxes.append(box_in_meter)
            grape = [box_in_meter[0],box_in_meter[1],box_in_meter[2],box_in_meter[3],box_in_meter[4],None,  det_box]
            add_to_TB(grape)




        print("boxes", boxes)
        # using the TB
        box_index = sort_by_and_check_for_grapes('rect_size')
        print("box_index", box_index)
        # img = cv.circle(img, (box[0],box[1]), radius=5, color=(0, 0, 255), thickness=5)

        # cv2.imshow("contours4", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # TODO: take_from_omer()
        d_new = take_from_omer()
        d = 520  # distance to object
        if d_new is not None:
            d = d_new

        # TODO: Omer! this is the function that calculates pixel_2_meter.


        # if box_index is None:
        #     return None
        # return 0

if __name__ == '__main__':
    d = 520 # comment when running from other moudle (single proccess)
    take_picture_and_run()