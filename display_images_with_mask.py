"""
Displaying images with masks on top of them of the experiment
"""

import random

import cv2 as cv
import sys
import os
import warnings
import matplotlib.pyplot as plt
from termcolor import colored
import glob, shutil
import numpy as np
from random import randint

# Next 2 lines are for the test set
masks_path = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\masks'
img_path = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\rgb_images\resized'


old_exp_path = r'D:\Users\NanoProject\old_experiments'
old_exps = os.listdir(old_exp_path)

for dir in old_exps:
    dir = os.path.join(old_exp_path, dir)
    dir_masks = os.path.join(dir, "masks")
    if len(os.listdir(dir_masks)) > 0:
        print(dir)
        masks_path = dir_masks
        img_path = os.path.join(dir, r"rgb_images\resized")
        images = os.listdir(img_path)
        npzs_without = os.listdir(masks_path)
        npzs_without = [x.split('_')[0] for x in npzs_without]
        npzs_without_int = [int(x.split('_')[0]) for x in npzs_without]
        images = [x for x in images if x.split('_')[0] in npzs_without]

        image_list = []
        npzs = os.listdir(masks_path)
        for i in range(len(npzs)):
            image_list.append([x for x in images if x.split('_')[0] in npzs[i].split('_')[0]])
        image_list = sum(image_list, [])
        pairs = [list(x) for x in zip(npzs, image_list, npzs_without_int)]

        pairs = sorted(pairs, key = lambda x: x[2])
        print(pairs)

        for i in range(len(os.listdir(dir_masks))):
            print(len(os.listdir(dir_masks)))
            path = masks_path
            npz_path = os.path.join(path, pairs[i][0])
            mask_npz = np.load(npz_path)
            mask = mask_npz.f.arr_0
            path = img_path
            image_path = os.path.join(path, pairs[i][1])
            image = cv.imread(image_path)
            # image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
            redImg = np.zeros(image.shape, image.dtype)
            redImg[:,:] = (0, 0, 255)
            redMask = cv.bitwise_and(redImg, redImg, mask=mask)
            cv.addWeighted(redMask, 1, image, 1, 0, image)
            cv.imshow(f'{pairs[i][2]}', image)
            cv.waitKey()
            cv.destroyAllWindows()


"""
For displaying only one exp at a time
"""
# masks_path = r'D:\Users\NanoProject\old_experiments\exp_data_11_06\masks'
# img_path = r'D:\Users\NanoProject\old_experiments\exp_data_11_06\rgb_images\resized'
# images = os.listdir(img_path)
# npzs_without = os.listdir(masks_path)
# npzs_without = [x.split('_')[0] for x in npzs_without]
# npzs_without_int = [int(x) for x in npzs_without]
# # npzs_without_int = sorted(npzs_without_int)
# # npzs_without_int_str = [str(x) for x in npzs_without_int]
# images = [x for x in images if x.split('_')[0] in npzs_without]
#
# image_list = []
# npzs = os.listdir(masks_path)
# for i in range(len(npzs)):
#     image_list.append([x for x in images if x.split('_')[0] in npzs[i].split('_')[0]])
# image_list = sum(image_list, [])
# print("image_list", image_list)
# print(npzs_without)
# print(images)
#
# pairs = [list(x) for x in zip(npzs, image_list, npzs_without_int)]
# import operator
# pairs = sorted(pairs, key = lambda x: x[2])
# print(pairs)
#
# for i in range(len(npzs)):
#     path = masks_path
#     npz_path = os.path.join(path, pairs[i][0])
#     mask_npz = np.load(npz_path)
#     mask = mask_npz.f.arr_0
#     path = img_path
#     image_path = os.path.join(path, pairs[i][1])
#     image = cv.imread(image_path)
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     redImg = np.zeros(image.shape, image.dtype)
#     redImg[:,:] = (0, 0, 255)
#     redMask = cv.bitwise_and(redImg, redImg, mask=mask)
#     cv.addWeighted(redMask, 1, image, 1, 0, image)
#     cv.imshow(f'{pairs[i][2]}', image)
#     cv.waitKey()
#     cv.destroyAllWindows()
