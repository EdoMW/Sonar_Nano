import cv2
import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# image_path = r'C:\Drive\Mask_RCNN-master\samples\grape\dataset\test\DSC_0280.JPG' # FIXME- grape 3

"""
1) make sure loading process works by mask!!!
2) make all sonar functions
"""
img = np.load(r'D:\Users\NanoProject\experiments\exp_data_10_46\masks\0_2_10_4.npy')
print(img.shape)
# for i in range(3):
plt.imshow(img[:, :], cmap="gray")
plt.show()


#
# ROOT_DIR = os.path.abspath("C:/Drive/Mask_RCNN-master/")
# warnings.filterwarnings("ignore")
# sys.path.append(ROOT_DIR)  # To find local version of the library
#
#
#
# image_path = r'C:\Drive\Mask_RCNN-master\samples\grape\dataset\test\DSC_0214.JPG'
# from mrcnn import utils
# img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)



# resized = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
# print('Resized Dimensions : ', resized.shape)
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


print(img.shape)
resized = utils.resize_image(img, max_dim = 1024, mode="square")
print(len(resized[0]), len(resized[0][0]), len(resized))
print('Resized Dimensions : ', len(resized))
resized, *_ = np.asarray(resized)
print(type(resized))
print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey()
cv2.destroyAllWindows()
#
resized = utils.resize_image(img, max_dim = 1024, mode="square")
resized, *_ = np.asarray(resized)


# new_corner_points = []
# for elem in corner_points:
#     if elem not in new_corner_points:
#         new_corner_points.append(elem)
# corner_points = new_corner_points

# a = np.array([[1,2,3],[2,4,2]])
# print(len(a[0]))
# import itertools
# import os, re
# import csv
# import cv2 as cv
# a = np.array([1,2,3])
# a = np.concatenate([a, [0]], axis=0)
# print(a)
# p1, p2 ,p3 = a
# print(p1, p3, p2)


# print(a)
# print(np.linalg.norm(np.array([0, 0, 0]) - np.array([-0.566, -0.087, 0.775])))
#
# num_of_pixels = 4
# x_val = np.random.choice(1024, num_of_pixels, replace=False)
# y_val = np.random.choice(1024, num_of_pixels, replace=False)
# combined = np.vstack((x_val, y_val)).T
# image_path_1 = d'D:\Users\NanoProject\Images_for_work\black.jpg'
# # image_path_1 = d'D:\Users\NanoProject\Images_for_work\1_13_06_16.jpeg'
#
# img = cv.imread(image_path_1)
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# black = True
# for i in range(num_of_pixels):
#     ans = np.array_equal(rgb[combined[i][0], combined[i][1]], (0, 0, 0))
#     if not ans:
#         black = False
#         break
#
#

#
#
# print("check if true")
# print(True if False in [True, True] else False)
# #
# # def load_image_path(self):
# #     image_number = g_param.image_number
# #     directory = self.exp_date_time
# #     parent_dir = d'D:\Users\NanoProject'
# #     path = os.path.join(parent_dir, directory)
# #     path = os.path.join(path, 'rgb_images')
# #     locations_list = os.listdir(path)
# #     res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
# #     path = os.path.join(path, res[0])
# #     return path
# #
# #
# #
# #
# #
# # def check_image(image_path):
# #     """
# #     :param image_path: path to last image taken
# #     :return: True if real image was taken (not only black pixels)
# #     """
# #     x_val = np.random.choice(1024, num_of_pixels, replace=False)
# #     y_val = np.random.choice(1024, num_of_pixels, replace=False)
# #     combined = np.vstack((x_val, y_val)).T
# #     img = cv.imread(image_path)
# #     rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# #     for i in range(num_of_pixels):
# #         ans = np.all(rgb[combined[i]] == (0, 0, 0), axis=-1)
# #         if not ans:
# #             return True
# #     else:
# #         return False
# #
# #
# #
# #
# #
# # image_path = ueye_take_picture_2(image_number)
# # for i in range(amount_of_tries):
# #     image_path = ueye_take_picture_2(image_number)
# #     image_taken = check_image(image_path)
# #     if image_taken:
# #         print(colored("Image taken successfully", 'green'))
# #         break
# #
# #
# #
#
#
# num_of_pixels = 4
# d = 5
# p1 = 15
# p2 = 13
# p3 = 12
# p4 = 16
# w = abs(p1 - p2)
# h = abs(p2 - p3)
# if w > h:
#     h, w = w, h
#     p1, p2, p3, p4 = p1, p4, p3, p2
# # without numpy :\
# import random
#
# # declaring list
# # pixel_list_x = [*range(0, 1024, 1)]
# # pixel_list_y = [*range(0, 1024, 1)]
# # x_val = random.sample(pixel_list_x, num_of_pixels)
# # y_val = random.sample(pixel_list_y, num_of_pixels)
# # print(x_val, y_val)
#
# # with numpy :)
# pixel_list_x = np.arange(0, 1024, 1)
# pixel_list_y = np.arange(0, 1024, 1)
# x_val = np.random.choice(1024, num_of_pixels, replace=False)
# y_val = np.random.choice(1024, num_of_pixels, replace=False)
#
# combined = np.vstack((x_val, y_val)).T
#
#
# def check_image(image_path):
#     """
#     :param image_path: path to last image taken
#     :return: True if real image was taken (not only black pixels)
#     """
#     img = cv.imread(image_path)
#     rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     for i in range(len(num_of_pixels)):
#         if not np.all(rgb[combined[:, i]] == (0, 0, 0), axis=-1):
#             return True
#     else:
#         return False
#
#
# image_path = d'D:\Users\NanoProject\Images_for_work\0_15_22_13.jpeg'
# # print(check_image(image_path))
#
#
# #
# # def check():
# #     if True:
# #         return
# #     print(hi)
# # check()
# #
# # a = np.array([[1, 2],[3,4]])
# # print(a)
# # a = np.insert(arr = a,
# #               obj = 2,
# #               values = 4,
# #               axis = 1)
# # print(a)
# # #
# # # b = [[1,2],[3,4],[5,6]]
# # # c = [["a","b"],["v", 'c'],["e","f"]]
# # # res = [list(itertools.chain(*i))
# # #        for i in zip(b, c)]
# # # print(res)
# # #
# # #
# # print("np")
# # for i in range(3):
# #     print(a)
# #     corner_list = a[i:i+1,:2]
# #     corn = corner_list.tolist()
# #     print(corn)
# #     print(type(corn))
#
#
# #
# # directory = 'exp_data_12_32'
# # parent_dir = d'D:\Users\NanoProject'
# # path = os.path.join(parent_dir, directory)
# # path = os.path.join(path, 'transformations')
# # print(path)
# # locations_list = os.listdir(path)
# # print(locations_list)
# # image_number = 0
# # res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
# # print(res)
# # path = os.path.join(path, res[0])
# # print(path)
# #
# # from numpy import genfromtxt
# #
# # my_data = np.genfromtxt(path, delimiter=",")
# # print(type(my_data))
# # print(my_data)