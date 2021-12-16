# import random
# import cv2 as cv
# import sys
# import os
# import warnings
#
# import matplotlib
import matplotlib.pyplot as plt
# from termcolor import colored
# import glob, shutil
import numpy as np
# from random import randint
# import colorsys
from PIL import Image

x_lim = (0, 1)
y_lim = (0, 1)
z_lim = (0, 1)
list_points = [[0.3, 0.3, 0],[0.4, 0.4, 0],[0,0,0],[.1, .1, .5], [0.3, 0.3, .2]]


def calc_single_axis_limits(axis):
    """
    calc the values and range between the 2 points that are the farthest away from each other on a single axis.
    :return: min and max value of values on this axis.
    """
    min_lim_x, max_lim_x = list_points[0][axis], list_points[0][axis]
    for grape_ind in range(len(list_points)):
        if list_points[grape_ind][axis] < min_lim_x:
            min_lim_x = list_points[grape_ind][axis]
        if list_points[grape_ind][axis] > max_lim_x:
            max_lim_x = list_points[grape_ind][axis]
    min_max_range = max_lim_x - min_lim_x
    # print('calc axis: ', float(min_lim_x - 0.5 * min_max_range), float(max_lim_x + 0.5 * min_max_range))
    if axis == 0:  # for x (distance) axis, for visualization only.
        return float(min_lim_x - 5.5 * min_max_range), float(max_lim_x + 2.5 * min_max_range)
    return float(min_lim_x - 0.5 * min_max_range), float(max_lim_x + 0.5 * min_max_range)


def calc_axis_limits():
    """
    Calculate the limits of each axis on the 3d plot
    """
    x_lim = calc_single_axis_limits(0)
    y_lim = calc_single_axis_limits(1)
    z_lim = calc_single_axis_limits(2)


def get_projections():
    """
    For each grape on the map (TB), add x,y,z coordinates to a list to be displayed on the 3D plot.
    :return:
    """
    x_list, y_list, z_list = [], [], []
    for item in range(len(list_points)):
        x_list.append(list_points[item][0])
        y_list.append(list_points[item][1])
        z_list.append(list_points[item][2])
    return x_list, y_list, z_list


def plot_tracking_map():
    """
    Visualize all grapes centers on a 3d map.
    This function generates a plot that represents the TB in 3D.
    """
    x_cors, y_cors, z_cors = [], [], []
    for i in range(len(list_points)):
        x_cor, y_cor, z_cor = list_points[i][0], list_points[i][1], list_points[i][2]
        x_cors.append(x_cor)
        y_cors.append(y_cor)
        z_cors.append(z_cor)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    # if len(list_points) < 5:
    #     calc_axis_limits()
    # ax.set_xlim(g_param.x_lim)
    # ax.set_ylim(g_param.y_lim)
    # ax.set_zlim(g_param.z_lim)

    # project each points on all planes.
    x, y, z = get_projections()
    ax.plot(x, z, '+', c='r', zdir='y', zs=y_lim[1])  # red pluses (+) on XZ plane
    ax.plot(y, z, 's', c='g', zdir='-x', zs=x_lim[0])  # red squares on YZ plane
    ax.plot(x, y, '*', c='b', zdir='-z', zs=z_lim[0])  # red pluses (*) on XY plane

    # labels titles
    ax.set_xlabel('X Label - distance')
    ax.set_ylabel('Y Label - advancement (moving from left to right)')
    ax.set_zlabel('Z Label - height')

    # change color of each plane
    ax.w_yaxis.set_pane_color((1.0, 0, 0.1))  # xy plane is red
    ax.w_xaxis.set_pane_color((0, 1.0, 0, 0.1))  # xy plane is green
    ax.w_zaxis.set_pane_color((0, 0, 1.0, 0.1))  # xy plane is blue

    # xx, yy = np.meshgrid([-3, 0, 3], [-3, 0, 3])
    # # zz = xx - yy
    # # yy = zz * 0.5
    # zz = xx - yy
    yy, zz = np.meshgrid(range(2), range(2))
    xx = yy


    s = ax.scatter(x_cors, y_cors, z_cors, s=400, marker='o')  # x,y,z coordinates, size of each point, colors.
    # controls the alpha channel. all points have the same value, ignoring their distance
    s.set_edgecolors = s.set_facecolors = lambda *args: None
    ax.title.set_text(f'Imgae number 1')
    elev = 20.0
    azim = 80.5
    ax.view_init(elev, azim)
    ax.plot_surface(xx, yy, zz, alpha=0.3)
    print(x, y, z)
    # ax.plot_trisurf(x, y, z)
    # ax.plot_surface(xx, yy, zz_1, alpha=0.5)
    plt.show()

plot_tracking_map()















# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from random import randrange
#
#
#
# img_ = cv2.imread(r'D:\Users\NanoProject\exp_data_13_46\rgb_images\original\0_13_46_54.jpeg')
# img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
# img = cv2.imread(r'D:\Users\NanoProject\exp_data_13_46\rgb_images\original\3_13_49_27.jpeg')
# img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
#
# # Apply ratio test
# good = []
# for m in matches:
#     if m[0].distance < 0.5*m[1].distance:
#         good.append(m)
# matches = np.asarray(good)
#
# if len(matches[:,0]) >= 4:
#     src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#     dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#     H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#     #print H
# else:
#     raise AssertionError("Can’t find enough keypoints.")
#
#
# dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
# plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
# plt.show()
# plt.figure()
# dst[0:img.shape[0], 0:img.shape[1]] = img
# cv2.imwrite('output.jpg', dst)
# plt.imshow(dst)
# plt.show()





# def show_images_with_masks():
#     def random_colors(N, bright=True):
#         """
#         Generate random colors.
#         To get visually distinct colors, generate them in HSV space then
#         convert to RGB.
#         """
#         brightness = 1.0 if bright else 0.7
#         hsv = [(i / N, 1, brightness) for i in range(N)]
#         colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#         random.shuffle(colors)
#         return colors
#
#
#     def apply_mask(image, mask, color, alpha=0.5):
#         """Apply the given mask to the image.
#         """
#         for c in range(3):
#             image[:, :, c] = np.where(mask == 1,
#                                       image[:, :, c] *
#                                       (1 - alpha) + alpha * color[c] * 255,
#                                       image[:, :, c])
#         return image
#
#     npz = np.load(r'D:\Users\NanoProject\old_experiments\exp_data_13_46\masks\38_14_22_28.npz')
#     image = cv.imread(r'D:\Users\NanoProject\old_experiments\exp_data_13_46\rgb_images\resized\38_14_22_28.jpeg')
#     masks = npz.f.arr_0
#     N = masks.shape[2]
#     colors = random_colors(N)
#     masked_image = image.astype(np.uint32).copy()
#     for i in range(N):
#         mask = masks[:, :, i]
#         masked_image = apply_mask(masked_image, mask, colors[i])
#     import matplotlib.pyplot as plt
#
#     auto_show = False
#     ax = None
#     if not ax:
#         _, ax = plt.subplots(1)
#         auto_show = True
#
#     # Generate random colors
#     colors = colors or random_colors(N)
#
#     # Show area outside image boundaries.
#     height, width = image.shape[:2]
#     ax.set_ylim(height + 10, -10)
#     ax.set_xlim(-10, width + 10)
#     ax.axis('off')
#     ax.set_title("aaa")
#     ax.imshow(masked_image.astype(np.uint8))
#     plt.show()
#
#
# # show_images_with_masks() # TODO: make it for entire dir, dir of dirs
# #aa
#
# path = r'D:\Users\NanoProject\old_experiments'
# dirs = os.listdir(path)
# count = 0
# distances = []
# for dir in dirs:
#     path_2 = os.path.join(path, dir)
#     path_3 = os.path.join(path_2, "sonar")
#     path_4 = os.path.join(path_3, "distances")
#     distance = os.listdir(path_4)
#     if len(distance) > 0:
#         distances.append(dir)
#         count += 1
#
# print(count, distances)
#
#
#
#
#
#
# # p = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\masks\38_14_22_28.npz'
# # # p = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\masks\37_14_21_09.npz'
# # npz = np.load(p)
# # img_array = npz.f.arr_0[:,:,:]
# # img_array = img_array.astype('uint8')
# # print(type(img_array[1,1,0]))
# # np.savez_compressed(p, img_array)
# # im = Image.fromarray(img_array)
# #
# # # this might fail if `img_array` contains a data type that is not supported by PIL,
# # # in which case you could try casting it to a different dtype e.g.:
# # # im = Image.fromarray(img_array.astype(np.uint8))
# #
# # im.show()
#
#
#
# #
# #
# # images = os.listdir(r'D:\Users\NanoProject\old_experiments\exp_data_13_46\rgb_images\resized')
# # npzs_without = os.listdir(r'D:\Users\NanoProject\old_experiments\exp_data_13_46\masks')
# # npzs_without = [x.split('_')[0] for x in npzs_without]
# # npzs_without_int = [int(x.split('_')[0]) for x in npzs_without]
# # # npzs_without_int = sorted(npzs_without_int)
# # # npzs_without_int_str = [str(x) for x in npzs_without_int]
# # images = [x for x in images if x.split('_')[0] in npzs_without]
# #
# # npzs = os.listdir(r'D:\Users\NanoProject\old_experiments\exp_data_13_46\masks')
# # pairs = [list(x) for x in zip(npzs, images, npzs_without_int)]
# # import operator
# # pairs = sorted(pairs, key = lambda x: x[2])
# # print(pairs)
# #
# # for i in range(len(npzs)):
# #     path = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\masks'
# #     npz_path = os.path.join(path, pairs[i][0])
# #     mask_npz = np.load(npz_path)
# #     mask = mask_npz.f.arr_0
# #     path = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\rgb_images\resized'
# #     image_path = os.path.join(path, pairs[i][1])
# #     image = cv.imread(image_path)
# #     image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
# #     redImg = np.zeros(image.shape, image.dtype)
# #     redImg[:,:] = (0, 0, 255)
# #     redMask = cv.bitwise_and(redImg, redImg, mask=mask)
# #     cv.addWeighted(redMask, 1, image, 1, 0, image)
# #     cv.imshow(f'{pairs[i][2]}', image)
# #     cv.waitKey()
#
# # image = image_path
# # im = image.copy()
# # print(image.shape)
# # print(mask.shape)
# # non_zeros = []
# # for i in range(len(mask[0][0] + 1)):
# #     non_zeros.append(np.count_nonzero(mask[:,:,i]))
# # min_index = non_zeros.index(max(non_zeros))
# # for i in range(len(mask[0][0] + 1)):
# #     # if i != min_index:
# #     rgb = (randint(0,255), randint(0,255), randint(0,255))
# #     mask_temp = mask[:,:,i].copy()
# #     image[mask_temp==1] = rgb
# # plt.figure(figsize=(12,8))
# # plt.imshow(im)
# # plt.imshow(image, 'gray', interpolation='none', alpha=alpha, vmin = 0.8) # alpha: [0-1], 0 is 100% transperancy, 1 is 0%
# # plt.show()
#
# #
#
#
#
# # ----------------Edo: checking speeds-------------------------
# #
# # # count amount of images
# # count = 0
# # path = r'C:\Users\Administrator\Desktop\experiments_backup\old_experiments'
# # path_images = r'C:\Users\Administrator\Desktop\experiments_backup\images'
# # single_count = 0
# # exp_lists = os.listdir(path)
# # for count, exp in enumerate(exp_lists):
# #     exp_path = os.path.join(path, exp)
# #     exp_path_images = exp_path + r'\rgb_images\original'
# #     images_list = os.listdir(exp_path_images)
# #     for i in range(len(images_list)):
# #         image = os.path.join(exp_path_images, images_list[i])
# #         shutil.copy(image, path_images)
# #         single_count += 1
# #     count += len(images_list)
# #     print(len(images_list))
# # print("count:", count)
# # print("single_count:", single_count)
# #
# #
#
#
# # ----------------Omer: checking speeds-------------------------
#
#
# import write_to_socket
# import read_from_socket
#
#
#
# # v = 0.35
# # V = [0.5, 0.4, 0.3, 0.2]
# # T = [3, 5, 5, 7]
# # t_sleep = 7
# # rs_rob = read_from_socket.ReadFromRobot()
# # ws_rob = write_to_socket.Send2Robot()
# # start_pos = np.array([-0.39, 0.035, 0.3, -1.144, -1.035, -0.128])
# # ws_rob.move_command(False, start_pos, t_sleep, v)
# # for i in range(12):
# #     location = np.asarray(rs_rob.read_tcp_pos())
# #     location[2] += 0.25
# #     ws_rob.move_command(False, location, T[i], V[i])
# #     location[2] += -0.25
# #     ws_rob.move_command(False, location, T[i], V[i])
# #     print(f"i: {i} " f"speed: {V[i]}")
#
# # ------------------------------------------------------------
#
# # print("Real grape?")
# # one = "\033[1m" + "0" + "\033[0m"
# # zero = "\033[1m" + "1" + "\033[0m"
# # a = [(500, 500)] * 12
# # # real_grape = input(colored("Yes: ", "cyan") + "press " + one + colored(" No: ", "red") + "Press " + zero + " ")
# # print(a[-6:])
# #
#
# # a = [[1],[3],[2]]
# # b = [[1],[3],[5],[2]]
# # a = a + b
# # print(a)
# # import itertools
# # # a = [['a','b'], ['c']]
# # print(list(itertools.chain.from_iterable(a)))
# # #
# #
# #
# # a = np.zeros((2,10))
# # for i in range(2):
# #     for j in range(10):
# #         if j % 2 == 0:
# #             if i % 2 == 1:
# #                 a[i][j] = j * 2
# #             else:
# #                 a[i][j] = j*2 + 1
# #         else:
# #             if i % 2 == 0:
# #                 a[i][j] = j * 2
# #             else:
# #                 a[i][j] = j * 2 + 1
# #
# # def get_index(index):
# #     """
# #     :param index: index of current image
# #     :return: low image idx, high image idx
# #     """
# #     if index % 2 == 0:
# #         lpi_temp = index * 2
# #         hpi_temp = lpi_temp + 1
# #     else:
# #         lpi_temp = index * 2 + 1
# #         hpi_temp = lpi_temp - 1
# #     return lpi_temp, hpi_temp
# #
# #
# # def build_array(step_size):
# #     """
# #     builds array to take image from
# #     :param step_size:
# #     :return:
# #     """
# #     direction = 0  # even = up, odd = down
# #     b = []
# #     for i in range(0, 10, step_size):
# #         lpi, hpi = get_index(i)
# #         if direction % 2 == 0:
# #             b.append(lpi)
# #             b.append(hpi)
# #         else:
# #             b.append(hpi)
# #             b.append(lpi)
# #         direction += 1
# #     return b
# #
# #
# # def get_image_num(imgae_num, step):
# #     b = build_array(step)
# #     return b[imgae_num]
# #
# # step_direction = ["right", "up", "right", "down"]  # the order of movement
# #
# # for i in range(10):
# #     num = get_image_num(i, 1)
# #     direction = step_direction[(num + 1) % 4]
# #     print(num, direction)
#
#
#
#
#
# #
# # from time import time
# # start_time = time()
# #
# # get_image_num(imgae_num=4 , step=1)
# # print("--- %s seconds ---" % (time() - start_time))
# # point1 = [[100,200],[200,300]]
# # point2 = [[100,201],[202,301]]
# #
# # a = np.isclose(point1, point2, atol=1.01)
# # ans = np.all(a)
# #
# # print(round(233, -1))
# # print(np.all(a))
# #
# # x = [[100, 200], [200, 300], [1000], [23, [123, 223]]]
# # c = np.hstack(x)
# # print(type(c))
# # print(c)
#
#
#
# # image_path = r'C:\Drive\Mask_RCNN-master\samples\grape\dataset\test\DSC_0280.JPG' #
# # # img = np.load(r'D:\Users\NanoProject\experiments\exp_data_10_46\masks\0_2_10_4.npy')
# # print(img.shape)
# # resized = utils.resize_image(img, max_dim = 1024, mode="square")
# # print(len(resized[0]), len(resized[0][0]), len(resized))
# # print('Resized Dimensions : ', len(resized))
# # resized, *_ = np.asarray(resized)
# # print(type(resized))
# # print('Resized Dimensions : ', resized.shape)
# #
# # cv2.imshow("Resized image", resized)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
#
#
#
# """
# 1) make sure loading process works by mask!!!
# 2) make all sonar functions
# """
# # img = np.load(r'D:\Users\NanoProject\experiments\exp_data_10_46\masks\0_2_10_4.npy')
# # print(img.shape)
# # # for i in range(3):
# # plt.imshow(img[:, :], cmap="gray")
# # plt.show()
#
#
# #
# # ROOT_DIR = os.path.abspath("C:/Drive/Mask_RCNN-master/")
# # warnings.filterwarnings("ignore")
# # sys.path.append(ROOT_DIR)  # To find local version of the library
# #
# #
# #
# # image_path = r'C:\Drive\Mask_RCNN-master\samples\grape\dataset\test\DSC_0214.JPG'
# # from mrcnn import utils
# # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#
#
#
# # resized = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
# # print('Resized Dimensions : ', resized.shape)
# # cv2.imshow("Resized image", resized)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# #
# # print(img.shape)
# # resized = utils.resize_image(img, max_dim = 1024, mode="square")
# # print(len(resized[0]), len(resized[0][0]), len(resized))
# # print('Resized Dimensions : ', len(resized))
# # resized, *_ = np.asarray(resized)
# # print(type(resized))
# # print('Resized Dimensions : ', resized.shape)
# #
# # cv2.imshow("Resized image", resized)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
# # #
# # resized = utils.resize_image(img, max_dim = 1024, mode="square")
# # resized, *_ = np.asarray(resized)
#
#
# # new_corner_points = []
# # for elem in corner_points:
# #     if elem not in new_corner_points:
# #         new_corner_points.append(elem)
# # corner_points = new_corner_points
#
# # a = np.array([[1,2,3],[2,4,2]])
# # print(len(a[0]))
# # import itertools
# # import os, re
# # import csv
# # import cv2 as cv
# # a = np.array([1,2,3])
# # a = np.concatenate([a, [0]], axis=0)
# # print(a)
# # p1, p2 ,p3 = a
# # print(p1, p3, p2)
#
#
# # print(a)
# # print(np.linalg.norm(np.array([0, 0, 0]) - np.array([-0.566, -0.087, 0.775])))
# #
# # num_of_pixels = 4
# # x_val = np.random.choice(1024, num_of_pixels, replace=False)
# # y_val = np.random.choice(1024, num_of_pixels, replace=False)
# # combined = np.vstack((x_val, y_val)).T
# # image_path_1 = d'D:\Users\NanoProject\Images_for_work\black.jpg'
# # # image_path_1 = d'D:\Users\NanoProject\Images_for_work\1_13_06_16.jpeg'
# #
# # img = cv.imread(image_path_1)
# # rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# # black = True
# # for i in range(num_of_pixels):
# #     ans = np.array_equal(rgb[combined[i][0], combined[i][1]], (0, 0, 0))
# #     if not ans:
# #         black = False
# #         break
# #
# #
#
# #
# #
# # print("check if true")
# # print(True if False in [True, True] else False)
# # #
# # # def load_image_path(self):
# # #     image_number = g_param.image_number
# # #     directory = self.exp_date_time
# # #     parent_dir = d'D:\Users\NanoProject'
# # #     path = os.path.join(parent_dir, directory)
# # #     path = os.path.join(path, 'rgb_images')
# # #     locations_list = os.listdir(path)
# # #     res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
# # #     path = os.path.join(path, res[0])
# # #     return path
# # #
# # #
# # #
# # #
# # #
# # # def check_image(image_path):
# # #     """
# # #     :param image_path: path to last image taken
# # #     :return: True if real image was taken (not only black pixels)
# # #     """
# # #     x_val = np.random.choice(1024, num_of_pixels, replace=False)
# # #     y_val = np.random.choice(1024, num_of_pixels, replace=False)
# # #     combined = np.vstack((x_val, y_val)).T
# # #     img = cv.imread(image_path)
# # #     rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# # #     for i in range(num_of_pixels):
# # #         ans = np.all(rgb[combined[i]] == (0, 0, 0), axis=-1)
# # #         if not ans:
# # #             return True
# # #     else:
# # #         return False
# # #
# # #
# # #
# # #
# # #
# # # image_path = ueye_take_picture_2(image_number)
# # # for i in range(amount_of_tries):
# # #     image_path = ueye_take_picture_2(image_number)
# # #     image_taken = check_image(image_path)
# # #     if image_taken:
# # #         print(colored("Image taken successfully", 'green'))
# # #         break
# # #
# # #
# # #
# #
# #
# # num_of_pixels = 4
# # d = 5
# # p1 = 15
# # p2 = 13
# # p3 = 12
# # p4 = 16
# # w = abs(p1 - p2)
# # h = abs(p2 - p3)
# # if w > h:
# #     h, w = w, h
# #     p1, p2, p3, p4 = p1, p4, p3, p2
# # # without numpy :\
# # import random
# #
# # # declaring list
# # # pixel_list_x = [*range(0, 1024, 1)]
# # # pixel_list_y = [*range(0, 1024, 1)]
# # # x_val = random.sample(pixel_list_x, num_of_pixels)
# # # y_val = random.sample(pixel_list_y, num_of_pixels)
# # # print(x_val, y_val)
# #
# # # with numpy :)
# # pixel_list_x = np.arange(0, 1024, 1)
# # pixel_list_y = np.arange(0, 1024, 1)
# # x_val = np.random.choice(1024, num_of_pixels, replace=False)
# # y_val = np.random.choice(1024, num_of_pixels, replace=False)
# #
# # combined = np.vstack((x_val, y_val)).T
# #
# #
# # def check_image(image_path):
# #     """
# #     :param image_path: path to last image taken
# #     :return: True if real image was taken (not only black pixels)
# #     """
# #     img = cv.imread(image_path)
# #     rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# #     for i in range(len(num_of_pixels)):
# #         if not np.all(rgb[combined[:, i]] == (0, 0, 0), axis=-1):
# #             return True
# #     else:
# #         return False
# #
# #
# # image_path = d'D:\Users\NanoProject\Images_for_work\0_15_22_13.jpeg'
# # # print(check_image(image_path))
# #
# #
# # #
# # # def check():
# # #     if True:
# # #         return
# # #     print(hi)
# # # check()
# # #
# # # a = np.array([[1, 2],[3,4]])
# # # print(a)
# # # a = np.insert(arr = a,
# # #               obj = 2,
# # #               values = 4,
# # #               axis = 1)
# # # print(a)
# # # #
# # # # b = [[1,2],[3,4],[5,6]]
# # # # c = [["a","b"],["v", 'c'],["e","f"]]
# # # # res = [list(itertools.chain(*i))
# # # #        for i in zip(b, c)]
# # # # print(res)
# # # #
# # # #
# # # print("np")
# # # for i in range(3):
# # #     print(a)
# # #     corner_list = a[i:i+1,:2]
# # #     corn = corner_list.tolist()
# # #     print(corn)
# # #     print(type(corn))
# #
# #
# # #
# # # directory = 'exp_data_12_32'
# # # parent_dir = d'D:\Users\NanoProject'
# # # path = os.path.join(parent_dir, directory)
# # # path = os.path.join(path, 'transformations')
# # # print(path)
# # # locations_list = os.listdir(path)
# # # print(locations_list)
# # # image_number = 0
# # # res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
# # # print(res)
# # # path = os.path.join(path, res[0])
# # # print(path)
# # #
# # # from numpy import genfromtxt
# # #
# # # my_data = np.genfromtxt(path, delimiter=",")
# # # print(type(my_data))
# # # print(my_data)