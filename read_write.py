import csv
import os
import g_param
from g_param import get_image_num_sim
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import shutil
import io
import pandas as pd
from time import sleep
from pprint import pprint
np.set_printoptions(precision=3)

take_last_exp = True  # take the exp that was conducted the latest


def get_local_time_date():  # TODO- make it work
    """
    :return: hours_min_sec_millisec
    """
    now = datetime.now()
    return now.strftime("%size_of_step/%d/%Y, %H:%M:%S")


def get_local_time_4():
    """
    :return: hours_min_sec_millisec
    """
    # Hours_minutes_seconds_milliseconds
    return datetime.now().strftime("%H_%M_%S_%f")[:-4]


def get_local_time_3():
    """
    :return: hours_min_sec
    """
    return datetime.now().strftime("%H_%M_%S")


def get_local_time_2():
    """
    :return: hours_min
    """
    return datetime.now().strftime("%H_%M")


def get_latest_dir():
    # return 0  # FIXME
    """
    if take_last_exp = True (default) it will return the last exp dir. AND
    :return: dir path of the exp to be analysed
    """
    if take_last_exp and g_param.process_type == "record":
        directory = r'D:\Users\NanoProject\experiments'
        return max([os.path.join(directory, d) for d in os.listdir(directory)], key=os.path.getmtime)
    else:
        print("Experiments dir is empty. taking last exp from old experiments")
        directory = r'D:\Users\NanoProject\old_experiments'
        return max([os.path.join(directory, d) for d in os.listdir(directory)], key=os.path.getmtime)


def generate_str_num_mask_dt(mask_id):
    current_time = get_local_time_3()
    path = 'num_maskNum_dt.csv'
    path = path.replace("num", str(g_param.image_number))
    path = path.replace("maskNum", str(mask_id))
    path = path.replace("dt", str(current_time))
    return path


def write_to_csv(path, data):
    """
    write data to csv file
    :param path: where to save the data
    :param data: data to be saved
    :return:
    """
    file = open(path, 'w+')
    with file:
        out = csv.writer(file)
        out.writerows(map(lambda x: [x], data))
        file.close()


def write_txt(mask_directory_path):
    """
    create a txt file with the name: 'Working in lab- no masks'
    :param mask_directory_path: path to save the txt file
    """
    mask_directory_path = mask_directory_path + r'\Working in lab- no masks.txt'
    with open(mask_directory_path, "w") as file:
        file.write("")


def write_txt_config(sim_directory_path):
    """
    create a txt file with the name: 'Working in lab- no masks'
    :param sim_directory_path: path to save the txt file
    """
    path = sim_directory_path + r'\config.csv'
    avg_dist = g_param.avg_dist
    height_step_size = g_param.height_step_size
    platform_step_size = g_param.platform_step_size
    resolution = g_param.platform_step_size
    image_cnn_path = g_param.image_cnn_path
    cnn_config = g_param.cnn_config
    horizontal_step_size = g_param.step_size
    param_list = [avg_dist, horizontal_step_size, str(round(height_step_size*horizontal_step_size, 2)),
                  platform_step_size, resolution, image_cnn_path]
    param_list_name = ["avg_dist", "horizontal_step_size", "height_step_size",
                       "platform_step_size", "resolution", "image_cnn_path"]
    headlines = ["Network configuration", "Running configuration"]
    with open(path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        cnn_used = False  # TODO- add to exp name real/fake
        if cnn_used:
            writer.writerow([headlines[0]])
            for key, value in cnn_config.items():
                writer.writerow([key, value])
        writer.writerow([headlines[1]])
        writer.writerows(zip(param_list_name, param_list))
        csv_file.close()
    text = open(path, "r")
    text = ''.join([i for i in text]).replace(", ", ": ")
    text = ''.join([i for i in text]).replace(",", ": ")
    x = open(path, "w")
    x.writelines(text)
    x.close()


def write_np_compressed(mask_path, mask):
    """
    saves both npy, npz files of the mask
    :param mask_path: path to save the masks at
    :param mask: np nd
    :return:
    """
    npz_path = mask_path[0:-8] + '.npz'
    np.savez_compressed(npz_path, mask)
    # mask_path = mask_path[0:-8] + '.npy'
    # np.save(mask_path, mask)
    # npz_path = mask_path[:-1] + 'z'
    # np.savez_compressed(npz_path, mask)


def calc_image_number():
    """
    calc image number to load in "load" mode
    """
    image_number = g_param.image_number
    skip = g_param.steps_gap
    if image_number % 4 == 2 or image_number % 4 == 3:
        pass


def move_old_directory():
    """
    Empty experiments, simulations directories
    if error occur- change manuel the name of folder with same HH:MM in old_experiments dir
    """
    source_dir = r'D:\Users\NanoProject\experiments'
    target_dir = r'D:\Users\NanoProject\old_experiments'
    file_names = os.listdir(source_dir)
    if len(file_names) != 0:
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), target_dir)
    source_dir = r'D:\Users\NanoProject\simulations'
    target_dir = r'D:\Users\NanoProject\old_simulations'
    file_names = os.listdir(source_dir)
    if len(file_names) != 0:
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), target_dir)


def read_3d_text(path_to_file):
    """
    :param path_to_file: s record as string
    :return: s, as 2D np array (array consists of flatten 2d arrays)
    """
    b = pd.read_csv(path_to_file, header=None)
    first_item = True
    list_of_2d = []
    for k in range(len(b)):
        chunk = b.iloc[k, 0]
        chank_list = chunk.split('\n')
        chank_list = [x.replace('[', '').replace(']', '') for x in chank_list]
        for i in range(30):
            chank_list[i] = chank_list[i].split(' ')
            for j in range(0, 7):
                if len(chank_list[i][j]) > 2:
                    if chank_list[i][j].endswith('\r'):
                        chank_list[i][j] = chank_list[i][j][:-2]
                if i > 0 and first_item:
                    chank_list[i] = chank_list[i][1:]
                    first_item = False
                chank_list[i][j] = float(chank_list[i][j])
            chank_list[i] = np.array(chank_list[i])
            first_item = True
        list_of_2d.append(np.array(chank_list).flatten())
    return np.array(list_of_2d)



class ReadWrite:
    """
    For read and write functions. save the exp data in real time to be redone later.
    """
    def __init__(self):
        self.location_path = None
        self.simulations = None
        self.rgb_images_path = None
        self.masks_path = None
        self.sonar_path = None
        self.distances = None
        self.classes = None
        self.class_sonar_path = None
        self.dist_sonar_path = None
        self.s_dist_sonar_path = None
        self.t_dist_sonar_path = None
        self.transformations_path = None
        self.platform_path = None
        self.TB_path = None
        self.transformations_path_ang_vec_tcp = None
        self.transformations_path_t_tcp2base = None
        self.transformations_path_t_cam2base = None
        self.transformations_path_t_cam2world = None
        self.rgb_image_resized = None
        self.rgb_image_orig = None
        self.rgb_image_manual = None
        self.exp_date_time = get_latest_dir()

    def create_directory(self):
        """
        create the relevant directory
        """
        if g_param.process_type == "record":
            self.create_directories()
            return
        if g_param.process_type == "load":
            self.create_sim_directory()
            return
        return

    def create_sim_directory(self):
        # move_old_directory()
        directory = "exp_data_dataTime"
        directory = directory.replace("dataTime", get_local_time_2())
        parent_dir_sim = r'D:\Users\NanoProject\simulations'
        path_sim = os.path.join(parent_dir_sim, directory)
        os.mkdir(path_sim)
        self.simulations = path_sim

    def create_directories(self):
        # FIXME uncomment next 2 lines
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        """
        Creates all directories and sub directories
        """
        move_old_directory()
        directory = "exp_data_dataTime"
        directory = directory.replace("dataTime", get_local_time_2())
        parent_dir = r'D:\Users\NanoProject\experiments'
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

        directory_1, directory_2, directory_3 = "locations", "rgb_images", "masks"
        sub_dir_sonar_1, sub_dir_sonar_2, sub_dir_sonar_3, sub_dir_sonar_4 = "class_sonar", "dist_sonar", "distances", "classes"
        sub_dir_sonar_2_1, sub_dir_sonar_2_2 = "s_dist_sonar", "t_dist_sonar"
        directory_4, directory_5, directory_6, directory_7 = "sonar", "transformations", "platform", "TB"
        sub_dir_1, sub_dir_2, sub_dir_3, sub_dir_4 = "ang_vec_tcp", "t_tcp2base", "t_cam2base", "t_cam2world"
        sub_dir_img_1, sub_dir_img_2, sub_dir_img_3 = "resized", "original", "manual"

        path_1 = os.path.join(path, directory_1)
        path_2 = os.path.join(path, directory_2)
        path_3 = os.path.join(path, directory_3)
        path_4 = os.path.join(path, directory_4)
        path_5 = os.path.join(path, directory_5)
        path_6 = os.path.join(path, directory_6)
        path_7 = os.path.join(path, directory_7)

        path_2_1 = os.path.join(path_2, sub_dir_img_1)
        path_2_2 = os.path.join(path_2, sub_dir_img_2)
        path_2_3 = os.path.join(path_2, sub_dir_img_3)

        path_4_1 = os.path.join(path_4, sub_dir_sonar_1)
        path_4_2 = os.path.join(path_4, sub_dir_sonar_2)
        path_4_2_1 = os.path.join(path_4_2, sub_dir_sonar_2_1)  # s
        path_4_2_2 = os.path.join(path_4_2, sub_dir_sonar_2_2)  # t
        path_4_3 = os.path.join(path_4, sub_dir_sonar_3)  # distance and real distance
        path_4_4 = os.path.join(path_4, sub_dir_sonar_4)  # class and real class

        path_5_1 = os.path.join(path_5, sub_dir_1)
        path_5_2 = os.path.join(path_5, sub_dir_2)
        path_5_3 = os.path.join(path_5, sub_dir_3)
        path_5_4 = os.path.join(path_5, sub_dir_4)

        folders = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_2_1, path_2_2, path_2_3, path_4_1,
                   path_4_2, path_4_3, path_4_4, path_4_2_1, path_4_2_2, path_5_1, path_5_2, path_5_3, path_5_4]
        for folder in folders:
            os.mkdir(folder)
        self.location_path = path_1
        self.rgb_images_path = path_2
        self.rgb_image_resized = path_2_1
        self.rgb_image_orig = path_2_2
        self.rgb_image_manual = path_2_3
        self.masks_path = path_3
        self.sonar_path = path_4
        self.distances = path_4_3
        self.classes = path_4_4
        self.transformations_path = path_5
        self.platform_path = path_6
        self.TB_path = path_7
        self.class_sonar_path = path_4_1
        self.dist_sonar_path = path_4_2
        self.s_dist_sonar_path = path_4_2_1
        self.t_dist_sonar_path = path_4_2_2
        self.transformations_path_ang_vec_tcp = path_5_1
        self.transformations_path_t_tcp2base = path_5_2
        self.transformations_path_t_cam2base = path_5_3
        self.transformations_path_t_cam2world = path_5_4

    def create_simulation_config_file(self):
        """
        Create a txt file with the configurations used for this simulation
        """
        path_sim = self.simulations
        write_txt_config(path_sim)

    def save_mask(self, mask, mask_id):
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        if g_param.work_place == 'lab':
            write_txt(self.masks_path)
            return
        mask_parent_path = self.masks_path
        mask_path = generate_str_num_mask_dt(mask_id)
        mask_path = os.path.join(mask_parent_path, mask_path)
        write_np_compressed(mask_path, mask)

    def save_manual_image(self, original_frame):
        """
        :param original_frame: frame in original resolution
        :return:
        """
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_for_manual_images = self.rgb_image_manual
        img_name = 'num_dt.jpeg'
        img_name = img_name.replace("num", str(g_param.image_number))
        img_name = img_name.replace("dt", str(get_local_time_3()))
        image_path_manual = os.path.join(folder_path_for_manual_images, img_name)
        plt.imsave(image_path_manual, original_frame)
        plt.clf()  # clears figure
        plt.close()  # closes figure

    def save_rgb_image(self, frame, original_frame):
        """
        :param original_frame: frame in original resolution
        :param frame: The image that was taken successfully
        :return:
        """
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_for_images = self.rgb_image_resized
        folder_path_for_orig_images = self.rgb_image_orig
        img_name = 'num_dt.jpeg'
        img_name = img_name.replace("num", str(g_param.image_number))
        img_name = img_name.replace("dt", str(get_local_time_3()))
        image_path_resized = os.path.join(folder_path_for_images, img_name)
        image_path_original = os.path.join(folder_path_for_orig_images, img_name)
        plt.imsave(image_path_original, original_frame)
        plt.imsave(image_path_resized, frame)  # saves frame in image_path
        plt.clf()  # clears figure
        plt.close()  # closes figure

    def write_platform_step(self, step_size):
        """
        write current step size in meters to csv
        :param step_size: current step size in meter
        """
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        """
        :param pos: current position of the robot
        :return:
        """
        data = [step_size]
        # opening the csv file in 'w+' mode
        current_time = get_local_time_4()
        folder_path_for_platform = self.platform_path
        platform_data = 'num_dt.csv'
        platform_data = platform_data.replace("num", str(g_param.image_number))
        platform_data = platform_data.replace("dt", str(current_time))
        platform_path = os.path.join(folder_path_for_platform, platform_data)

        file = open(platform_path, 'w+')
        # writing the data into the file
        with file:
            out = csv.writer(file)
            out.writerows(map(lambda x: [x], data))
            file.close()

    def write_transformations_to_csv(self):
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        """
        :param pos: current position of the robot
        :return:
        """
        current_time = get_local_time_4()
        transformation_data = 'num_dt.csv'
        transformation_data = transformation_data.replace("num", str(g_param.image_number))
        transformation_data = transformation_data.replace("dt", str(current_time))

        data_t_cam2base = g_param.trans.t_cam2base.tolist()
        data_t_cam2world = g_param.trans.t_cam2world.tolist()
        data_ang_vec_tcp = g_param.trans.ang_vec_tcp.tolist()
        data_t_tcp2base = g_param.trans.t_tcp2base.tolist()

        folder_path_for_t_cam2base = self.transformations_path_t_cam2base
        folder_path_for_t_cam2world = self.transformations_path_t_cam2world
        folder_path_for_ang_vec_tcp = self.transformations_path_ang_vec_tcp
        folder_path_for_data_t_tcp2base = self.transformations_path_t_tcp2base

        t_cam2base_path = os.path.join(folder_path_for_t_cam2base, transformation_data)
        t_cam2world_path = os.path.join(folder_path_for_t_cam2world, transformation_data)
        ang_vec_tcp_path = os.path.join(folder_path_for_ang_vec_tcp, transformation_data)
        t_tcp2base_path = os.path.join(folder_path_for_data_t_tcp2base, transformation_data)

        np.savetxt(t_cam2base_path, data_t_cam2base, delimiter=",")
        np.savetxt(t_cam2world_path, data_t_cam2world, delimiter=",")
        np.savetxt(ang_vec_tcp_path, data_ang_vec_tcp, delimiter=",")
        np.savetxt(t_tcp2base_path, data_t_tcp2base, delimiter=",")

    def write_location_to_csv(self, pos):
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        """
        :param pos: current position of the robot
        :return:
        """
        # print(type(pos), pos)
        data = pos.tolist()
        # print(type(data), data)
        # opening the csv file in 'w+' mode
        current_time = get_local_time_4()
        folder_path_for_location = self.location_path
        location_data = 'num_dt.csv'
        location_data = location_data.replace("num", str(g_param.image_number))
        location_data = location_data.replace("dt", str(current_time))
        location_path = os.path.join(folder_path_for_location, location_data)
        write_to_csv(location_path, data)

    def write_sonar_class_to_csv(self, record, mask_id):
        """
        :param record: recording of the sonar
        :param mask_id: id of the mask that the recording refers to.
        """
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_for_sonar_class = self.class_sonar_path
        sonar_class_data = generate_str_num_mask_dt(mask_id)
        sonar_class_data = os.path.join(folder_path_for_sonar_class, sonar_class_data)
        write_to_csv(sonar_class_data, record)

    def write_sonar_dist_s_to_csv(self, s, mask_id):
        """
        :param s: s recording of the sonar
        :param mask_id: id of the mask that the recording refers to.
        """
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_for_sonar_dist_s = self.s_dist_sonar_path
        sonar_dist_s_data = generate_str_num_mask_dt(mask_id)
        sonar_dist_s_data = os.path.join(folder_path_for_sonar_dist_s, sonar_dist_s_data)
        write_to_csv(sonar_dist_s_data, s)

    def write_sonar_dist_t_to_csv(self, t, mask_id):
        """
        :param t: t recording of the sonar
        :param mask_id: id of the mask that the recording refers to.
        """
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_for_sonar_class = self.t_dist_sonar_path
        sonar_dist_t_data = generate_str_num_mask_dt(mask_id)
        sonar_dist_t_data = os.path.join(folder_path_for_sonar_class, sonar_dist_t_data)
        write_to_csv(sonar_dist_t_data, t)

    def write_sonar_distances(self, mask_id, distances):
        """
        write the measured distance and the real distance.
        :param mask_id: id of the grape
        :param distances: measured, real distance from sonar to grape
        """
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_for_sonar_dist_s = self.distances
        distances_path = generate_str_num_mask_dt(mask_id)
        distances_path = os.path.join(folder_path_for_sonar_dist_s, distances_path)
        write_to_csv(distances_path, distances)

    def write_sonar_classes(self, mask_id, classes):
        """
        write the measured distance and the real distance.
        :param mask_id: id of the grape
        :param classes: measured, real distance from sonar to grape
        """
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_for_sonar_classes = self.classes
        distances_path = generate_str_num_mask_dt(mask_id)
        distances_path = os.path.join(folder_path_for_sonar_classes, distances_path)
        write_to_csv(distances_path, classes)

    def write_tb(self):
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_TB = self.TB_path
        tb_path = os.path.join(folder_path_TB, 'TB.csv')
        for i in range(0, len(g_param.TB)):
            g_param.TB[i].mask = None  # TODO - make that it will receive the mask path as it was saved to exp_data
        file = open(tb_path, 'w+')
        with file:
            out = csv.writer(file)
            out.writerows(map(lambda x: [x], g_param.TB))
            file.close()

    # ------------------------------------------------------------------
    # ---------------------------- read --------------------------------
    # ------------------------------------------------------------------

    def load_image_path(self):
        """
        :return: path for image to be loaded
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'rgb_images')
        locations_list = os.listdir(path)
        res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
        path = os.path.join(path, res[0])
        return path

    def read_location_from_csv(self):
        """
        read location from csv.
        it takes the location according to the current image number
        :return:
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'locations')
        locations_list = os.listdir(path)
        res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
        path = os.path.join(path, res[0])
        pos = np.genfromtxt(path, delimiter=',')
        return pos

    def read_transformations_from_csv(self):
        """
        read location from csv.
        it takes the location according to the current image number
        :return:
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'transformations')
        locations_list = os.listdir(path)
        res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
        path = os.path.join(path, res[0])
        pos = np.genfromtxt(path, delimiter=",")
        return pos

    def read_platform_step_size_from_csv(self):
        """
        read platform step size from csv.
        it takes the platform according to the current image number
        :return:
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'platform')
        locations_list = os.listdir(path)
        res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
        path = os.path.join(path, res[0])
        step_size = np.genfromtxt(path, delimiter=',')
        return step_size

    def read_sonar_class_from_csv(self, mask_id):
        """
        reads sonar recording by image+mask_id
        :param mask_id: id of the mask
        :return: recording of class
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'sonar')
        path = os.path.join(path, 'class_sonar')
        records_list = os.listdir(path)
        res = [i for i in records_list if i.startswith(str(image_number) + "_" + str(mask_id))]
        path = os.path.join(path, res[0])
        rec = np.genfromtxt(path, delimiter=",")
        return rec

    def read_sonar_dist_t_to_csv(self, mask_id):
        """

        :param mask_id: id of the mask
        :return:
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'sonar')
        path = os.path.join(path, 'dist_sonar')
        path = os.path.join(path, 't_dist_sonar')
        records_list = os.listdir(path)
        res = [i for i in records_list if i.startswith(str(image_number) + "_" + str(mask_id))]
        path = os.path.join(path, res[0])
        t = np.genfromtxt(path, delimiter=",")
        return t

    def read_sonar_dist_s_to_csv(self, mask_id):
        """
        :param mask_id: id of the mask
        :return: the s of the dist activation
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'sonar')
        path = os.path.join(path, 'dist_sonar')
        path = os.path.join(path, 's_dist_sonar')
        records_list = os.listdir(path)
        res = [i for i in records_list if i.startswith(str(image_number) + "_" + str(mask_id))]
        path = os.path.join(path, res[0])
        s = read_3d_text(path)
        return s

    def read_sonar_distances(self, mask_id, real_dist_calc):
        """
        :param real_dist_calc: calculated/ measured distance
        :param mask_id: id of the mask
        :return: measured distance by sonar, real distance
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'sonar')
        path = os.path.join(path, 'distances') #FIXME
        records_list = os.listdir(path)
        if len(records_list) == 0:  # if dir is empty, take calculated/ measured distance
            return real_dist_calc, real_dist_calc
        res = [i for i in records_list if i.startswith(str(image_number) + "_" + str(mask_id))]
        path = os.path.join(path, res[0])
        distances = np.genfromtxt(path, delimiter=",")
        return distances[0], distances[1]

    def read_sonar_classes(self, mask_id, sonar_class):
        """
        :param sonar_class: class calculated by sonar CNN
        :param mask_id: id of the mask
        :return: Real class of the object
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'sonar')
        path = os.path.join(path, 'classes')
        records_list = os.listdir(path)
        if len(records_list) == 0:  # if dir is empty, take calculated/ measured distance
            return sonar_class, sonar_class
        res = [i for i in records_list if i.startswith(str(image_number) + "_" + str(mask_id))]
        path = os.path.join(path, res[0])
        classes = np.genfromtxt(path, delimiter=",")
        return classes[0], classes[1]

    def load_mask(self, mask_id):
        """
        :param mask_id: mask id to load
        :return: mask
        """
        image_number = get_image_num_sim(g_param.image_number)
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'masks')
        records_list = os.listdir(path)
        res = [i for i in records_list if i.startswith(str(image_number) + "_" + str(mask_id))]
        path = os.path.join(path, res[0])
        mask = np.load(path)
        mask = mask.f.arr_0
        return mask

