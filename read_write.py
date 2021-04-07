import time
import csv
import os
import g_param
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

take_last_exp = True  # take the exp that was conducted the latest


def get_local_time_date(): # TODO- make it work
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
    """
    if take_last_exp = True (default) it will return the last exp dir.
    :return: dir path of the exp to be analysed
    """
    if take_last_exp:
        directory = r'D:\Users\NanoProject\experiments'
        return max([os.path.join(directory, d) for d in os.listdir(directory)], key=os.path.getmtime)
    else:
        return 'exp_data_10_53'


class ReadWrite:
    """
    For read and write functions. save the exp data in real time to be redone later.
    """
    def __init__(self):
        self.location_path = None
        self.rgb_images_path = None
        self.masks_path = None
        self.sonar_path = None
        self.transformations_path = None
        self.platform_path = None
        self.TB_path = None
        self.transformations_path_ang_vec_tcp = None
        self.transformations_path_t_tcp2base = None
        self.transformations_path_t_cam2base = None
        self.transformations_path_t_cam2world = None
        self.exp_date_time = get_latest_dir()

    def create_directories(self):  # TODO: add Date to the name of the directory
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        """
        Creates all directories and sub directories
        """
        # Directory
        directory = "exp_data_dataTime"
        directory = directory.replace("dataTime", get_local_time_2())
        parent_dir = r'D:\Users\NanoProject\experiments'
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        directory_1, directory_2, directory_3 = "locations", "rgb_images", "masks"
        directory_4, directory_5, directory_6, directory_7 = "sonar", "transformations", "platform", "TB"
        sub_dir_1, sub_dir_2, sub_dir_3, sub_dir_4 = "ang_vec_tcp", "t_tcp2base", "t_cam2base", "t_cam2world"

        path_1 = os.path.join(path, directory_1)
        path_2 = os.path.join(path, directory_2)
        path_3 = os.path.join(path, directory_3)
        path_4 = os.path.join(path, directory_4)
        path_5 = os.path.join(path, directory_5)
        path_6 = os.path.join(path, directory_6)
        path_7 = os.path.join(path, directory_7)

        path_5_1 = os.path.join(path_5, sub_dir_1)
        path_5_2 = os.path.join(path_5, sub_dir_2)
        path_5_3 = os.path.join(path_5, sub_dir_3)
        path_5_4 = os.path.join(path_5, sub_dir_4)

        folders = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_5_1, path_5_2, path_5_3, path_5_4]
        for folder in folders:
            os.mkdir(folder)
        self.location_path = path_1
        self.rgb_images_path = path_2
        self.masks_path = path_3
        self.sonar_path = path_4
        self.transformations_path = path_5
        self.platform_path = path_6
        self.TB_path = path_7
        self.transformations_path_ang_vec_tcp = path_5_1
        self.transformations_path_t_tcp2base = path_5_2
        self.transformations_path_t_cam2base = path_5_3
        self.transformations_path_t_cam2world = path_5_4

    def save_rgb_image(self, frame):
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_for_images = self.rgb_images_path
        img_name = 'num_dt.jpeg'
        img_name = img_name.replace("num", str(g_param.image_number))
        img_name = img_name.replace("dt", str(get_local_time_3()))
        image_path = os.path.join(folder_path_for_images, img_name)
        plt.imsave(image_path, frame)  # saves frame in image_path
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
        print(type(pos), pos)
        data = pos.tolist()
        print(type(data), data)
        # opening the csv file in 'w+' mode
        current_time = get_local_time_4()
        folder_path_for_location = self.location_path
        location_data = 'num_dt.csv'
        location_data = location_data.replace("num", str(g_param.image_number))
        location_data = location_data.replace("dt", str(current_time))
        location_path = os.path.join(folder_path_for_location, location_data)

        file = open(location_path, 'w+')
        # writing the data into the file
        with file:
            # write = csv.writer(file)
            # write.writerows(data)
            out = csv.writer(file)
            out.writerows(map(lambda x: [x], data))
            file.close()

    def write_tb(self):
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        folder_path_TB = self.TB_path
        tb_path = os.path.join(folder_path_TB, 'TB.csv')
        for i in range(0,len(g_param.TB)):
            g_param.TB[i].mask = None  # TODO - make that it will receive the mask path as it was saved to exp_data
        file = open(tb_path, 'w+')
        # writing the data into the file
        with file:
            # write = csv.writer(file)
            # write.writerows(data)
            out = csv.writer(file)
            out.writerows(map(lambda x: [x], g_param.TB))
            file.close()


########################## read ################################

    def load_image_path(self):
        image_number = g_param.image_number
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
        image_number = g_param.image_number
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
        image_number = g_param.image_number
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'transformations')
        locations_list = os.listdir(path)
        res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
        path = os.path.join(path, res[0])
        pos = my_data = np.genfromtxt(path, delimiter=",")
        return pos

    def read_platform_step_size_from_csv(self):
        """
        read platform step size from csv.
        it takes the platform according to the current image number
        :return:
        """
        image_number = g_param.image_number
        directory = self.exp_date_time
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        path = os.path.join(path, 'platform')
        locations_list = os.listdir(path)
        res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
        path = os.path.join(path, res[0])
        step_size = np.genfromtxt(path, delimiter=',')
        return step_size

