import time
import csv
import os
import g_param
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def get_local_time_date(): # TODO- make it work
    """
    :return: hours_min_sec_millisec
    """
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")


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
        self.exp_date_time = 'exp_data_12_32'

    def create_directories(self):  # TODO: add Date to the name of the directory
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        """
        Creates all directories and sub directories
        """
        # Directory
        directory = "exp_data_dataTime"
        directory = directory.replace("dataTime", get_local_time_2())
        parent_dir = r'D:\Users\NanoProject'
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        directory_1, directory_2, directory_3 = "locations", "rgb_images", "masks"
        directory_4, directory_5 = "sonar", "transformations"
        path_1 = os.path.join(path, directory_1)
        path_2 = os.path.join(path, directory_2)
        path_3 = os.path.join(path, directory_3)
        path_4 = os.path.join(path, directory_4)
        path_5 = os.path.join(path, directory_5)
        os.mkdir(path_1)
        os.mkdir(path_2)
        os.mkdir(path_3)
        os.mkdir(path_4)
        os.mkdir(path_5)
        self.location_path = path_1
        self.rgb_images_path = path_2
        self.masks_path = path_3
        self.sonar_path = path_4
        self.transformations_path = path_5

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

    def write_transformations_to_csv(self):
        if g_param.process_type == "work" or g_param.process_type == "load":
            return
        """
        :param pos: current position of the robot
        :return:
        """
        print(type(g_param.trans.t_cam2base))
        data = g_param.trans.t_cam2base.tolist()
        # opening the csv file in 'w+' mode
        current_time = get_local_time_4()
        folder_path_for_transformation = self.transformations_path
        transformation_data = 'num_dt.csv'
        transformation_data = transformation_data.replace("num", str(g_param.image_number))
        transformation_data = transformation_data.replace("dt", str(current_time))
        transformation_path = os.path.join(folder_path_for_transformation, transformation_data)

        np.savetxt(transformation_path, data, delimiter=",")
        # file = open(transformation_path, 'w+')
        # # writing the data into the file
        # with file:
        #     # write = csv.writer(file)
        #     # write.writerows(data)
        #     out = csv.writer(file)
        #     out.writerows(map(lambda x: [x], data))
        #     file.close()

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

########################## read ################################

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
        path = os.path.join(path, 'transformations')
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

