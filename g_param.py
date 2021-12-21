import os

manual_work = False
TB = []
masks_image = None
half_width_meter = 0.34
half_height_meter = 0.2
auto_time_display = 100  # time to display image automatically
show_images = True
spray_sonar = True
trans = None

avg_dist = 0.63  # 75
platform_step_size = 0.5  # TODO (change to 0.5 for exp movement default)
sum_platform_steps = 0  # sum of all platform steps
last_grape_dist = 0.63
step_size = 0.1
height_step_size = 1.5  # parameter to_tune
# avg_dist = (avg_dist * 10 + average(TB.distance) * len(TB)) / (10 + len(TB)) TODO (after exp)
image_number = 0
plat_position_step_number = 0
read_write_object = None
direction = None
# safety_dist = 0.20  # distance of spraying (in lab!! needs to be changed)
safety_dist = 0
time_to_move_platform = False
image_cnn_path = r'weights\2021_weights.h5'
cnn_config = None
min_spray_dist = 0.10
max_spray_dist = 0.25
min_vel = 0.2
max_vel = 0.5
const_vel = max_vel
x_lim = (-0.5, 1)
y_lim = (-0.5, 1)
z_lim = (-0.5, 1)
# x_lim = (0, 0.1)
# y_lim = (0, 0.1)
# z_lim = (0, 0.1)
# UR5 limitation:
max_euclid_dist = 0.97
z_max = 0.85
z_min = -0.35
y_max = 0.6
sonar_x_length = 0.075
sprayer_x_length = 0.095
base_rotation_ang = 225  # 180 for lab 225 for volcani # TODO- don't forget to change value!
images_in_run = 1  # amount of images in the current run
table_of_matches = None
table_of_stats = None

"""
steps_gap: determines how many horizontal steps should be done.
for example, if step size is 0.1m and we want to keep it that (default) than steps_gap = 1.
if an experiment wants to test step size of 0.2m, than steps_gap should be equal to 2.
"""
steps_gap = 1
"""
work_place: lab/field/lab_grapes:

lab) white shapes on blue background.
field) real grapes in field
lab_grapes) plastic grapes on white background
process_type: work/record/load
work- don't save any data except the Final TB 
record- save all relevant data in folders (as CSV/JPG..)
load- load all the date that was recorded.
change parameters if necessary.
"""
process_type = 'load'
work_place = 'field'  # lab. to know which function of image processing to use.
eval_mode = None


def calc_image_width():
    """
     calculates the width in meters of half image (center ot edge).
    :return: half image width
    """
    half_image_width = avg_dist * 0.5 * (7.11 / 8)
    half_image_width = round(half_image_width, 2)
    print("half image width : ", half_image_width)
    return half_image_width


def calc_image_height():
    """
    calculates the height in meters of half image (center ot edge).
    :return: half image height
    """
    half_image_height = avg_dist * 0.595 * 0.5
    half_image_height = round(half_image_height, 2)
    print("half image height : ", half_image_height)
    return half_image_height


def get_cnn_config(train_config_obj):
    """
    :param train_config_obj:
    :return: mask rcnn config as a dictionary for writing it down
    """
    cnn_config_temp = {"BACKBONE": train_config_obj.BACKBONE,
                       "DETECTION_MIN_CONFIDENCE": train_config_obj.DETECTION_MIN_CONFIDENCE,
                       "DETECTION_NMS_THRESHOLD": train_config_obj.DETECTION_NMS_THRESHOLD,
                       "GPU_COUNT": train_config_obj.GPU_COUNT,
                       "IMAGES_PER_GPU": train_config_obj.IMAGES_PER_GPU,
                       "LEARNING_MOMENTUM": train_config_obj.LEARNING_MOMENTUM,
                       "LEARNING_RATE": train_config_obj.LEARNING_RATE,
                       "STEPS_PER_EPOCH": train_config_obj.STEPS_PER_EPOCH,
                       "WEIGHT_DECAY": train_config_obj.WEIGHT_DECAY
                       }
    return cnn_config_temp


def get_index(index):
    """
    :param index: index of current image
    :return: low image idx, high image idx
    """
    if index % 2 == 0:
        lpi_temp = index * 2
        hpi_temp = lpi_temp + 1
    else:
        lpi_temp = index * 2 + 1
        hpi_temp = lpi_temp - 1
    return lpi_temp, hpi_temp


def build_array(step_size_sim):
    """
    builds array to take image from
    range is 0-50 because max size of the array is 41.
    :param step_size_sim:
    :return:
    """
    move_direction = 0  # even = up, odd = down
    b = []
    for i in range(0, 50, step_size_sim):
        lpi, hpi = get_index(i)
        if move_direction % 2 == 0:
            b.append(lpi)
            b.append(hpi)
        else:
            b.append(hpi)
            b.append(lpi)
        move_direction += 1
    return b


def get_image_num_sim(image_num):
    global steps_gap
    b = build_array(steps_gap)
    print(b[image_num])
    return b[image_num]


def empty_npz_dir():
    """
    Delete all mask npz files from previous run
    """
    path = r'D:\Users\NanoProject\Sonar_Nano\npzs'
    file_list = [f for f in os.listdir(path)]
    for f in file_list:
        os.remove(os.path.join(path, f))


def init():
    """
    initializes all global variables.
    """
    global TB, masks_image, show_images, trans, avg_dist, time_to_move_platform, plat_position_step_number, \
        image_number, safety_dist, half_width_meter, half_height_meter, sum_platform_steps, work_place, step_size, \
        read_write_object, process_type, last_grape_dist, height_step_size, direction, platform_step_size, \
        image_cnn_path, cnn_config, steps_gap, min_spray_dist, max_spray_dist, max_euclid_dist, z_max, z_min, y_max,\
        manual_work, base_rotation_ang, eval_mode, auto_time_display, x_lim, y_lim, z_lim, table_of_matches,\
        table_of_stats
    half_width_meter = calc_image_width()
    half_height_meter = calc_image_height()
    empty_npz_dir()

