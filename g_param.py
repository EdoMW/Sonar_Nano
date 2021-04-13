TB = []
masks_image = None
half_width_meter = 0.34
half_height_meter = 0.2
show_images = None
spray_sonar = True
trans = None
avg_dist = 0.71
platform_step_size = 0
sum_platform_steps = 0  # sum of all platform steps
last_grape_dist = 0.71
step_size = 0.25
height_step_size = 0.6  # parameter to_tune
# avg_dist = (avg_dist * 10 + average(TB.distance) * len(TB)) / (10 + len(TB)) TODO: ask Sigal
image_number = 0
plat_position_step_number = 0
read_write_object = None
direction = None
safety_dist = 0.20  # distance of spraying (in lab!! needs to be changed)
time_to_move_platform = False
image_cnn_path = r'C:\Drive\Mask_RCNN-master\logs_to_import\exp_7\mask_rcnn_grape_0080.h5'
cnn_config = None


"""
steps_gap: determines how many horizontal steps should be done.
for example, if step size is 0.1m and we want to keep it that wat (default) than steps_gap = 1.
if an experiment wants to test step size of 0.2m, than steps_gap should be equal to 2.
"""
steps_gap = 1

"""
work_place: lab/field
process_type: work/record/load
work- don't save any data except the Final TB 
record- save all relevant data in folders (as CSV/JPG..)
load- load all the date that was recorded.
change parameters if necessary.
"""

process_type = "work"  # TODO-add save of the TB before ending the program. also descriptive statistic
work_place = "lab"  # lab. to know which function of image processing to use.


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
    cnn_config = {"BACKBONE": train_config_obj.BACKBONE,
                  "DETECTION_MIN_CONFIDENCE": train_config_obj.DETECTION_MIN_CONFIDENCE,
                  "DETECTION_NMS_THRESHOLD": train_config_obj.DETECTION_NMS_THRESHOLD,
                  "GPU_COUNT": train_config_obj.GPU_COUNT,
                  "IMAGES_PER_GPU": train_config_obj.IMAGES_PER_GPU,
                  "LEARNING_MOMENTUM": train_config_obj.LEARNING_MOMENTUM,
                  "LEARNING_RATE": train_config_obj.LEARNING_RATE,
                  "STEPS_PER_EPOCH": train_config_obj.STEPS_PER_EPOCH,
                  "WEIGHT_DECAY": train_config_obj.WEIGHT_DECAY
                  }
    return cnn_config


def init():
    """
    initializes all global variables.
    """
    global TB, masks_image, show_images, trans, avg_dist, time_to_move_platform, plat_position_step_number, \
        image_number, safety_dist, half_width_meter, half_height_meter, sum_platform_steps, work_place, step_size,\
        read_write_object, process_type, last_grape_dist, height_step_size, direction, platform_step_size, \
        image_cnn_path, cnn_config, steps_gap
    half_width_meter = calc_image_width()
    half_height_meter = calc_image_height()
