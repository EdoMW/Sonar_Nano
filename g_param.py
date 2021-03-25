
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
height_step_size = 0.6  # parameter to_tune
# avg_dist = (avg_dist * 10 + average(TB.distance) * len(TB)) / (10 + len(TB)) TODO: this
image_number = 0
plat_position_step_number = 0
read_write_object = None
direction = None
safety_dist = 0.20  # distance of spraying (in lab!! needs to be changed)
time_to_move_platform = False

"""
process_type: work/record/load
work- don't save any data except the Final TB 
record- save all relevant data in folders (as CSV/JPG..)
load- load all the date that was recorded.
change parameters if necessary.
"""
process_type = "work"  # TODO-add save of the TB before ending the program.




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


def init():
    """
    initializes all global variables.
    """
    global TB, masks_image, show_images, trans, avg_dist, time_to_move_platform, plat_position_step_number,\
        image_number, safety_dist, half_width_meter, half_height_meter, sum_platform_steps,\
        read_write_object, process_type, last_grape_dist, height_step_size, direction, platform_step_size
    half_width_meter = calc_image_width()
    half_height_meter = calc_image_height()
