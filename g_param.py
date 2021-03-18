
TB = []
masks_image = None
half_width_meter = 0.34
half_height_meter = 0.2
show_images = None
spray_sonar = True
trans = None
const_dist = 0.71
image_number = 0
# const_dist = 0.51 # it was 1cm bigger than measured

"""
process_type: work/record/load
work- don't save any data except the Final TB 
record- save all relevant data in folders (as CSV/JPG..)
load- load all the date that was recorded.
change parameters if necessary.
"""
process_type = "work"  # TODO-add save of the TB before ending the program

read_write_object = None
safety_dist = 0.26
time_to_move_platform = False


def calc_image_width():
    """
     calculates the width in meters of half image (center ot edge).
    :return: half image width
    """
    half_image_width = const_dist * 0.5 * (7.11 / 8)
    half_image_width = round(half_image_width, 2)
    print("half image size : ", half_image_width)
    return half_image_width


def calc_image_height():
    """
    calculates the height in meters of half image (center ot edge).
    :return: half image height
    """
    half_image_height = const_dist * 0.595 * 0.5
    half_image_height = round(half_image_height, 2)
    print("image height : ", half_image_height)
    return half_image_height


def init():
    """
    initializes all global variables.
    """
    global TB, masks_image, show_images, trans, const_dist, time_to_move_platform,\
        image_number, safety_dist, half_width_meter, half_height_meter, read_write_object, process_type
    half_width_meter = calc_image_width()
    half_height_meter = calc_image_height()
