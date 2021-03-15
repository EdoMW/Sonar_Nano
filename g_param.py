
TB = []
masks_image = None
half_width_meter = 0.34
show_images = True
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
process_type = "load"  # TODO-add save of the TB before ending the program

read_write_object = None
safety_dist = 0.16
time_to_move_platform = False
def calc_image_width():
    """
    :return: calculates the width in meters of half image (center ot edge).
    """
    const_dist_temp = const_dist
    print("half image size : ", const_dist_temp * 0.5 * (7.11 / 8))
    return const_dist_temp * 0.5 * (7.11 / 8)


def init():
    """
    initializes all global variables.
    """
    global TB, masks_image, show_images, trans, const_dist, time_to_move_platform,\
        image_number, safety_dist, half_width_meter, read_write_object, process_type
    half_width_meter = calc_image_width()
