
TB = []
masks_image = None
half_width_meter = 0.34
half_height_meter = 0.2
show_images = None
spray_sonar = True
trans = None
avg_dist = 0.71
last_grape_dist = 0.71
# avg_dist = (avg_dist * 10 + average(TB.distance) * len(TB)) / (10 + len(TB)) TODO: this
image_number = 0
read_write_object = None
safety_dist = 0.40 # distance of spraying (in lab!! needs to be changed)
time_to_move_platform = False

"""
process_type: work/record/load
work- don't save any data except the Final TB 
record- save all relevant data in folders (as CSV/JPG..)
load- load all the date that was recorded.
change parameters if necessary.
"""
process_type = "work"  # TODO-add save of the TB before ending the program




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
    global TB, masks_image, show_images, trans, avg_dist, time_to_move_platform,\
        image_number, safety_dist, half_width_meter, half_height_meter,\
        read_write_object, process_type, last_grape_dist
    half_width_meter = calc_image_width()
    half_height_meter = calc_image_height()
