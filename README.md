# Sonar_Nano

# grape = ID, mask, pixel center, world center, angle, pixel width
# , pixel length , sprayed, dist_to_center
#
# dist_to_center = euclidean distance between pixel_center to center of image.
# ID = unique ID per grape cluster.
# grapes = grapes detected in image
# target_bank = list of grapes, sorted first by sprayed/ not sprayed, second by a chosen parameter
# in our case for example always the grape that is in the opposite direction of advancement
# import packages and the ReadFromRobot class- do not change

# list of parameters to tune:
#     step_size = 0.45
#     sleep_time = 2.9
#     safety_dist = 0.30
#     distance = 680 # change
#     same_grape_distance_threshold = 9 cm (0.09m)
#     show_images = True (default) show images
#     Vertical/ horizontal step size
#     amount of horizontal steps before moving the platform
