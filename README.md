# Sonar_Nano

grape = ID, mask, pixel center, world center, angle, pixel width
, pixel length , sprayed, dist_to_center

dist_to_center = euclidean distance between pixel_center to center of image.

ID = unique ID per grape cluster.

grapes = grapes detected in image

target_bank = list of grapes, sorted first by sprayed/ not sprayed, second by a chosen parameter

in our case for example always the grape that is in the opposite direction of advancement

import packages and the ReadFromRobot class- do not change

list of parameters to tune:

    step_size = 0.1
    
    sleep_time = 2.9
    
    safety_dist = 0.20
    
    distance = 0.75
    
    same_grape_distance_threshold =0.04m
    
    show_images = True (default) show images
    
    Vertical step size = step_size * 0.9
    
    amount of horizontal steps before moving the platform  = 1
    
    DETECTION_MIN_CONFIDENCE = 0.8

    DETECTION_NMS_THRESHOLD = 0.1
