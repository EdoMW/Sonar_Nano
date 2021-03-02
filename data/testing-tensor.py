import tensorflow as tf

from mask_rcnn import take_picture_and_run as TPAR
cen_poi_x_0, cen_poi_y_0 = TPAR()
print("Center in cm: ", cen_poi_x_0, ",", cen_poi_y_0)