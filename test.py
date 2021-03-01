# import packages and the ReadFromRobot class- do not change
import sys
import numpy as np
import math
import DAQ_BG
from test_one_record import test_spec
from preprocessing_and_adding import preprocess_one_record
from distance import correlation_dist
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
import tensorflow as tf

counter = 0  # intiate flag
record = DAQ_BG.rec()
counter, x_test = preprocess_one_record(record, counter,
                                        adding='zero_cols')  # if counter==1 means new acquisition # adding (way of preprocess) - gets 'noise' or 'zero_cols'