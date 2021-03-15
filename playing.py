import numpy as np
import itertools
import os, re
import csv


directory = 'exp_data_12_32'
parent_dir = r'D:\Users\NanoProject'
path = os.path.join(parent_dir, directory)
path = os.path.join(path, 'transformations')
print(path)
locations_list = os.listdir(path)
print(locations_list)
image_number = 0
res = [i for i in locations_list if i.startswith(str(image_number) + "_")]
print(res)
path = os.path.join(path, res[0])
print(path)

from numpy import genfromtxt

my_data = np.genfromtxt(path, delimiter=",")
print(type(my_data))
print(my_data)


#
# def check():
#     if True:
#         return
#     print(hi)
# check()
#
# a = np.array([[1, 2],[3,4]])
# print(a)
# a = np.insert(arr = a,
#               obj = 2,
#               values = 4,
#               axis = 1)
# print(a)
# #
# # b = [[1,2],[3,4],[5,6]]
# # c = [["a","b"],["v", 'c'],["e","f"]]
# # res = [list(itertools.chain(*i))
# #        for i in zip(b, c)]
# # print(res)
# #
# #
# print("np")
# for i in range(3):
#     print(a)
#     corner_list = a[i:i+1,:2]
#     corn = corner_list.tolist()
#     print(corn)
#     print(type(corn))