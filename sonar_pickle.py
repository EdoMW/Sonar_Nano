# import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import cv2
# import shutil
import io
# import pandas as pd
#
import pickle
import pandas as pd



path = r'C:\Users\Administrator\Desktop\Grapes Moran\2degreeproj\spec_grapes_500_noise_learn123_test4_nfft512.pkl'
file = open(path, 'rb')
object_file = pickle.load(file)
file.close()

data = pd.read_pickle(path)[4]
df = pd.DataFrame.from_dict(data).T
df.reset_index(level=1, inplace=True)

df.reset_index(level=0, inplace=True)
df = df.drop(columns=['index', 2, 3])
df = df.rename(columns={"level_1": "dataset", 0: "path", 1: "file_name"})
df_1 = df.loc[:, ['dataset', 'file_name']]
df_2 = df['path'].str.rsplit("/", expand=True)
df_2 = df_2.drop(columns=[0, 1, 2, 3, 4, 5, 6])
df_2 = df_2.rename(columns={7: "day", 8: "type", 9: "branch", 10: "file_num"})
df = pd.concat([df_1, df_2], axis=1)
# print(df.columns)
df_1 = df.loc[df['dataset'] != "train"]
path_to_test = []
for i in range(len(df_1)):
    rlist = '/'.join(df_1[i:i+1].values.tolist()[0][2:])
    rlist = rlist + '/' + df_1[i:i+1].values.tolist()[0][1]
    path_to_test.append(rlist)
path_to_test = ['/'.join(x.split('/')[:-1]) for x in path_to_test]
print(len(path_to_test))
print(path_to_test)

# npz = np.load(path)
# npy = npz.f.arr_0
# im = Image.fromarray(npy)
# plt.imshow(im)
# plt.show()
# print(npy)

# a = np.zeros(shape=(5,2))
# a = a.reshape((5,2,1))
# print(a.shape)



corners = np.array([[93, 207], [295, 550], [238, 185], [150, 572]])


def bbox_from_corners(corners):
    x1 = min(corners[:, 0])
    x2 = max(corners[:, 0])
    y1 = min(corners[:, 1])
    y2 = max(corners[:, 1])
    bbox = np.array([x1, y1, x2, y2])
    return bbox

# bbox = bbox_from_corners(corners).reshape(1, 4)
# print(bbox)
# print(x1, y1)
# print(x2, y2)


# # 7 (columns), 30 (rows), 128 (chanks)
#
# path = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\sonar\dist_sonar\s_dist_sonar\1_1_13_48_40.csv'
# b = pd.read_csv(path, header=None)
# first_item = True
# list_of_2d = []
# for k in range (len(b)):
#     chunk = b.iloc[k, 0]
#     chank_list = chunk.split('\n')
#     chank_list = [x.replace('[', '').replace(']', '') for x in chank_list]
#     for i in range(30):
#         chank_list[i] = chank_list[i].split(' ')
#         for j in range(0, 7):
#             if len(chank_list[i][j]) > 2:
#                 if chank_list[i][j].endswith('\r'):
#                     chank_list[i][j] = chank_list[i][j][:-2]
#             if i > 0 and first_item:
#                 chank_list[i] = chank_list[i][1:]
#                 first_item = False
#             chank_list[i][j] = float(chank_list[i][j])
#         chank_list[i] = np.array(chank_list[i])
#         first_item = True
#     list_of_2d.append(np.array(chank_list))
# s = np.array(list_of_2d)
#
