import time
import numpy as np
import g_param
from g_param import get_image_num_sim, build_array
import pandas as pd

#######################################################################
# ---------- tracking results analysis -------------


def create_track_gt_df():
    """
    read the csv file that describes the 2d tracking of the grape clusters.
    each column represent an image.
    each row represent a grape cluster
    the number (ranging 0 - 6) represnt the id of the grape in the frame (from left to right).

    This function converts it to a "2d" table, with the columns (left to right):
    frame  general_id  frame_id.
    sorted by frame (image ID), than by general_id and than by frame_id

    a similar function exits for converting the detections that had IoU > 0.5 into the same type of table.

    Later, a comparison should be made between these two tables.
    """
    gt_track = pd.read_csv(r'C:\Users\Administrator\Desktop\grapes\2d_track.csv',
                           header=None)
    rows_num = gt_track.shape[0]  # amount of total grape clusters in all GT.
    frames_num = gt_track.shape[1]  # 41 images
    table_3_l = []
    # print('Frame | ID in Frame | Cluster ID')
    for col in range(0, rows_num):
        for row in range(0, frames_num):
            if not pd.isna(gt_track[row][col]):
                if float(gt_track[row][col]) or gt_track[row][col] == 0:
                    # print(col, row, gt_track[row][col])
                    table_3_l.append([row, col,  int(gt_track[row][col])])
    table_3 = pd.DataFrame(table_3_l, columns=['frame', 'general_id', 'frame_id_gt'])
    table_3 = table_3.sort_values(["frame", "general_id"], ascending=(True, True))
    table = remove_unreachable_frames(table_3)
    return table


def remove_unreachable_frames(t):
    """
    Remove records where the frame won't be part of the frames in simulations (will be skipped).
    Note: if g_param.steps_gap == 1, no filter will be done.
    :param t: table to filter.
    :return: filterd table.
    """
    if g_param.steps_gap == 1:
        return t
    a = build_array(g_param.steps_gap)
    t = t[t['frame'].isin(a)]
    t = t.reset_index()
    return t


def create_track_pred_fillterd_df():
    """
    same as create_track_gt_df
    """
    pred_track = g_param.pred_gt_tracking
    pred_track = pred_track[pred_track['global_id'] > -1]
    return pred_track


def create_track_gt_from_pred_df():
    """
    same as create_track_pred_df but consitent with GT numnering of id_in_frame.
    """
    pred_track = g_param.table_of_matches_gt
    rows_num = pred_track.shape[0]  # amount of total grape clusters in all GT.
    frames_num = pred_track.shape[1]  # 41 images
    table_3_l = []
    for col in range(0, rows_num):
        for row in range(0, frames_num):
            if pred_track[row][col] > -1:
                table_3_l.append([row, col, pred_track[row][col]])
    table_3 = pd.DataFrame(table_3_l, columns=['frame', 'general_id', 'frame_id'])
    table_3 = table_3.sort_values(["frame", "general_id"], ascending=(True, True))
    g_param.pred_gt_df = table_3
    return table_3


def print_line_sep_time():
    """
    prints a line separation of ___ and the current time
    """
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print('-' * 40, current_time, '-' * 40, '\n')
    image_masks = []
    for ind in range(len(g_param.TB)):
        if g_param.TB[ind].last_updated == get_image_num_sim(g_param.image_number):  # == g_param.image_number:
            image_masks.append([ind, g_param.TB[ind].index])
    print(image_masks)


def log_statistics():
    """
    write down some descriptive statics of what just recorded (total performance of the system,
    not specifically the tracking system.
    """
    # take_manual_image()
    amount_of_grapes = len(g_param.TB)
    amount_of_fake = sum(g.fake_grape is True for g in g_param.TB)
    amount_sprayed = sum((g.sprayed is False and g.fake_grape is False) for g in g_param.TB)
    print('-' * 50, '\n', '-' * 50, '\n', '-' * 25 + ' summary ' + '-' * 25)
    print(f'Total grapes detected: {amount_of_grapes}')
    print(f'Total "false" grapes detected: {amount_of_fake}')
    print(f'Precision : {round(((amount_of_grapes - amount_of_fake) / amount_of_grapes + 0.0000001), 3)}')
    print(f'percentage of grapes that were not reachable: {round((amount_sprayed / amount_of_grapes + 0.0000001), 3)}'
          f' \n')
    print('-' * 50, '\n', '-' * 50)
    pass


def create_centers_df():
    """
    Updates g_param.centers_df, a minimized version of TB to help follow the TB (first few frames).
    """
    vals, g_w_list = [], []
    distances = np.zeros((len(g_param.TB),len(g_param.TB)))
    for i in range(len(g_param.TB)):
        vals.append([g_param.TB[i].index, g_param.TB[i].x_p, g_param.TB[i].y_p, g_param.TB[i].grape_world[0],
                     g_param.TB[i].grape_world[1], g_param.TB[i].grape_world[2]])
        g_w_list.append(g_param.TB[i].grape_world)
    g_param.centers_df = pd.DataFrame(vals, columns=['index', 'x_p', 'y_p','grape_w_X','grape_w_Y','grape_w_Z'])
    for i in range(len(g_w_list)):
        for j in range(i + 1, len(g_w_list)):
            distances[i][j] = np.linalg.norm(g_w_list[i]-g_w_list[j])
    g_param.distances_matrix_2d = distances
