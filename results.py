import os
import pandas as pd
from pathlib import *

grapes_count_gt = 15
"""
This module includes all functions for evaluating the results of a single simulation.
"""


class Result:
    def __init__(self, sim_time_path):
        self.sim_time_path = sim_time_path
        self.sim_time = sim_time_path.parts[-1]
        self.num_grapes_sprayed = self.read_tb_summary()
        self.gt_df = self.read_track_gt_df()
        self.pred_df = self.read_track_pred_df()
        self.pred_df_fil = self.read_track_pred_fil_df()
        self.hit = self.pred_df_fil['global_id'].nunique()  # k1
        self.miss = grapes_count_gt - self.hit  # K0
        self.recall = self.hit / grapes_count_gt
        self.precision = self.hit / self.num_grapes_sprayed
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
        self.s = self.num_grapes_sprayed
        self.k = grapes_count_gt
        self.s_f = self.num_grapes_sprayed - self.hit
        self.s_t = self.hit
        self.k_0 = self.miss
        self.k_1 = self.hit
        self.k_2 = 0

    def __repr__(self):
        result_obj = f'simulation time: {self.sim_time}'
        return result_obj

    def read_track_gt_df(self):
        gt_path = self.sim_time_path.joinpath(r'tracking\gt\tracking_gt.csv')
        gt_track = pd.read_csv(gt_path)
        return gt_track

    def read_track_pred_df(self):
        gt_path = self.sim_time_path.joinpath(r'tracking\pred\tracking_pred.csv')
        pred_track = pd.read_csv(gt_path)
        pred_track = pred_track.astype({'frame': int, 'frame_id_gt': int,
                                        'frame_id_pred': int, 'global_id': int})
        return pred_track

    def read_track_pred_fil_df(self):
        gt_path = self.sim_time_path.joinpath(r'tracking\pred\tracking_pred_filtered.csv')
        pred_track_fil = pd.read_csv(gt_path)
        pred_track_fil = pred_track_fil.astype({'frame': int, 'frame_id_gt': int,
                                                'frame_id_pred': int, 'global_id': int})
        return pred_track_fil

    def read_tb_summary(self):
        path = self.sim_time_path.joinpath('grape_count.txt')
        f = open(path, "r")
        return int(f.read())

#
#
# def print_simulations_list():
#     """
#     print first 10 simulations on the list
#     """
#     sims = get_simulations_list()
#     print(sims[:10])


def get_simulation_results(take_last_exp):
    """
    :return: list of all simulations
    """
    directory = Path(r'D:\Users\NanoProject\simulations')
    if take_last_exp and len(os.listdir(directory)) > 0:
        return max([directory.joinpath(d) for d in directory.iterdir()], key=os.path.getmtime)
    else:
        return []


# -------- Efficiency ---------


if __name__ == '__main__':
    result = Result(get_simulation_results(True))
    df = result.read_track_pred_df()[['global_id', 'global_id_TB']]
    for i in range(0,15):
        temp_df = df[df['global_id'] == i]
        vals_par_id = temp_df['global_id'].unique()
        if len(vals_par_id) > 0:
            print(f'index {i}: matching TB indexes {vals_par_id}')
    # print(df)
