import csv
import os
import pandas as pd
from pathlib import *

import g_param

"""
This module includes all functions for evaluating the results of a single simulation.
"""


class Result:
    def __init__(self, sim_time_path):
        self.sim_time_path = sim_time_path
        self.sim_time = sim_time_path.parts[-1]
        self.num_grapes_sprayed = self.read_tb_summary()
        self.gt_df = self.read_track_gt_df()
        self.grapes_count_gt = len(self.gt_df['general_id'].unique()) # 15- the real amount of grapes.
        self.pred_df = self.read_track_pred_df()
        self.pred_df_fil = self.read_track_pred_fil_df()
        self.hit = self.pred_df_fil['global_id'].nunique()  # k1
        self.miss = self.grapes_count_gt - self.hit  # K0

        self.s = self.num_grapes_sprayed
        self.k = self.grapes_count_gt
        self.s_f = self.num_grapes_sprayed - self.hit
        self.s_t = self.hit
        self.k_0 = self.miss
        self.k_2 = check_for_duplicates(self.read_track_pred_df()[['global_id', 'global_id_TB']])
        self.k_1 = self.hit - self.k_2
        self.recall = self.k_1 / self.grapes_count_gt
        self.precision = self.calc_precision()
        self.f1 = self.calc_f1()

    def __repr__(self):
        result_obj = f'simulation time: {self.sim_time}'
        return result_obj

    def calc_precision(self):
        try:
            return self.k_1 / self.num_grapes_sprayed
        except ValueError:
            return 0

    def calc_f1(self):
        if (self.precision + self.recall) > 0:
            return (2 * self.precision * self.recall) / (self.precision + self.recall)
        return 0

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
        if g_param.process_type == 'load':
            path = self.sim_time_path.joinpath('grape_count.txt')
            f = open(path, "r")
            return int(f.read())
        else:
            return len(g_param.TB)

    def write_txt(self):
        """
        create a txt file with the name: 'Working in lab- no masks'
        steps gap- horizontal gap (normal, x2,..)
        :param sim_directory_path: path to save the txt file
        """
        path = self.sim_time_path.joinpath('results.csv')
        hit = self.hit
        miss = self.miss
        recall = self.recall
        precision = self.precision
        f1 = self.f1
        num_grapes_sprayed = self.s
        s_f = self.s_f
        s_t = self.s_t
        k_0 = self.k_0
        k_1 = self.k_1
        k_2 = self.k_2
        param_list = [num_grapes_sprayed, hit, miss, recall, precision, f1, s_f, s_t, k_0, k_1, k_2]
        param_list_name = ["num_grapes_sprayed", "hit", "miss", "recall", "precision", "f1",
                           's_f', 's_t', 'k_0', 'k_1', 'k_2']
        with open(path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(zip(param_list_name, param_list))
            csv_file.close()
        # text = open(path, "r")
        # text = ''.join([i for i in text]).replace(", ", ": ")
        # text = ''.join([i for i in text]).replace(",", ": ")
        # x = open(path, "w")
        # x.writelines(text)
        # x.close()

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
        return Path(r'D:\Users\NanoProject\simulations\exp_data_12_13')


def check_for_duplicates(df):
    df2 = df.groupby(['global_id', 'global_id_TB']).size().reset_index()
    k_n = len(df2.global_id.value_counts().reset_index(name="count").query("count > 1")["index"])
    return k_n


def get_results():
    """
    calculates the results for the last exp (when set to ture)
    or any desired exp- the specific exp should be specified inside get_simulation_results.
    Write the results into 'result' file.
    """
    if g_param.eval_mode:
        result = Result(get_simulation_results(True))
        result.write_txt()
        """
        code for re running all the results calculations.
        Not needed, unless a change in all the results files is required.
        :return: 
        """
        # sim_path = r'D:\Users\NanoProject\simulations'
        # dirs = os.listdir(sim_path)
        # for dir_name in dirs:
        #     dir_path = Path(os.path.join(sim_path, dir_name))
        #     result = Result(dir_path)
        #     result.write_txt()


if __name__ == '__main__':
    get_results()
