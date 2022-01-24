import pandas as pd
from pathlib import *


def get_results(path_to_dir):
    """
    :param path_to_dir: windowsPath instance of path to dir
    :return: table of this simulation
    """
    for file in path_to_dir.iterdir():
        file_name = file.parts[-1]
        if file.is_file() and file_name.startswith('res'):
            f = pd.read_csv(file).T
            f.columns = f.iloc[0]
            f = f.drop(f.index[0])
            f['num_grapes_sprayed'] = f.index
            f = f.reset_index(drop=True)
            return f


def get_data(path_to_dir):
    for file in path_to_dir.iterdir():
        file_name = file.parts[-1]
        if file.is_file() and file_name.startswith('cs'):
            fs = file_name.split('_')
            fs[-1] = fs[-1][:4]
            if fs[-1][-1] == '.':
                fs[-1] = fs[-1][:-1]
            row = pd.DataFrame([float(fs[1]), float(fs[3]), float(fs[5]), float(fs[7])]).T
            return row


def get_results_values(path):
    table_t = None
    for dir_f in path.iterdir():
        temp_t = get_results(dir_f)
        table_t = pd.concat([temp_t, table_t], axis=0)
    table_t = table_t.rename({'index': 'total_detected'}, axis='index')
    table_t = table_t.reset_index(drop=True)
    return table_t


def get_options_tested(path):
    table_t = pd.DataFrame(columns=['', 'IoU', 'steps_gap', 'same_grape'])
    for dir_f in path.iterdir():
        temp_t = get_data(dir_f)
        table_t = pd.concat([temp_t, table_t], axis=0)
    table_t = table_t.rename({0: 'confidence_score', 1: 'IoU', 2: 'steps_gap', 3: 'same_grape'}, axis='columns')
    table_t = table_t.dropna(axis=1, how='all')
    table_t = table_t.reset_index(drop=True)
    return table_t


if __name__ == '__main__':
    path_sim = Path(r'D:\Users\NanoProject\simulations')
    table_0 = get_results_values(path_sim)
    table_1 = get_options_tested(path_sim)
    print(table_0)
    print(table_1)
