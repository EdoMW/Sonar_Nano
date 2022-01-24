import numpy as np
import pandas as pd
import csv
from pathlib import *


def get_data(table_t, path_to_dir):
    for file in path_to_dir.iterdir():
        file_name = file.parts[-1]
        if file.is_file() and file_name.startswith('res'):
            f = pd.read_csv(file).T
            f.columns = f.iloc[0]
            f = f.drop(f.index[0])
            if table_t is None:
                table_t = f
            else:
                table_t = pd.concat([table_t, f], axis=0)
    return table_t


if __name__ == '__main__':
    path = Path(r'D:\Users\NanoProject\simulations')
    table = None
    for dir in path.iterdir():
        table = get_data(table, dir)
    print(table)
