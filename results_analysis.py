import pandas as pd
from pathlib import *
import matplotlib.pyplot as plt


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
    # table_t = None
    table_t = pd.DataFrame(columns=['hit', 'miss', 'recall', 'precision', 'f1', 's_f', 's_t',
                                    'k_0', 'k_1', 'k_2', 'num_grapes_sprayed'])
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


def scatter_recall_precision(table_0, table_1):
    scatter = plt.scatter(x=table_0['recall'], y=table_0['precision'], c=table_1['steps_gap'])
    classes = ['1', '2']
    # values = table_1['steps_gap']
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.show()


def f1_chart(table_0, table_1, conf_level):
    x = table_1.apply(lambda row: [row['IoU'], row['steps_gap']], axis=1)
    y = [table_0['f1'][i] for i in table_1.index]
    plt.xticks(x.index, x)
    scatter = plt.scatter(x=x.index, y=y, c=table_1['steps_gap'])
    classes = ['1', '2']
    plt.xlabel('IoU, Steps')
    plt.ylabel('F1')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.show()


def stack_bar_confidence(table_0, table_1, conf_level):
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Hit, miss at confidence level of {conf_level}')
    ax.legend()
    x = table_1.apply(lambda row: [row['IoU'], row['steps_gap']], axis=1)
    plt.xticks(table_1.index, x)
    y1 = [table_0['hit'][i] for i in table_1.index]
    y2 = [table_0['miss'][i] for i in table_1.index]

    # plot bars in stack manner
    plt.bar(x.index, y1, color='g')
    plt.bar(x.index, y2, bottom=y1, color='r')
    plt.show()


if __name__ == '__main__':
    path_sim = Path(r'D:\Users\NanoProject\simulations')
    table_0 = get_results_values(path_sim)
    table_1 = get_options_tested(path_sim)
    table_2 = pd.concat([table_1, table_0], axis=1)
    table_2.to_csv(r'C:\Users\Administrator\Desktop\grapes\results.csv')

    print(table_0)
    print(table_1)
    print('total number is', len(table_0))
    scatter_recall_precision(table_0, table_1)

    for conf_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        f1_chart(table_0, table_1[table_1['confidence_score'] == conf_val], conf_val)
        stack_bar_confidence(table_0, table_1[table_1['confidence_score'] == conf_val], conf_val)
    # [xyUnique, ignore, ixs] = unique(xy,'rows')

