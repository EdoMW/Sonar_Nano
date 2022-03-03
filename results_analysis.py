import pandas as pd
from pathlib import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
import numpy as np
from matplotlib import cm
import matplotlib as mpl

"""
This script was writen to summarize the results of all the experiments of the 3D map.
It can produce various types of plots.
"""


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
    """
    :param path_to_dir: path to dir of the exp parameters.
    :return: df of single row. contains the parameters of the experiment.
    """
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
    """
    Read all files to create one big data frame.
    :param path: path to experiments results dir.
    :return: df of all experiments results (metrics)
    """
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
    """

    :param path:
    :return: df of all experiments parameters
    """
    table_t = pd.DataFrame(columns=['', 'IoU', 'steps_gap', 'same_grape'])
    for dir_f in path.iterdir():
        temp_t = get_data(dir_f)
        table_t = pd.concat([temp_t, table_t], axis=0)
    table_t = table_t.rename({0: 'confidence_score', 1: 'IoU', 2: 'steps_gap', 3: 'same_grape'}, axis='columns')
    table_t = table_t.dropna(axis=1, how='all')
    table_t = table_t.reset_index(drop=True)
    return table_t


def scatter_recall_precision(table_0, table_1):
    """
    :param table_0:
    :param table_1:
    :return:
    """
    scatter = plt.scatter(x=table_0['recall'], y=table_0['precision'], c=table_1['steps_gap'])
    classes = ['1', '2']
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


def three_d(df, metric):
    df = df[df['steps_gap'] == 1]
    df = df.loc[:, df.columns.drop('steps_gap')]
    # thickness of the bars
    dx, dy = .05, .05
    # prepare 3d axes
    # ax, fig = plt.figure(figsize=(10, 6),subplot_kw={'projection': ccrs.PlateCarree()}))
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Standard')
    # set up positions for the bars
    xpos = np.linspace(0.3, 0.7, 5)
    ypos = np.linspace(0.5, 0.9, 5)

    # set the ticks in the middle of the bars
    ax.set_xticks(xpos + dx / 2)
    ax.set_yticks(ypos + dy / 2)
    ax.set_zlim(0, 1)
    # create meshgrid
    # print xpos before and after this block if not clear
    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    # the bars starts from 0 attitude
    zpos = np.zeros(df.shape[0]).flatten()
    # the bars' heights
    dz = df[metric].values.ravel()

    colors = cm.winter(dz)

    # plot
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
    # put the column / index labels
    ax.w_xaxis.set_ticklabels([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.w_yaxis.set_ticklabels(np.linspace(0.5, 0.9, 5))

    # name the axes
    ax.set_xlabel('IoU')
    ax.set_ylabel('Classification \n confidence score')
    ax.set_zlabel(f'{metric}')
    ax.set_title(f'{metric} vs (IoU, and Confidence score)', y=1.1)
    cmap = cm.winter
    N = 11
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.05, 0.1, 0.03, 0.8])
    plt.colorbar(sm, ticks=np.linspace(0, 1, N), cax=cbaxes,
                 boundaries=np.arange(0, 1.05, .1), shrink=0.9, pad=0.15)
    # ax.azim = -20
    # ax.elev = 15

    plt.show()


def three_d_two_colors(df, metric):
    dzs = []
    df = df[df['steps_gap'] == 4]
    df = df.loc[:, df.columns.drop('steps_gap')]
    z1 = df['k_1'].values
    z2 = df['k_2'].values
    dzs.append(z1)
    dzs.append(z2)
    # thickness of the bars
    dx, dy = .05, .05
    # prepare 3d axes
    # ax, fig = plt.figure(figsize=(10, 6),subplot_kw={'projection': ccrs.PlateCarree()}))
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Standard')
    # set up positions for the bars
    xpos = np.linspace(0.3, 0.7, 5)
    ypos = np.linspace(0.5, 0.9, 5)

    # set the ticks in the middle of the bars
    ax.set_xticks(xpos + dx / 2)
    ax.set_yticks(ypos + dy / 2)
    # ax.set_zlim(0, 1)
    # create meshgrid
    # print xpos before and after this block if not clear
    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    # the bars starts from 0 attitude
    zpos = np.zeros(df.shape[0]).flatten()
    # the bars' heights
    dz = df[metric].values.ravel()

    # colors = cm.winter(dz)
    colors = ['green', '#FFC04C']
    # plot
    for i in range(2):
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz=dzs[i], color=colors[i])
        zpos += dzs[i]
    # put the column / index labels
    ax.w_xaxis.set_ticklabels([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.w_yaxis.set_ticklabels(np.linspace(0.5, 0.9, 5))

    # name the axes
    ax.set_xlabel('IoU')
    ax.set_ylabel('Classification \n confidence score')
    ax.set_zlabel(f'{metric}')
    ax.set_title(f'{metric} vs (IoU, and Confidence score)', y=1.1)
    cmap = cm.winter
    N = 11
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.05, 0.1, 0.03, 0.8])
    plt.colorbar(sm, ticks=np.linspace(0, 1, N), cax=cbaxes,
                 boundaries=np.arange(0, 1.05, .1), shrink=0.9, pad=0.15)
    ax.azim = 10
    ax.elev = 30

    plt.show()


def three_d_two():

    columns = ['R','Users','A','B','C']

    df=pd.DataFrame({'R':[2,2,2,4,4,4,6,6,6,8,8],
                     'Users':[80,400,1000,80,400,1000,80,400,1000,80,400],
                     'A':[ 0.05381,0.071907,0.08767,0.04493,0.051825,0.05295,0.05285,0.0804,0.0967,0.09864,0.1097],
                     'B':[0.04287,0.83652,5.49683,0.02604,.045599,2.80836,0.02678,0.32621,1.41399,0.19025,0.2111],
                     'C':[0.02192,0.16217,0.71645, 0.25314,5.12239,38.92758,1.60807,262.4874,8493,11.6025,6288]},
                    columns=columns)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")

    df["A"] = np.log(df["A"]+1)
    df["B"] = np.log(df["B"]+1)
    df["C"] = np.log(df["C"]+1)

    colors = ['r', 'g', 'b']

    num_bars = 11
    x_pos = []
    y_pos = []
    x_size = np.ones(num_bars*3)/4
    y_size = np.ones(num_bars*3)*50
    c = ['A','B','C']
    z_pos = []
    z_size = []
    z_color = []
    for i,col in enumerate(c):
        x_pos.append(df["R"])
        y_pos.append(df["Users"]+i*50)
        z_pos.append([0] * num_bars)
        z_size.append(df[col])
        z_color.append([colors[i]] * num_bars)

    x_pos = np.reshape(x_pos,(33,))
    y_pos = np.reshape(y_pos,(33,))
    z_pos = np.reshape(z_pos,(33,))
    z_size = np.reshape(z_size,(33,))
    z_color = np.reshape(z_color,(33,))

    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color=z_color)

    plt.xlabel('R')
    plt.ylabel('Users')
    ax.set_zlabel('Time')

    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='A',markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='B',markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='C',markerfacecolor='b', markersize=10)
                       ]
    # Make legend
    ax.legend(handles=legend_elements, loc='best')
    # Set view
    ax.view_init(elev=35., azim=35)
    plt.show()


def plot_spray_of_targets_3(df, metric):
    """
    plot total sprays of targets vs iou, step size, ccs.
    """
    fig, axs = plt.subplots(1, 5, figsize=(18,4), sharex='all')
    fig.suptitle(f'total sprays of targets for different overlapping size, IoU and classification confidence score', fontweight='bold')
    labels = df['IoU'].unique()
    for i, con_sco in enumerate(np.linspace(0.5, 0.9, 5)):
        ax = axs[i]
        metric_df = df[df['confidence_score'] == con_sco][['k_1', 'k_2', 'steps_gap']]
        df_0 = metric_df[metric_df['steps_gap'] == 1]['k_1'] + metric_df[metric_df['steps_gap'] == 1]['k_2']
        df_1 = metric_df[metric_df['steps_gap'] == 2]['k_1'] + metric_df[metric_df['steps_gap'] == 2]['k_2']
        df_2 = metric_df[metric_df['steps_gap'] == 4]['k_1'] + metric_df[metric_df['steps_gap'] == 4]['k_2']

        ax.bar(labels, df_0, width=0.03,  label='1')
        # ax.bar(labels, df_0_1, width=0.03, bottom=df_0, label='1, k_2')
        ax.bar(labels + 0.03, df_1, width=0.03, label='2')
        # ax.bar(labels + 0.03, df_1_1, width=0.03, bottom=df_1,label='2, k_2')
        ax.bar(labels - 0.03, df_2, width=0.03, label='4')
        # ax.bar(labels - 0.03, df_2_1, width=0.03, bottom=df_2,label='4, k_2')
        ax.set_ylim(0, 15)
        ax.set_xlabel("IoU", fontweight='bold')
        if i == 0:
            ax.set_ylabel("Score", fontweight='bold', fontsize=25)
        plt.legend()
        ax.set_title(f'{con_sco}')
        plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    # plt.rcParams["figure.figsize"] = (25, 4)
    plt.show()


def plot_spray_of_targets_2(df, metric):
    """
    plot k_1 vs iou, step size, ccs.
    """
    fig, axs = plt.subplots(1, 5, figsize=(18,4), sharex='all')
    fig.suptitle(f'Hit (k_1) for different overlapping size, IoU and classification confidence score', fontweight='bold')
    labels = df['IoU'].unique()
    for i, con_sco in enumerate(np.linspace(0.5, 0.9, 5)):
        ax = axs[i]
        metric_df = df[df['confidence_score'] == con_sco][['k_1', 'k_2', 'steps_gap']]
        df_0 = metric_df[metric_df['steps_gap'] == 1]['k_1']
        df_1 = metric_df[metric_df['steps_gap'] == 2]['k_1']
        df_2 = metric_df[metric_df['steps_gap'] == 4]['k_1']
        df_0_1 = metric_df[metric_df['steps_gap'] == 1]['k_2']
        df_1_1 = metric_df[metric_df['steps_gap'] == 2]['k_2']
        df_2_1 = metric_df[metric_df['steps_gap'] == 4]['k_2']
        ax.bar(labels, df_0, width=0.03,  label='1')
        # ax.bar(labels, df_0_1, width=0.03, bottom=df_0, label='1, k_2')
        ax.bar(labels + 0.03, df_1, width=0.03, label='2')
        # ax.bar(labels + 0.03, df_1_1, width=0.03, bottom=df_1,label='2, k_2')
        ax.bar(labels - 0.03, df_2, width=0.03, label='4')
        # ax.bar(labels - 0.03, df_2_1, width=0.03, bottom=df_2,label='4, k_2')
        ax.set_ylim(0, 15)
        ax.set_xlabel("IoU", fontweight='bold')
        if i == 0:
            ax.set_ylabel("Score", fontweight='bold')
        plt.legend()
        ax.set_title(f'{con_sco}')
        plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    # plt.rcParams["figure.figsize"] = (25, 4)
    plt.show()


def plot_spray_of_targets(df, metric):
    """
    plot k1, k2 (on top of it) vs step size, iou, ccs
    """
    fig, axs = plt.subplots(1, 5, figsize=(18,4), sharex='all', constrained_layout = True)
    fig.suptitle(f'k_1, k_2 for different overlapping size, IoU and classification confidence score', fontweight='bold')
    labels = df['IoU'].unique()
    for i, con_sco in enumerate(np.linspace(0.5, 0.9, 5)):
        ax = axs[i]
        metric_df = df[df['confidence_score'] == con_sco][['k_1', 'k_2', 'steps_gap']]
        df_0 = metric_df[metric_df['steps_gap'] == 1]['k_1']
        df_1 = metric_df[metric_df['steps_gap'] == 2]['k_1']
        df_2 = metric_df[metric_df['steps_gap'] == 4]['k_1']
        df_0_1 = metric_df[metric_df['steps_gap'] == 1]['k_2']
        df_1_1 = metric_df[metric_df['steps_gap'] == 2]['k_2']
        df_2_1 = metric_df[metric_df['steps_gap'] == 4]['k_2']
        ax.bar(labels, df_0, width=0.03,  label='1, k_1')
        ax.bar(labels, df_0_1, width=0.03, bottom=df_0, label='1, k_2')
        ax.bar(labels + 0.03, df_1, width=0.03, label='2, k_1')
        ax.bar(labels + 0.03, df_1_1, width=0.03, bottom=df_1,label='2, k_2')
        ax.bar(labels - 0.03, df_2, width=0.03, label='4, k_1')
        ax.bar(labels - 0.03, df_2_1, width=0.03, bottom=df_2,label='4, k_2')
        ax.set_ylim(0, 15)
        if i == 3 or i == 4 or i == 2:
            ax.set_xlabel("IoU", fontweight='bold', fontsize=16)
        if i == 0 or i == 3:
            ax.set_ylabel("Score", fontweight='bold', fontsize=16)
        plt.legend()
        ax.set_title(f'{con_sco}')
        plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        ax.tick_params(axis='both', labelsize =12)
    # plt.rcParams["figure.figsize"] = (25, 4)
    plt.show()


def plot_spray_of_targets_2_lines_2(df, metric):
    """
    plot k1, k2 (on top of it) vs step size, iou, ccs
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(f'Hit rate (k1) for Step size, IoU, Classification confidence score '
                 f'and classification confidence score', fontweight='bold', fontsize=23)
    labels = df['IoU'].unique()
    df = df.sort_values(by=['IoU'], ascending=False)
    for i, con_sco in enumerate(np.linspace(0.5, 0.9, 5)):
        row = 0 if i < 3 else 1
        ax = axs[row, i % 3]
        if i % 2 == 1 and i % 3 == 2:
            continue
        metric_df = df[df['confidence_score'] == con_sco][['IoU', 'k_1', 'k_2', 'steps_gap']]
        metric_df = metric_df.sort_values(by=['IoU'], ascending=True)
        df_0 = metric_df[metric_df['steps_gap'] == 1]['k_1'] / 15
        df_1 = metric_df[metric_df['steps_gap'] == 2]['k_1'] / 15
        df_2 = metric_df[metric_df['steps_gap'] == 4]['k_1'] / 15
        ax.bar(labels - 0.03, df_0, width=0.03,  label='0.1m', edgecolor="black", color='green')
        ax.bar(labels, df_1, width=0.03, label='0.2m', edgecolor="black", color='lime')
        ax.bar(labels + 0.03, df_2, width=0.03, label='0.4m', edgecolor="black", color='greenyellow')
        ax.set_ylim(0, 1)
        if i == 3 or i == 4:
            ax.set_xlabel("IoU", fontweight='bold', fontsize=21)
        if i == 0 or i == 3:
            ax.set_ylabel("Hit rate", fontweight='bold', fontsize=22)
        plt.legend()
        ax.set_title(f'confidence score {con_sco}', fontweight='bold', fontsize=21)
        plt.xticks(df_0.IoU.unique())#[0.3, 0.4, 0.5, 0.6, 0.7])
        ax.tick_params(axis='both', labelsize=20)
    # plt.rcParams["figure.figsize"] = (25, 4)
    axs[0, 0].legend(loc="upper right")
    for ax in axs.flat:
        ## check if something was plotted
        if not bool(ax.has_data()):
            fig.delaxes(ax)
    plt.show()


def plot_spray_of_targets_2_lines_3(df, metric):
    """
    plot k1 + k2 (on top of it) vs step size, iou, ccs
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(f'Total sprays of targets for different overlapping size, '
                 f'IoU and classification confidence score', fontweight='bold', fontsize=23)
    labels = df['IoU'].unique()
    for i, con_sco in enumerate(np.linspace(0.5, 0.9, 5)):
        row = 0 if i < 3 else 1
        ax = axs[row, i % 3]
        if i % 2 == 1 and i % 3 == 2:
            continue
        metric_df = df[df['confidence_score'] == con_sco][['k_1', 'k_2', 'steps_gap']]
        df_0 = metric_df[metric_df['steps_gap'] == 1]['k_1'] + metric_df[metric_df['steps_gap'] == 1]['k_2']
        df_1 = metric_df[metric_df['steps_gap'] == 2]['k_1'] + metric_df[metric_df['steps_gap'] == 2]['k_2']
        df_2 = metric_df[metric_df['steps_gap'] == 4]['k_1'] + metric_df[metric_df['steps_gap'] == 4]['k_2']
        df_0 /= 15
        df_1 /= 15
        df_2 /= 15
        ax.bar(labels, df_0, width=0.03,  label='0.1, k_1', edgecolor="black", color='darkred')
        ax.bar(labels + 0.03, df_1, width=0.03, label='0.2, k_1', edgecolor="black", color='red')
        ax.bar(labels - 0.03, df_2, width=0.03, label='0.4, k_1', edgecolor="black", color='lightsalmon')
        ax.set_ylim(0, 1)
        if i == 3 or i == 4:
            ax.set_xlabel("IoU", fontweight='bold', fontsize=21)
        if i == 0 or i == 3:
            ax.set_ylabel("Score", fontweight='bold', fontsize=22)
        plt.legend()
        ax.set_title(f'{con_sco}', fontweight='bold', fontsize=21)
        plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        ax.tick_params(axis='both', labelsize=20)
    # plt.rcParams["figure.figsize"] = (25, 4)
    axs[0, 0].legend(loc="upper right")
    for ax in axs.flat:
        ## check if something was plotted
        if not bool(ax.has_data()):
            fig.delaxes(ax)
    plt.show()


def gt_counter(size):
    if size < 1.5:
        return 15
    elif size < 2.5:
        return 14
    return 12


def plot_k1_k2_spraying(df, show_limits):
    """
    plot k1, k2 (on top of it) vs step size, iou, ccs.
    Plot it in two lines [(0.5, 0.6, 0.7), (0.8, 0.9)
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(f'Spray rate (k1, k2) for Step size, IoU, and'
                 f' Classification confidence score', fontweight='bold', fontsize=23)
    labels = df['IoU'].unique()
    for i, con_sco in enumerate(np.linspace(0.5, 0.9, 5)):
        row = 0 if i < 3 else 1
        ax = axs[row, i % 3]
        if i % 2 == 1 and i % 3 == 2:
            continue
        labels = np.sort(labels)
        metric_df = df[df['confidence_score'] == con_sco][['IoU', 'k_1', 'k_2', 'steps_gap']]
        metric_df = metric_df.sort_values(by=['IoU', 'steps_gap'])
        metric_df['gt_count'] = metric_df.apply(lambda x: gt_counter(x.steps_gap), axis=1)
        df_0 = metric_df[metric_df['steps_gap'] == 1]['k_1'] / 15
        df_1 = metric_df[metric_df['steps_gap'] == 2]['k_1'] / 15
        df_2 = metric_df[metric_df['steps_gap'] == 4]['k_1'] / 15
        df_0_1 = metric_df[metric_df['steps_gap'] == 1]['k_2'] / 15
        df_1_1 = metric_df[metric_df['steps_gap'] == 2]['k_2'] / 15
        df_2_1 = metric_df[metric_df['steps_gap'] == 4]['k_2'] / 15

        if show_limits:  # draw red lines with the gt count of grape.
            df_0_2 = metric_df[metric_df['steps_gap'] == 1]['gt_count'] / 15
            df_1_2 = metric_df[metric_df['steps_gap'] == 2]['gt_count'] / 15
            df_2_2 = metric_df[metric_df['steps_gap'] == 4]['gt_count'] / 15
            ax.bar(labels - 0.03, df_0_2,  color='none', edgecolor='red', width=0.03)
            ax.bar(labels, df_1_2,   color='none', edgecolor='red', width=0.03)
            ax.bar(labels + 0.03, df_2_2, color='none', edgecolor='red', width=0.03)
        params_0 = {'edgecolor': "black", 'color': 'green', 'width': 0.03}
        params_1 = {'edgecolor': "black", 'color': 'lime', 'width': 0.03}
        params_2 = {'edgecolor': "black", 'color': 'lightgreen', 'width': 0.03}
        ax.bar(labels - 0.03, df_0,   label='0.1[m], k1', **params_0 )
        ax.bar(labels - 0.03, df_0_1,  bottom=df_0, label='0.1[m], k2', alpha=0.3,  **params_0)
        ax.bar(labels, df_1, label='0.2[m], k1', **params_1)
        ax.bar(labels, df_1_1, bottom=df_1,  label='0.2[m], k2', alpha=0.3, **params_1)
        ax.bar(labels + 0.03, df_2, label='0.4[m], k1', **params_2)
        ax.bar(labels + 0.03, df_2_1, bottom=df_2, label='0.4[m], k2', **params_2, alpha=0.3)
        ax.set_ylim(0, 1)
        if i == 3 or i == 4:
            ax.set_xlabel("IoU", fontweight='bold', fontsize=21)
        if i == 0 or i == 3:
            ax.set_ylabel("Spray rate", fontweight='bold', fontsize=22)
        # plt.legend() #FIXME- add legend()
        ax.set_title(f'Confidence score {con_sco}', fontweight='bold', fontsize=21)
        # plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        ax.tick_params(axis='both', labelsize=20)
    if not show_limits:
        axs[1, 1].legend(loc="upper right")
    # plt.rcParams["figure.figsize"] = (25, 4)
    for ax in axs.flat:
        # check if something was plotted, else delete plot (remove empty plots).
        if not bool(ax.has_data()):
            fig.delaxes(ax)
    plt.show()


if __name__ == '__main__':
    # three_d_two()
    path_sim = Path(r'D:\Users\NanoProject\simulations')
    table_0 = get_results_values(path_sim)
    table_1 = get_options_tested(path_sim)
    table_2 = pd.concat([table_1, table_0], axis=1)

    # todo: uncomment to write results to csv
    table_2.to_csv(r'C:\Users\Administrator\Desktop\grapes\results.csv')
    table_2 = table_2.apply(pd.to_numeric)
    metric = 'recall'

    plot_k1_k2_spraying(table_2, False) # K1, K2
    # plot_spray_of_targets_2_lines_2(table_2, metric=metric)  # K1
    # plot_spray_of_targets_2_lines_3(table_2, metric=metric) # K1 + K2
    # plot_spray_of_targets(table_2, metric=metric)
    # plot_spray_of_targets_2(table_2, metric=metric)
    # plot_spray_of_targets_3(table_2, metric=metric)
    # three_d_two_colors(table_2[[metric, 'IoU', 'confidence_score', 'steps_gap', 'k_1', 'k_2']], metric)
    # three_d(table_2[[metric, 'IoU', 'confidence_score', 'steps_gap']], metric)
    # three_d_plot(big_df)
    cs = 0.9

    print(table_0)
    print(table_1)
    print('total number is', len(table_0))
    # scatter_recall_precision(table_0, table_1)

    # for conf_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
    #     f1_chart(table_0, table_1[table_1['confidence_score'] == conf_val], conf_val)
    #     stack_bar_confidence(table_0, table_1[table_1['confidence_score'] == conf_val], conf_val)
    # [xyUnique, ignore, ixs] = unique(xy,'rows')

