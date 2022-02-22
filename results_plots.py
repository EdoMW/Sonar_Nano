import pandas as pd
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D
from matplotlib import cm
import matplotlib as mpl


def read_file(iou_value):
    path = r'F:\פרויקט\סיכום תזהההה\volcani_test_results'
    file = f'eval_{iou_value}.csv'
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path)
    df['confidence_score'] = iou_value
    df.rename(columns={'Unnamed: 0': 'iou'}, inplace=True)
    return df


def plot_mean_std(df, iou_val):
    x = df.index
    y_r = df.recall
    y_p = df.precision
    yerr_r = df.r_std
    yerr_p = df.p_std
    fig, ax = plt.subplots()
    trans1 = Affine2D().translate(-0.01, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.01, 0.0) + ax.transData
    ax.errorbar(x, y_r, yerr=yerr_r, marker="o", linestyle="none", capsize=6, transform=trans1,label="Recall")
    ax.errorbar(x, y_p, yerr=yerr_p, marker="o", linestyle="none", capsize=6, transform=trans2, label="Precision")
    plt.xlabel("IoU")
    plt.ylabel("Score")
    plt.legend()
    plt.title(f'Recall, Precision mean and std, using {iou_val} confidence score value')
    plt.show()


def plot_recall_precision_per_cs():
    for iou in np.linspace(0.5, 0.9, 5):
        df_5 = read_file(iou)
        plot_mean_std(df_5, iou_val=iou)


def merge_tables():
    df = pd.DataFrame()
    for iou in np.linspace(0.5, 0.9, 5):
        df_temp = read_file(iou)
        df = pd.concat([df, df_temp])
    df = df.reset_index(drop=True)
    print(df)
    return df


def r_p_curve(df, confidence_score):
    # precision recall curve
    df = df[df['confidence_score'] == confidence_score]
    print(df)
    precision = df['recall']
    recall = df['precision']
    f1_score = df['f1']

    X = np.linspace(0.3, 0.8, 6)
    plt.bar(X, precision, color='b', width=0.025, label='Precision')
    plt.bar(X + 0.025, recall, color='g', width=0.025, label='Recall')
    plt.bar(X + 0.050, f1_score, color='r', width=0.025, label='F1 score')
    plt.title(f'Mean values for Recall, Precision, F1 score \n using {confidence_score} confidence score value')
    plt.xlabel("IoU")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()



def three_d_plot(df):
    result = [['122', '109', '2343', '220', '19'],
              ['15', '407', '37', '10', '102'],
              ['100', '100', '100', '100', '100'],
              ['113', '25', '19', '31', '112'],
              ['43', '219', '35', '33', '14'],
              ['132', '108', '256', '119', '14'],
              ['22', '48', '352', '51', '438']]
    result = np.array(result, dtype=np.int)
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax1 = fig.add_subplot(111, projection='3d')
    xlabels = np.array(['10/11/2013', '10/12/2013', '10/13/2013',
                        '10/14/2013', '10/15/2013'])
    xpos = np.arange(xlabels.shape[0])
    ylabels = np.array(['A1', 'C1', 'G1', 'M1', 'M2', 'M3', 'P1'])
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = result
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos

    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.w_yaxis.set_ticklabels(ylabels)

    values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
    plt.show()


from mpl_toolkits.mplot3d import Axes3D
def three_d(df, metric):
    # thickness of the bars
    dx, dy = .05, .05
    # prepare 3d axes
    # ax, fig = plt.figure(figsize=(10, 6),subplot_kw={'projection': ccrs.PlateCarree()}))
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Standard')
    # set up positions for the bars
    xpos = np.linspace(0.3, 0.8, 6)
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
    cb = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    # put the column / index labels
    ax.w_xaxis.set_ticklabels([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
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

    plt.show()


def anova_test():
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    #perform two-way ANOVA
    model = ols('height ~ C(water) + C(sun) + C(water):C(sun)', data=df).fit()
    sm.stats.anova_lm(model, typ=2)


if __name__ == '__main__':
    big_df = merge_tables()
    metric = 'f1'  # recall/ precision/ f1
    three_d(big_df[[metric, 'iou', 'confidence_score']], metric)
    # three_d_plot(big_df)
    cs = 0.9
    # r_p_curve(big_df, cs)
    plot_recall_precision_per_cs()
    # anova_test()
