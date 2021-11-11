# preprocessing for new data for testing by adding noise or zero columns

import scipy.io
import scipy.signal
import os
import numpy as np
import pickle
from keras.utils import to_categorical
from sklearn.cluster import DBSCAN
from sklearn import linear_model
import matplotlib.pyplot as plt
import cv2 as cv

def preprocess_one_record(record, counter, adding):
    X_test_per_rec = []
    # mat = scipy.io.loadmat(os.path.join(d, file))['records']
    # if 'leaves' in d:
    #     y_int = 0
    # elif 'fruit' in d:
    #     y_int = 1
    # y_binary = to_categorical(y_int, num_classes=2)
    # for channel in range(mat.shape[1]):
    X_samples = np.zeros([36, 980, 1])
    # fft
    # f, t, Sxx = scipy.signal.spectrogram(mat[:, channel], fs=500000, window=('tukey', 0.25), nperseg=None,
    #                                      noverlap=250, nfft=256,
    #                                      detrend='constant', return_onesided=True, scaling='density', axis=-1,
    #                                      mode='magnitude')
    f, t, Sxx = scipy.signal.spectrogram(record, fs=250000, window=('tukey', 0.25), nperseg=None,
                                         noverlap=250, nfft=256,
                                         detrend='constant', return_onesided=True, scaling='density', axis=-1,
                                         mode='magnitude')

    eps = 1e-10
    Sxx = np.log10(Sxx + eps)
    Sxx_before_process = Sxx
    plt.figure()
    img = plt.plot(record)
    #img = plt.pcolormesh(Sxx)
    # a_string = "C:/Users/Administrator/Desktop/spac/check17.jpeg"
    a_string = "D:/Users/NanoProject/spac/check17.jpeg"
    print("print.shape", np.shape(Sxx))
    plt.imsave(a_string, Sxx)
    cv.waitKey(0)
    cv.destroyAllWindows()
    f = f[15:-78]
    Sxx = Sxx[15:-78, :]

    # threshold before clustering transmit and recieve
    samples = np.array(np.where(Sxx > np.percentile(Sxx, 98)))

    # clustering
    clustering = DBSCAN(eps=3., min_samples=8).fit(samples.transpose())
    # sns.scatterplot(t[samples[1]], f[samples[0]], hue=clustering.labels_)
    clustering.labels_[0] = clustering.labels_[1]
    c_labels_list, c_counts = np.unique(clustering.labels_, return_counts=True)
    if len(c_counts) > 1:
        c_i, c_j = np.argsort(c_counts)[-2:]
        in_cluster_i = clustering.labels_ == c_labels_list[c_i]
        in_cluster_j = clustering.labels_ == c_labels_list[c_j]
        t_avg_cluster_i = np.average(t[samples[1]][in_cluster_i.nonzero()])
        t_avg_cluster_j = np.average(t[samples[1]][in_cluster_j.nonzero()])
        if t_avg_cluster_i < t_avg_cluster_j:
            c_v = c_labels_list[c_j]
        else:
            c_v = c_labels_list[c_i]

        in_cluster = clustering.labels_ == c_v
        # not_in_cluster = 1 - in_cluster
    else:
        in_cluster = clustering.labels_ == c_labels_list[0]

    # finding the fitted diag to cut the transmit signal
    diag_line_dic = {}
    for frq in np.unique(np.sort(f[samples[0][in_cluster.nonzero()]])):
        diag_line_dic[frq] = 1000
    for index, sample_t in enumerate(t[samples[1]][in_cluster.nonzero()]):
        if diag_line_dic[f[samples[0]][in_cluster.nonzero()][index]] > sample_t:
            diag_line_dic[f[samples[0]][in_cluster.nonzero()][index]] = sample_t

    xs = [x for x in diag_line_dic.values()]
    ys = [y for y in diag_line_dic.keys()]
    xs = np.array(xs).reshape(-1, 1)
    ransac = linear_model.RANSACRegressor()
    if xs.shape[0] > 1:
        ransac.fit(xs, ys)
    else:
        counter += 1
        print(counter)
        # continue
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(xs.min(), xs.max(), 0.0005)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    y_up = np.max(line_y_ransac)
    y_down = np.min(line_y_ransac)
    t_start_recieve = np.min(line_X)
    t_end_recieve = np.max(line_X)

    m = (y_up - y_down) / (t_end_recieve - t_start_recieve + eps)
    b = y_down - m * (t_start_recieve - 0.0012)  # if fs = 500000 so should be b = y_down - size_of_step * (t_start_recieve - 0.0012/2)

    # delete the transmit signal
    min_Sxx = np.min(Sxx)
    t_min = 1000
    it_min = 10000
    for i_t, t_s in enumerate(t):
        for i_f, f_s in enumerate(f):
            if f_s > (m * t_s + b):
                Sxx[i_f, i_t] = int(min_Sxx)
            else:
                if i_f == 0:
                    if t_min > t_s:
                        t_min = t_s
                        it_min = i_t

    if Sxx[:, it_min: it_min + 980].shape[1] < 980:
        counter += 1
        print(counter)
        # continue
    X_samples[:, :, 0] = Sxx[:, it_min: it_min + 980]


    # adding noise instead of the transmited signal or adding zero columns

    if adding == "noise":
        background_ind = X_samples.shape[1] - 200
        bg_power = np.percentile(X_samples[:, background_ind:, 0], 0.03)
        factor = bg_power / int(min_Sxx)
        for i_ind in range(X_samples.shape[0]):
            def inner(X_samples, i_ind):
                for j_ind in range(X_samples.shape[1]):
                    if X_samples[i_ind, j_ind, 0] == int(min_Sxx):
                        X_samples[i_ind, j_ind, 0] = factor * Sxx_before_process[
                            -X_samples.shape[0] + i_ind, -X_samples.shape[1] + j_ind]
                    else:
                        return X_samples

            inner(X_samples, i_ind)
        X_test_per_rec.append(X_samples)
    elif adding=="zero_cols":
        num_of_zero_columns = np.random.randint(0, 50)
        new_samples = np.zeros([X_samples.shape[0], X_samples.shape[1] + 50, 1]) + int(min_Sxx)
        new_samples[:, num_of_zero_columns:num_of_zero_columns + 980, :] = X_samples
        X_test_per_rec.append(new_samples)

    return counter, X_test_per_rec


if __name__ == '__main__':
    counter = 0
    X_test = []
    data_path = r'C:\Users\Moran\Documents\project_2deg\grapes_data'
    for r, d, f in os.walk(data_path):
        for file in f:
            if '.mat' in file:
                counter, X_test_per_rec = preprocess_one_record(r, file, counter, 'noise')
                X_test = X_test + X_test_per_rec

    # saving data into pkl file
    with open("spec_grapes_noise_500.pkl", "wb") as pickle_out:
        pickle.dump((X_test), pickle_out)
