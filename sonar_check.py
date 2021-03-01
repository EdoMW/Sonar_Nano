from scipy.signal import find_peaks
from scipy import signal, io
import matplotlib.pyplot as plt
import DAQ_BG
import numpy as np


def distance2(sig):
    f, t, s = signal.spectrogram(sig, fs=5e5, nfft=256, noverlap=200, mode='magnitude')
    # print(np.shape(s))
    # s = s[0, :,:]
    # new_s = s[30, :] / np.max(s[30, :])
    new_s=np.sum(s[15:40],axis=0)

    peaks, heights = find_peaks(new_s, height=0.3, distance=5)
    emitted = t[peaks[0]]
    new_peaks = peaks[1:]
    peak_ix = np.argmax(new_peaks)
    received = -t[new_peaks[peak_ix]]
    # print(heights)
    # print(new_peaks)
    plt.figure()
    plt.plot(new_s)
    dist = abs((t[peaks[0]] - t[peaks[1]])) * 340 / 2
    # print(emitted)
    # print(received)
    return dist





real_dist = 60
counter = 1
while True:
    input("press enter")
    record = DAQ_BG.rec()
    dist = distance2(record)
    print("distance = ", dist)
    import csv
    # str = "C:\Drive\exp_2\cm_60\disance_record_{}_{}_{}.csv"
    # np.savetxt(str.format(dist, real_dist, counter), record, delimiter="," )
    counter += 1



# x = str(record).encode('utf-8', 'ignore')
# np.savetxt("C:\Drive\disance_record.csv", record, delimiter="," ) #TypeError: can only concatenate str (not "numpy.float64") to str
# print(dist)
#
# import csv
#
# f = open('C:\Drive\disance_record.csv', 'w')
# for item in record:
#     f.write(','.join([str(x) for x in item]) + '\n')
# f.close()

