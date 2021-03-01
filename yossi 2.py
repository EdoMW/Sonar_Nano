from scipy.signal import find_peaks
from scipy import signal, io
import matplotlib.pyplot as plt
import DAQ_BG
import numpy as np
def distance2(sig):
    f, t, s = signal.spectrogram(sig, fs=5e5, nfft=256, noverlap=200, mode='magnitude')
    new_s = s[30, :] / np.max(s[30, :])
    peaks, heights = find_peaks(new_s, height=0.4)
    print(peaks)
    if (len(peaks)>1):
        #emitted = t[peaks[0]]
        #received = -t[peaks[1]]
        # print(peaks)
        #plt.figure()
        #plt.plot(new_s)
        dist1 = abs((t[peaks[0]] - t[peaks[1]])) * 340 / 2
        if dist1 < 0.1:
            #received = -t[peaks[2]]
            # print(peaks)
            #plt.figure()
            #plt.plot(new_s)
            print('second peak')
            dist1 = abs((t[peaks[0]] - t[peaks[2]])) * 340 / 2
        # print(emitted)
        # print(received)
    else:
        print('no returning echo')
        dist1 = -1
    return dist1

record = DAQ_BG.rec()
dist = distance2(record)
print("record: type = ", type(record), "record = ", record)

# x = str(record).encode('utf-8', 'ignore')
# np.savetxt("C:\Drive\disance_record.csv", record, delimiter="," ) #TypeError: can only concatenate str (not "numpy.float64") to str
print(dist)