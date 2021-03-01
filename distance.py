from DAQ_BG import chirp_gen
from scipy.signal import find_peaks
from scipy import signal, io
import numpy as np


def correlation_dist(sig,chirp):
    # correlation = np.correlate(chirp,sig[int((chirp.size)*1.3):])
    correlation = np.correlate(chirp, sig)
    Pick = np.min(np.argsort(correlation)[0:2])
    dist = abs(correlation.size-Pick)*340/(5e5*2)
    return dist,abs(correlation.size-Pick)

def distance():
    chirp = chirpAmp = 1
    chirpTime = 0.005  # 5 milliseconds
    f0 = 35e3
    f_end = 90e3
    update_freq = 5e5 #D2A sampling rate
    sample_rate = 5e5  # A2D sample rate in Hz
    trapRel = 5 # trap to chirp relation

    chirp = chirp_gen(chirpAmp, chirpTime, f0, f_end, update_freq, trapRel)
    sig = io.loadmat("D:\8")['records'][:,1]
    sig = sig - np.mean(sig)
    sig = sig/np.max(sig)
    dist, Pick = correlation_dist(sig, chirp)
    # print(dist)
    #
    # # a,b,c = signal.spectrogram(sig, 5e5, nfft=400, noverlap=200, mode='magnitude')
    # # plt.pcolormesh(b,a,c)
    # plt.plot(sig, alpha=0.5)
    # plt.scatter(Pick,sig[Pick])
    # plt.show()
    return dist


def distance2(sig):
    f, t, s = signal.spectrogram(sig, fs=5e5, nfft=256, noverlap=200, mode='magnitude')
    new_s = np.sum(s[15:40], axis=0)
    peaks, heights = find_peaks(new_s, height=0.003, distance=5)
    new_peaks = peaks[1:]
    peak_ix = np.argmax(new_peaks)
    dist = abs((t[peaks[0]] - t[peaks[1]])) * 340 / 2
    return dist

if __name__ == '__main__':
    distance2()
