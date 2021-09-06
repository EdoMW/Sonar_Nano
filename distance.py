from DAQ_BG import chirp_gen
from scipy.signal import find_peaks
from scipy import signal, io
import numpy as np
import g_param


def correlation_dist(sig, chirp):
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


def load_sonar_dist(mask_id):
    """
    loading the sonar distance parameters t,s
    :param mask_id: id of the grape
    :return: f (1), t, s
    """
    t = g_param.read_write_object.read_sonar_dist_t_to_csv(mask_id)
    s = g_param.read_write_object.read_sonar_dist_s_to_csv(mask_id)
    return 1, t, s  # 1 (first argument) has no use for now.


def write_sonar_dist(t, s, mask_id):
    """
    writing the sonar distance parameters t,s
    :param t: t of distance reading
    :param s: s of distance reading
    :param mask_id: id of the grape
    """
    t = g_param.read_write_object.write_sonar_dist_t_to_csv(t, mask_id)
    s = g_param.read_write_object.write_sonar_dist_s_to_csv(s, mask_id)


def get_t_s(sig, mask_id):
    if g_param.process_type == "load":
        f, t, s = load_sonar_dist(mask_id)
    else:
        f, t, s = signal.spectrogram(sig, fs=5e5, nfft=256, noverlap=200, mode='magnitude')
        if g_param.process_type == "record":
            write_sonar_dist(t, s, mask_id)
    return f, t, s


def distance2(sig, mask_id):
    """
    calc distance to grape
    :param sig: signal
    :param mask_id: id of the grape
    :return: distance to grape in Meters
    """
    f, t, s = get_t_s(sig, mask_id)
    new_s = s[30, :] / np.max(s[30, :])
    peaks, heights = find_peaks(new_s, height=0.4)
    # new_s = np.sum(s[15:40], axis=0)
    # peaks, heights = find_peaks(new_s, height=0.003, distance=5)
    try:
        dist = abs((t[peaks[0]] - t[peaks[1]])) * 340 / 2
        if dist < 0.1:
            print('second peak')
            dist = abs((t[peaks[0]] - t[peaks[2]])) * 340 / 2
        # dist = abs((t[peaks[0]] - t[peaks[1]])) * 340 / 2
    except IndexError:
        while True:
            dist = input("Please enter distance to grape in Meters: ")
            try:
                float(dist)
                if float(dist) < 0.35:
                    print("Check again. grape is too close")
                else:
                    break
            except ValueError:
                print("enter float number")

    # Old version: Edo 11.4.21 changed to try,catch + input
    # new_peaks = peaks[1:]
    # peak_ix = np.argmax(new_peaks)
    # dist = abs((t[peaks[0]] - t[peaks[1]])) * 340 / 2
    return dist


if __name__ == '__main__':
    distance2()
