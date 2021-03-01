import check_install
# check_install.main()
import DAQ_BG
import matplotlib.pyplot as plt
from scipy import signal


def main():
    record = DAQ_BG.main()

    plt.figure(1)
    plt.plot(record)

    plt.figure(2)
    a,b,c = signal.spectrogram(record)
    plt.pcolor(b,a,c)
    plt.show()



if __name__ == '__main__':
    main()

