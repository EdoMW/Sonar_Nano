import nidaqmx.constants as con
import numpy as np
import nidaqmx
import scipy.signal as signal

import matplotlib.pyplot as plt
from preprocessing_and_adding import preprocess_one_record
np.set_printoptions(precision=3)
device = 'dev1'  # device name
ai_channel = 0
a0_channel = 0
recordTime = 0.024

chirpAmp = 2
chirpTime = 0.005  # 5 milliseconds
f0 = 35e3
f_end = 90e3
update_freq = 5e5 #D2A sampling rate
sample_rate = 5e5  # A2D sample rate in Hz
trapRel = 5 # trap to chirp relation


def meas(device, TransData, mic, speaker, sample_rate, record_time):

    record_length = int(sample_rate * record_time)
    with nidaqmx.Task() as ai_task, nidaqmx.Task() as ao_task:

        ai_task.ai_channels.add_ai_voltage_chan(device + '/ai' + str(mic))  # add analog input channel
        ao_task.ao_channels.add_ao_voltage_chan(device + '/ao' + str(speaker))  # add analog output channel
        ao_task.timing.cfg_samp_clk_timing(sample_rate,  # set the output channel configuration
                                           sample_mode=con.AcquisitionType.FINITE,
                                           samps_per_chan=record_length)
        ai_task.timing.cfg_samp_clk_timing(sample_rate, source='ao/SampleClock',
                                           # set the output channel configuration (clock paired with AO channel)
                                           samps_per_chan=record_length)

        # set the record time to the transmitted signal
        record_signal = np.append(TransData, np.zeros(record_length - np.size(TransData)))

        ao_task.write(record_signal, auto_start=False)  # write the transmission to the channel but dont start yet

        # start the AI and AO together
        ai_task.start()
        ao_task.start()

        # read the data to AI channel
        collected_data = ai_task.read(number_of_samples_per_channel=record_length)

        ao_task.wait_until_done()
        ai_task.wait_until_done()

        # return the data as a numpy array
        return np.asarray(collected_data)


def chirp_gen(chirp_amp, chirpTime, f0, f_end, update_frq, trap_rel):
    # build the chirp
    chirp = signal.chirp(np.linspace(0, chirpTime, int(5e5*chirpTime)), f0, chirpTime, f_end)

    # build the trap for the chirp
    trap = np.append(np.linspace(1 / (update_frq * chirpTime / (2 * trap_rel)),
                                 1, int(update_frq * chirpTime / (2 * trap_rel))),
                     np.append(np.ones(int(update_frq * chirpTime - update_frq * chirpTime / trap_rel)),
                               np.linspace(1, 1 / (update_frq * chirpTime / (2 * trap_rel)),
                                           int(update_frq * chirpTime / (2 * trap_rel)))))

    traped_chirp = chirp * trap * chirp_amp
    return traped_chirp


def rec():
    # device = nidaqmx.system.system.System.devices  # device name

    transmition_Chirp = chirp_gen(chirpAmp, chirpTime, f0, f_end, update_freq, trapRel)
    A=meas(device, transmition_Chirp, ai_channel, a0_channel, sample_rate, recordTime)
    #plt.figure()
    #plt.plot(A)
    #plt.show()
    return meas(device, transmition_Chirp, ai_channel, a0_channel, sample_rate, recordTime)


if __name__ == "__main__":
    rec()


    print("emited")