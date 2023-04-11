# from filters.filter_avoid_amplitude import FiltersAvoidLowFreq
import pandas as pd

if __name__ == '__main__':
    import numpy as np
    from scipy.signal import butter, filtfilt

    # create a sample 1D signal
    fs = 92554  # sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)
    # x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

    payload_hex = "../Datasets/dataset_TPV_sensors/hex/payloadHex.csv"

    df = pd.read_csv(payload_hex)

    # define the default filter parameters
    order = 1
    cutoff_freq = 0.01 * fs / 2  # Hz
    nyquist_freq = 0.5 * fs
    normal_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff_freq, btype='low', analog=False)

    # apply zero-phase filtering
    y = filtfilt(b, a, df["InverterSpeed"].tolist())

    # plot the original and filtered signals
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(t, df["InverterSpeed"].tolist(), 'b', alpha=0.5)
    plt.plot(t, y, 'r')
    plt.legend(['Original', 'Filtered'])
    plt.show()