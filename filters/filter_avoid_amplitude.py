import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd


#
class FiltersAvoidLowFreq:
    def __init__(self, points, value_cutoff_freq=0.01, order=1):
        self.fs = points
        self.cutoff_freq = value_cutoff_freq * self.fs / 2
        self.nyquist_freq =  0.5 * self.fs
        self.normal_cutoff_freq = self.cutoff_freq / self.nyquist_freq
        self.order = order

    def apply(self, signal):
        b, a = butter(self.order, self.normal_cutoff_freq, btype='low', analog=False)
        # apply zero-phase filtering
        return filtfilt(b, a, signal)

    def filter_multiple_sensor(self, df_signal, *signals_keys):
        dict_of_signals = {}

        for signal_key in signals_keys:
            x = df_signal[signal_key].tolist()
            x_signal_filter = self.apply(x)

            dict_of_signals[signal_key] = x_signal_filter

        return pd.DataFrame(dict_of_signals)

