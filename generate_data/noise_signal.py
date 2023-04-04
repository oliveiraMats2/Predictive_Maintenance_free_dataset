import numpy as np


class NoiseGaussian:
    def __init__(self, amplitude=1):
        self.noise_amplitude = amplitude

    def signal_noise_generate(self, len_signal):
        return np.random.normal(0, self.noise_amplitude, len_signal)

    def apply_noise_on_signal(self, signal):
        return self.signal_noise_generate(len(signal)) + signal
