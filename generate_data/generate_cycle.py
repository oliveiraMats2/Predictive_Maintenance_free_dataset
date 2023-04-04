from build_signal_sintetic import BuildSignal
import numpy as np
from noise_signal import NoiseGaussian
import matplotlib.pyplot as plt
from h5_save import SaveData


class GenerateCycle:
    def __init__(self, slot=10):
        self.slot = slot

    @staticmethod
    def __concat_arrays(*args):
        concatenated_array = np.concatenate(args)
        return concatenated_array

    def generate_data_ciclic(self, repeat=5):
        crescent = BuildSignal.f_linear_crescent(60, 15)
        descending = BuildSignal.f_linear_descending(60, 15)

        constant_up = BuildSignal.f_constant(constant=60, qtd=200)
        constant_down = BuildSignal.f_constant(constant=15, qtd=200)

        self.data = np.transpose(np.concatenate(repeat * [constant_down, crescent, constant_up, descending]))
        return self.data


if __name__ == "__main__":
    generate = GenerateCycle()
    noise_signal = NoiseGaussian(amplitude=0.3)
    data = generate.generate_data_ciclic()

    data = noise_signal.apply_noise_on_signal(data)

    plt.scatter(np.arange(len(data)), data, s=1)

    SaveData.save_data(data, dir_data='../Datasets/sintetic_dataset/train_compressor_data.h5')

    data = noise_signal.apply_noise_on_signal(data)

    plt.scatter(np.arange(len(data)), data, s=1)

    SaveData.save_data(data, dir_data='../Datasets/sintetic_dataset/test_compressor_data.h5')
