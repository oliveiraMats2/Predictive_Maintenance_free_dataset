from build_signal_sintetic import BuildSignal
import numpy as np


class GenerateCycle:
    def __init__(self, slot=10):
        self.slot = slot

    @staticmethod
    def __concat_arrays(*args):
        concatenated_array = np.concatenate(args)
        return concatenated_array

    def generate_data_ciclic(self, repeat=5):
        crescent = BuildSignal.f_linear_crescent(120)
        descending = BuildSignal.f_linear_descending(120)

        constant_up = BuildSignal.f_constant(constant=60, qtd=200)
        constant_down = BuildSignal.f_constant(constant=15, qtd=200)

        self.data = np.array(repeat * [constant_down, crescent, constant_up, descending])
        return self.data
