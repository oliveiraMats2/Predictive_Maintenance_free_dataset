import numpy as np


class SaveNumpy:
    @staticmethod
    def save(vector, name_vector="actuals.npy"):
        return np.save(name_vector, vector)

    @staticmethod
    def load(name_vector="actuals.npy"):
        return np.load(name_vector)
