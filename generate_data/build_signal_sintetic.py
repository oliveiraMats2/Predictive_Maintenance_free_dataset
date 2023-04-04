import numpy as np


class BuildSignal:
    @staticmethod
    def f_constant(constant=1, qtd=100):
        return np.array(qtd * [constant])

    @staticmethod
    def f_linear_crescent(max_up=120):
        return np.array([x for x in range(max_up)])

    @staticmethod
    def f_linear_descending(max_down=120):
        return np.array([x for x in range(max_down, -1, -1)])
