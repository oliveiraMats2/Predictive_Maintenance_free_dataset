import numpy as np


class BuildSignal:
    @staticmethod
    def f_constant(constant=1, qtd=100):
        return np.array(qtd * [constant])

    @staticmethod
    def f_linear_crescent(max_up=60, init=15):
        return np.array([x for x in range(init, max_up)])

    @staticmethod
    def f_linear_descending(max_down=60, end_value=5, step=-1):
        return np.array([x for x in range(max_down, end_value, step)])

    @staticmethod
    def f_linear_crescent_array(begin, end, step):
        return np.array([round(i, 10) for i in list(np.arange(end, begin-step, -step))])

    @staticmethod
    def f_linear_descending_array(begin, end, step):
        return np.array([round(i, 10) for i in list(np.arange(begin, end+step, step))])

