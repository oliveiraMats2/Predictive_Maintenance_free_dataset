from classical_methods_forecast.statistics_methods.arima_family.methods import *


class MethodsStatistics:
    @staticmethod
    def auto_regression(train, test):
        print("---------------- auto_regression")
        return auto_regression(train, test)

    @staticmethod
    def moving_average(train, test):
        print("---------------- moving_average")
        return moving_average(train, test)

    @staticmethod
    def autoregressive_moving_average(train, test):
        print("---------------- autoregressive_moving_average")
        return autoregressive_moving_average(train, test)

    @staticmethod
    def autoregressive_integrated_moving_average(train, test):
        print("---------------- autoregressive_integrated_moving_average")
        return autoregressive_integrated_moving_average(train, test)

    @staticmethod
    def seasonal_autoregressive_integrated_moving_average(train, test):
        print("---------------- seasonal_autoregressive_integrated_moving_average")
        return seasonal_autoregressive_integrated_moving_average(train, test)

    @staticmethod
    def seasonal_Autoregressive_integrated_moving_average_with_exogenous_regressors(train, test):
        print("---------------- seasonal_Autoregressive_integrated_moving_average_with_exogenous_regressors")
        return seasonal_Autoregressive_integrated_moving_average_with_exogenous_regressors(train, test)

    @staticmethod
    def vector_autoregression_moving_average(train, test):
        print("---------------- Vector Autoregression Moving-Average")
        return vector_autoregression_moving_average(train, test)