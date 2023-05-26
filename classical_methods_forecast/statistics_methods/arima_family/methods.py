from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX


def auto_regression(train, test):
    model = AutoReg(train, lags=1)
    model_fit = model.fit()

    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)

    return yhat


def moving_average(train, test):
    model = ARMA(train, order=(0, 1))
    model_fit = model.fit(disp=False)

    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)

    return yhat


def autoregressive_moving_average(train, test):
    model = ARMA(train, order=(1, 2))

    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)
    return yhat


def autoregressive_integrated_moving_average(train, test):
    model = ARMA(train, order=(1, 2))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)
    return yhat


def seasonal_autoregressive_integrated_moving_average(train, test):
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(train), len(train) + len(test) - 1, typ='levels')
    return yhat


def seasonal_Autoregressive_integrated_moving_average_with_exogenous_regressors(train, test):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 2))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)

    return yhat


def vector_autoregression_moving_average(train, test):
    # fit model
    model = VARMAX(train, order=(0, 1, 2))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.forecast(steps=len(test))
    return yhat

def simple_exp_smoothing(train, test):
    # fit model
    model = SimpleExpSmoothing(train)
    model_fit = model.fit()
    # make prediction
    y_hat = model_fit.predict(len(train), len(train) + len(test) - 1)
    return y_hat
