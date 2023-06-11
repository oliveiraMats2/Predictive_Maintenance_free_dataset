from sktime.datasets import load_airline
from sktime.forecasting.ets import AutoETS

y = load_airline()
forecaster = AutoETS(auto=True, n_jobs=-1, sp=12)
forecaster.fit(y)

y_pred = forecaster.predict(fh=[1, 2, 3])
