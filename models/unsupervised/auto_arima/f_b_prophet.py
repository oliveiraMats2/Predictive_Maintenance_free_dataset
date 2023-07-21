import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from fbprophet import Prophet

df = pd.read_csv('../../../Datasets/dataset_TPV_sensors/hex/payloadHex.csv')

df.rename(columns={"InletPressure":"y", "Time":"ds"}, inplace=True)
hto9p
print(df.keys())

model = Prophet(interval_width=0.9)

model.add_regressor("OutletPressure", standardize=False)
model.add_regressor("OutletTemperature", standardize=False)
model.add_regressor("InverterSpeed", standardize=False)
model.fit(df)

forecast = model.predict(df)