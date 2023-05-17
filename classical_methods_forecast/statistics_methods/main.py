import matplotlib.pyplot as plt
from arima_family.statistics_method_classic import MethodsStatistics
import numpy as np
import pandas as pd
import time
from utils import filter_dataframe
from metrics import avaliable_vector_auto_regressive_model
from arima_family.print_statistics import PrintStatistics


import numpy as np
import warnings
import matplotlib.pyplot as plt



# 2023-02-16 02:30:31
#Fim: 2023-03-09 17:22:07
from matplotlib.pyplot import figure

figure(figsize=(16, 4), dpi=80)

df = pd.read_csv('../../Datasets/dataset_TPV_sensors/ite/payloadITE.csv')

#2023-02-16 02:30:31
#Fim: 2023-03-09 17:22:07

df["Time"] = df["time"]
df = df.drop("time", axis=1)

df = df.drop("phaseA_tc_config", axis=1)
df = df.drop("phaseB_tc_config", axis=1)
df = df.drop("phaseC_tc_config", axis=1)

df = filter_dataframe(df,
                      "2023-02-16 02:30:31",
                      "2023-03-09 17:22:07")

print(df["Time"].loc[df.index[0]], df["Time"].loc[df.index[-1]])

df = df.drop("Time", axis=1)

slice_train_test = 3*(len(df)//4) # 75%
train = df[:slice_train_test]
test = df[slice_train_test:]

print_statistics = PrintStatistics()

print_statistics(train,
                 test,
                 MethodsStatistics.vector_autoregression_moving_average,
                 avaliable_vector_auto_regressive_model)
