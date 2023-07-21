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
import pandas as pd

figure(figsize=(16, 4), dpi=80)

df = pd.read_csv('../../Datasets/dataset_TPV_sensors/ite/payloadITE.csv')

# df = df[["phaseA_current", "phaseB_current", "phaseC_current",
#          "phaseA_voltage", "phaseB_voltage", "phaseC_voltage", "time"]]

# df_1 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/x.csv')
# df_2 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/y.csv')
# df_3 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/z.csv')
#
# df = pd.DataFrame()
# df["OAVelocity_x"] = df_1["OAVelocity"].astype('float64')
# df["OAVelocity_y"] = df_2["OAVelocity"].astype('float64')
# df["OAVelocity_z"] = df_3["OAVelocity"].astype('float64')

# df["time"] = df_3["Time"]
# Exibir o DataFrame resultante
print(df)

# Pegando os sensores temos. Temperatura (Outlet Temperature no hexa),
# Pressão de Entrada (Inlet Pressure no hexa),
# Pressão de Saída (Outlet Pressure no hexa),

# e Vibração (OAVelocity eixos x, y e z no wise)

df["time"] = df["Time"]
df = df.drop("time", axis=1)

# df = df.drop("phaseA_tc_config", axis=1)
# df = df.drop("phaseB_tc_config", axis=1)
# df = df.drop("phaseC_tc_config", axis=1)

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
