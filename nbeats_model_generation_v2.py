import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
import pandas as pd
import matplotlib.pyplot as plt
from classical_methods_forecast.statistics_methods.metrics import MeasureMultiSensors
import torch
import torch.optim as optim
def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())


# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Define a dataset
data_path_hex = "Datasets/dataset_TPV/base_23042023_A/hex/hex.csv"
data_path_ite = "Datasets/dataset_TPV/base_23042023_A/ite/ite.csv"
data_path_wise_x = "Datasets/dataset_TPV/base_23042023_A/wise/x.csv"
data_path_wise_y = "Datasets/dataset_TPV/base_23042023_A/wise/y.csv"
data_path_wise_z = "Datasets/dataset_TPV/base_23042023_A/wise/z.csv"

data_hex = pd.read_csv(data_path_hex)
data_ite = pd.read_csv(data_path_ite)
data_wise_x = pd.read_csv(data_path_wise_x)
data_wise_y = pd.read_csv(data_path_wise_y)
data_wise_z = pd.read_csv(data_path_wise_z)

medidas_hex = data_hex[['InletPressure','OutletPressure','OutletTemperature']]
medidas_ite = data_ite[['phaseA_voltage','phaseB_voltage','phaseC_voltage', 'phaseA_current', 'phaseB_current', 'phaseC_current']]

medidas_wise_x = data_wise_x[['OAVelocity']]
medidas_wise_y = data_wise_y[['OAVelocity']]
medidas_wise_z= data_wise_z[['OAVelocity']]

# df_hex = pd.DataFrame({'Time': pd.to_datetime(data_hex[['Time']]), 'Data': medidas_hex})
# df_ite = pd.DataFrame({'Time': pd.to_datetime(data_ite[['Time']]), 'Data': medidas_ite})
# df_wise = pd.DataFrame({'Time': pd.to_datetime(dat_wise_x[['Time']]), 'Data': pd.concat(medidas_wise_x, medidas_wise_y, medidas_wise_z, axis=1)})


num_amostras_sensor1 = len(data_hex[['Time']])
num_amostras_sensor2 = len(data_ite[['Time']])
num_amostras_sensor3 = len(data_wise_x[['Time']])

num_amostras_min = min(num_amostras_sensor3, num_amostras_sensor2, num_amostras_sensor1)

# Combine all variables into a single dataframe
medidas = pd.concat([medidas_hex, medidas_ite, medidas_wise_x, medidas_wise_y, medidas_wise_z], axis=1).head(num_amostras_min)

#Normalize by columns

medidas_norm = medidas.apply(min_max_normalize)
# Split the dataset into training and test sets
medidas_array = medidas_norm.values

series = TimeSeries.from_values(medidas_array)

# Split the dataset into training and test sets
train_size = int(len(series) * 0.8)
train_series = series[:train_size]
test_series = series[train_size:]


#print(test_series.values())
# Create an NBEATS model for multivariate time series forecasting
model = NBEATSModel(input_chunk_length=1000, output_chunk_length=1, n_epochs=100, optimizer_kwargs={"lr": 1e-5})

# Definir otimizador e taxa de aprendizado
learning_rate = 0.001

# Treinar o modelo
model.fit(train_series)


# Treinar o modelo
#model.fit(train_series)
# Make predictions on the test set
pred_series = model.predict(n=len(test_series), series=train_series)


test_series_df = pd.DataFrame(test_series.values())
test_series_df.columns = ['InletPressure','OutletPressure','OutletTemperature',
                          'phaseA_voltage','phaseB_voltage','phaseC_voltage',
                          'phaseA_current', 'phaseB_current', 'phaseC_current',
                          'OAVelocity_x','OAVelocity_y', 'OAVelocity_z']

pred_series_df = pd.DataFrame(pred_series.values())
pred_series_df.columns = ['InletPressure','OutletPressure','OutletTemperature',
                          'phaseA_voltage','phaseB_voltage','phaseC_voltage',
                          'phaseA_current', 'phaseB_current', 'phaseC_current',
                          'OAVelocity_x','OAVelocity_y', 'OAVelocity_z']

metrics_NBeats = MeasureMultiSensors()
metrics_NBeats_mse= metrics_NBeats.mean_square_error(test_series_df, pred_series_df)
metrics_NBeats_mae=metrics_NBeats.mean_absolute_error(test_series_df,pred_series_df)
metrics_NBeats_smape=metrics_NBeats.smape_loss(test_series_df,pred_series_df)

print(metrics_NBeats_mse)
print(metrics_NBeats_mae)
print(metrics_NBeats_smape)

# # Print the predictions
# print(pred_series.pd_dataframe())

# Plot each variable separately
n_variables = series.width
for i in range(n_variables):
    plt.figure(figsize=(8,5))  # create a new figure for each variable

    # Extract the i-th component of the multivariate series, both for test and pred series
    test_series_i = test_series.univariate_component(i)
    pred_series_i = pred_series.univariate_component(i)

    # plot the test (actual) and forecasted values
    test_series_i.plot(label='actual')
    pred_series_i.plot(label='forecast')

    plt.title(f'Variable {i+1}')
    plt.legend()
    plt.show()
