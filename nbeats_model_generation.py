import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
import pandas as pd


# Defina um conjunto de dados de exemplo com mais pontos
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
medidas_wise = pd.concat([data_wise_x['OAVelocity'], data_wise_y['OAVelocity'], data_wise_z['OAVelocity']], axis=1)

# Combine all variables into a single dataframe
medidas = pd.concat([medidas_hex, medidas_wise, medidas_ite], axis=1)

# Create a Darts TimeSeries object from the multivariate data
series = TimeSeries.from_dataframe(medidas)

# Split the dataset into training and test sets
train_size = int(len(series) * 0.8)
train_series = series[:train_size]
test_series = series[train_size:]

# Create an NBEATS model for multivariate time series forecasting
model = NBEATSModel(input_chunk_length=100, output_chunk_length=10, n_epochs=100)
# Fit the model to the training data
model.fit(train_series)

# Make predictions on the test set
prediction = model.predict(len(test_series))

# Print the predictions
print(prediction.to_dataframe())