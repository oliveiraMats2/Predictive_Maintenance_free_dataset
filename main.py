import pandas as pd

dataset = pd.read_csv("uci_base_machine_learning.csv")

dataset.keys()

x_dataset = dataset.drop(columns=['UDI', 'Product ID', 'Type','Tool wear [min]',
                                  'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF',
                                  'RNF'])

y_dataset = dataset.drop(columns=['UDI', 'Product ID', 'Type', 'Air temperature [K]',
                               'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                               'Tool wear [min]', 'Machine failure'])

channels = list(x_dataset.keys())

sequence = 30

import numpy as np

dict_features_x = {}
#dict_keys(['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]'])
matrix_channels = np.array([np.array((x_dataset[feature].tolist())) for feature in x_dataset.keys()])

context = 5

