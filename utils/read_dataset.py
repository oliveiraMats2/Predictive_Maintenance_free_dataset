import h5py
import numpy as np
from typing import List
import pandas as pd


class ReadDatasets:
    @staticmethod
    def read_h5(filename: str) -> List[np.ndarray]:
        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]

            data = list(f[a_group_key])

            return data

    @staticmethod
    def read_csv_uci(filename: str) -> (np.ndarray, np.ndarray):
        dataset = pd.read_csv(filename)

        x_dataset = dataset.drop(columns=['UDI', 'Product ID', 'Type', 'Tool wear [min]',
                                          'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF',
                                          'RNF'])

        y_dataset = dataset.drop(columns=['UDI', 'Product ID', 'Type', 'Air temperature [K]',
                                          'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                                          'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF',
                                          'RNF'])

        matrix_channels = np.array([np.array((x_dataset[feature].tolist())) for feature in x_dataset.keys()])

        y_data_array = np.array(y_dataset['Machine failure'].tolist())

        return matrix_channels, y_data_array
