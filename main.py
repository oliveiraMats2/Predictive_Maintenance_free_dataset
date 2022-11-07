import pandas as pd
from models import LSTM

dataset = pd.read_csv("uci_base_machine_learning.csv")

dataset.drop('Air temperature [K]',
             'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
             'Tool wear [min]', 'Machine failure')
# lstm_classifier = LSTM()


#Datasets and DataLoaders