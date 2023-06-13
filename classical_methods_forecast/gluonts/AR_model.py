
from classical_methods_forecast.statistics_methods.metrics import avaliable_vector_auto_regressive_model
from utils_ar import split_on_dict_gluon, length_lower, min_max_normalize
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_1 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/x.csv')
df_2 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/y.csv')
df_3 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/z.csv')

# Combina as colunas relevantes dos DataFrames
df = pd.DataFrame()
df["OAVelocity_x"] = df_1["OAVelocity"].astype('float64')
df["OAVelocity_y"] = df_2["OAVelocity"].astype('float64')
df["OAVelocity_z"] = df_3["OAVelocity"].astype('float64')
df["ds"] = pd.to_datetime(df_1["Time"])

# Remove as datas duplicadas
df = df.drop_duplicates(subset=["ds"])

# Ordena o DataFrame pelo índice de datas
df = df.sort_values('ds')

# Redefine o índice do DataFrame
df = df.reset_index(drop=True)

df['timestamp'] = pd.to_datetime(df['ds'])
df.set_index('timestamp', inplace=True)
df = df[["OAVelocity_x", "OAVelocity_y", "OAVelocity_z"]]

# df = pd.read_csv("/mnt/arquivos_linux/download/base_23042023_A_multisensors_first_11365_samples.csv")

dicio_resample = {"resample_1min":
                          ["../../Datasets/dataset_TPV/base_pump_23042023_A_resampled_1min",
                           "base_pump_23042023_A_resampled_1min.csv", "base_pump_23042023_A_resampled_1min"],
                      "resample_10min":
                          ["../../Datasets/dataset_TPV/base_pump_23042023_A_resampled_10min",
                           "base_pump_23042023_A_resampled_10min.csv", "base_pump_23042023_A_resampled_10min"]}

# key = 'resample_1min'
key = 'resample_10min'

df = pd.read_csv(f"{dicio_resample[key][0]}/{dicio_resample[key][1]}")

df = df[['InletPressure',
         'OutletPressure',
         'OutletTemperature',
         'InverterSpeed',
         'phaseA_active',
         'phaseB_active',
         'phaseC_active',
         'phaseA_current',
         'phaseB_current',
         'phaseC_current',
         'Time',
         'OAVelocity_y',
         'OAVelocity_x',
         'OAVelocity_z']]

df['timestamp'] = pd.to_datetime(df['Time'])
df.set_index('timestamp', inplace=True)

df = df[['InletPressure',
         'OutletPressure',
         'OutletTemperature',
         'InverterSpeed',
         'phaseA_active',
         'phaseB_active',
         'phaseC_active',
         'phaseA_current',
         'phaseB_current',
         'phaseC_current',
         'OAVelocity_y',
         'OAVelocity_x',
         'OAVelocity_z']].apply(min_max_normalize)

# Imprime o DataFrame atualizado
print("DataFrame com datas organizadas:")
print(df.shape)

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

slice_data = int(3*(df.shape[0]/4))

train_data = df[:slice_data]
test_data = df[slice_data:]

def to_deepar_format(dataframe):
    freq = pd.infer_freq(dataframe.index)
    start_index = dataframe.index.min()
    data = [{
                FieldName.START:  start_index,
                FieldName.TARGET:  dataframe[c].values,
            }
            for c in dataframe.columns]
    print(data[0])
    return ListDataset(data, freq='10T')

train_data_lds = to_deepar_format(train_data)
test_data_lds = to_deepar_format(test_data)


from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

prediction_length = 200
context_length = 7
num_cells = 32
num_layers = 2
epochs= 5
freq="D"

estimator = DeepAREstimator(freq=freq,
                            context_length=context_length,
                            prediction_length=prediction_length,
                            num_layers=num_layers,
                            num_cells=num_cells,
                            cardinality=[1],
                            trainer=Trainer(epochs=epochs))


predictor = estimator.train(train_data_lds)

from gluonts.evaluation.backtest import make_evaluation_predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data_lds,
    predictor=predictor,
)
tss = list(ts_it)
forecasts = list(forecast_it)

import matplotlib.pyplot as plt
def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = prediction_length
    prediction_intervals = (80.0, 95.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


from gluonts.evaluation import Evaluator
dict_ground_truth, dict_forecast = split_on_dict_gluon(tss, forecasts)

df_gt = pd.DataFrame(dict_ground_truth)
df_forecast = pd.DataFrame(dict_forecast)

slice_data = length_lower(df_gt, df_forecast)

df_gt = df_gt[:slice_data]

dict_mae, dict_smape, dict_mse = avaliable_vector_auto_regressive_model(df_gt,
                                                                        df_forecast,
                                                                        type_model="multiple")

print(f"dict mae {dict_mse}")
print(f"dict mae {dict_mae}")
print(f"dict smape {dict_smape}")

# evaluator = Evaluator()
# agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data_lds))
#
# import json
# print(json.dumps(agg_metrics, indent=4))
# item_metrics