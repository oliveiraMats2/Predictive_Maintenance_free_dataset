import pandas as pd
import matplotlib.pyplot as plt
from utils.read_dataset import ReadDatasets
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR

def smape_loss(y_pred, target):
    # y_pred = y_pred.squeeze(2)
    loss = 2 * abs((y_pred - target)) / (abs(y_pred) + abs(target) + 1e-8)
    return loss.mean()*100

def mean_absolute_error(slice_data_true, slice_data_pred):
    abs_data = abs(slice_data_true - slice_data_pred)
    return abs_data.mean()
# Mean absolut error

def mean_square_error(slice_data_true, slice_data_pred):
    sum_square = 0
    for i, elem in enumerate(slice_data_true):
        sum_square += (slice_data_true[i] - slice_data_pred[i]) ** 2

    sum_square = sum_square / (i + 1)

    return sum_square
def define_df(vector_series):
    return pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),#)
                         "ds": np.arange(len(vector_series))})

def avaliable_vector_auto_regressive_model(truth, prediction):

    mae = mean_absolute_error(truth, prediction)
    smape = smape_loss(prediction, truth)
    mse = mean_square_error(truth, prediction)

    print(f"Mean absolut error: {mae}")
    print(f"smape_loss: {smape}")
    print(f"mean absolute error: {mse}")

def vector_autoRegressive_model():
    absolut_path = "../../../Datasets/sintetic_dataset"

    vector_series = ReadDatasets.read_h5(f"{absolut_path}/train_compressor_data.h5")

    df_train = define_df(vector_series)

    vector_series = ReadDatasets.read_h5(f"{absolut_path}/test_compressor_data.h5")

    df_test = define_df(vector_series)

    split_test = 5000
    split_test_end = 16000

    # df = df.drop(labels=["Time"], axis=1)

    # 'temperature', 'frequency'
    temp_series = df_train["temp_series"].tolist()

    fs = len(temp_series)

    # filters_avoid_low_freq = FiltersAvoidLowFreq(fs, value_cutoff_freq=0.05)

    for i in range(5):
        model = VAR(df_train)
        results = model.fit(i)
        print('Order =', i)
        print('AIC: ', results.aic)
        print('BIC: ', results.bic)
        print()

    window = 50
    result = model.fit(window)

    lagged_Values = df_train.values[-window:]
    pred = result.forecast(y=lagged_Values, steps=split_test_end - split_test)

    df_forecast = pd.DataFrame(data=pred, columns=["temp_series_forecast", 'ds_forecast'])

    df_test.index = pd.to_datetime(df_test.index)

    feature = 'temp_series'

    print((df_test[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(df_test["temp_series"].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.title(f"{feature}")
    plt.xlim(400, 600)
    plt.show()

    plt.savefig(f'{feature}.png')

    ground_truth = np.array(df_test["temp_series"].tolist()[400:600])
    pred = np.array(df_forecast[f'{feature}_forecast'].tolist()[400:600])

    return ground_truth, pred


if __name__ == '__main__':
    ground_truth, pred = vector_autoRegressive_model()
    avaliable_vector_auto_regressive_model(ground_truth, pred)
