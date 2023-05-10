import pandas as pd
import matplotlib.pyplot as plt
from utils.read_dataset import ReadDatasets
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from figures_save import  SaveFigures

def define_df(vector_series):
    return pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),  # )
                         "ds": np.arange(len(vector_series))})


def vector_autoRegressive_model_real(data_,
                                     window=3000,
                                     steps=200,
                                     order=50,
                                     calcs_range_model=10,
                                     first_limiar=2000,
                                     sub_path="payloadITE"):
    df = pd.read_csv(data_)

    print(f"samples: {df.shape[0]}, begin: {df.Time[0]}, end: {df.Time[df.shape[0] - 1]}")
    print(f"samples test: {df.shape[0]//4} | samples train: {3*(df.shape[0]//4)}")
    raise("error")

    df = df.drop('Time', axis=1)

    just_temperature = df# [["temperature", "phaseA_voltage"]]

    one_quarter_base = len(just_temperature) // 4
    inv_one_quarter_base = len(just_temperature) - len(just_temperature) // 4

    just_temperature_train = just_temperature[:inv_one_quarter_base]
    just_temperature_test = just_temperature[inv_one_quarter_base:]

    for i in range(calcs_range_model):
        model = VAR(just_temperature_train)
        results = model.fit(i)
        print('Order =', i)
        print('AIC: ', results.aic)
        print('BIC: ', results.bic)
        print()

    result = model.fit(order)

    lagged_Values = just_temperature_train.values[-window:]
    pred = result.forecast(y=lagged_Values, steps=steps)

    #feature = 'phaseA_voltage'

    col_predicted = [f"{x}_forecast" for x in just_temperature_train.keys()]

    df_forecast = pd.DataFrame(data=pred, columns=col_predicted)

    just_temperature_test.index = pd.to_datetime(just_temperature_test.index)

    # print((just_temperature_test[feature].shape,
    #        df_forecast[f'{feature}_forecast'].shape))

    SaveFigures.save(df_test=just_temperature_test,
                     df_pred=df_forecast,
                     lim_end=steps,
                     first_limiar=first_limiar,
                     sub_path=sub_path)

    # ground_truth = np.array(just_temperature_test[feature].tolist())
    # pred = np.array(df_forecast[f'{feature}_forecast'].tolist())

    ground_truth, pred = just_temperature_test, df_forecast

    pred_slice = pred.shape[0]
    ground_truth = ground_truth[:pred_slice]

    return ground_truth, pred


def vector_autoRegressive_model_sintetic(abs_path_train="../../../Datasets/sintetic_dataset/train_compressor_data.h5",
                                         abs_path_test="../../../Datasets/sintetic_dataset/test_compressor_data.h5"):
    vector_series = ReadDatasets.read_h5(abs_path_train)

    df_train = define_df(vector_series)

    vector_series = ReadDatasets.read_h5(abs_path_test)

    df_test = define_df(vector_series)

    split_test = 5000
    split_test_end = 16000

    steps = split_test_end - split_test

    steps = 26000

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

    lagged_Values = df_test.values[-window:]
    pred = result.forecast(y=lagged_Values, steps=steps)

    df_forecast = pd.DataFrame(data=pred, columns=["temp_series_forecast", 'ds_forecast'])

    df_test.index = pd.to_datetime(df_test.index)

    feature = 'temp_series'

    print((df_test[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))


    ground_truth = np.array(df_test["temp_series"].tolist())
    pred = np.array(df_forecast[f'{feature}_forecast'].tolist())

    return ground_truth, pred
