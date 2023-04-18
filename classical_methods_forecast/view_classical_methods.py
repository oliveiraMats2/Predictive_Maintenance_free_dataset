from filters.filter_avoid_amplitude import FiltersAvoidLowFreq
from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def view_filter_low_pass():
    # plot the original and filtered signals

    payload_hex = "../Datasets/dataset_TPV_sensors/hex/payloadHex.csv"

    df = pd.read_csv(payload_hex)

    feature = "InverterSpeed"

    x = df[feature].tolist()

    fs = len(x)  # sampling frequency
    filters_avoid_low_freq = FiltersAvoidLowFreq(fs, value_cutoff_freq=0.05)

    y = filters_avoid_low_freq.apply(x)

    t = np.linspace(0, 1, fs, endpoint=False)

    # 'InletPressure',
    # 'OutletPressure',
    # 'OutletTemperature',
    # 'InverterSpeed'

    plt.figure()
    plt.plot(t, x, 'b', alpha=0.5)
    plt.plot(t, y, 'r')
    plt.legend(['Original', 'Filtered'])
    plt.savefig(f'savefig/{feature}.png')
    plt.show()


def vector_autoRegressive_model():
    #payload_hex = "../Datasets/dataset_TPV_sensors/hex/payloadHex.csv"
    payload_ite = "../Datasets/dataset_TPV_sensors/ite/payloadITE.csv"

    #df = pd.read_csv(payload_hex)
    df = pd.read_csv(payload_ite)

    split_test = 5000
    split_test_end = 16000

    # df = df.drop(labels=["Time"], axis=1)

    #'temperature', 'frequency'
    inlet_pressure = df["temperature"].tolist()

    fs = len(inlet_pressure)

    # filters_avoid_low_freq = FiltersAvoidLowFreq(fs, value_cutoff_freq=0.05)

    df_old = df

    # df = filters_avoid_low_freq.filter_multiple_sensor(df, "InletPressure",
    #                                                    "OutletPressure",
    #                                                    "OutletTemperature",
    #                                                    "InverterSpeed")

    train = df[:split_test]
    test = df[split_test:split_test_end]

    for i in range(5):
        model = VAR(train)
        results = model.fit(i)
        print('Order =', i)
        print('AIC: ', results.aic)
        print('BIC: ', results.bic)
        print()

    window = 100
    result = model.fit(window)
    # print(result.summary())

    lagged_Values = train.values[-window:]
    pred = result.forecast(y=lagged_Values, steps=split_test_end - split_test)

    # idx = pd.date_range('2015-01-01', periods=split_test, freq='MS')
    df_forecast = pd.DataFrame(data=pred, columns=['InletPressure_forecast',
                                                   'OutletPressure_forecast',
                                                   'OutletTemperature_forecast',
                                                   'InverterSpeed_forecast'])

    #test_original = df[:-split_test]
    test.index = pd.to_datetime(test.index)
    # 'InletPressure',
    # 'OutletPressure',
    # 'OutletTemperature',
    # 'InverterSpeed'

    feature = 'InletPressure'

    print((test[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(test[feature].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.title(f"{feature}")
    plt.savefig(f'savefig_VAR/{feature}.png')

    feature = 'OutletPressure'

    print((test[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(test[feature].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.title(f"{feature}")
    plt.savefig(f'savefig_VAR/{feature}.png')

    feature = 'OutletTemperature'

    print((test[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(test[feature].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.title(f"{feature}")
    plt.savefig(f'savefig_VAR/{feature}.png')

    feature = 'InverterSpeed'

    print((test[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(test[feature].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.title(f"{feature}")
    plt.savefig(f'savefig_VAR/{feature}.png')

    plt.show()


if __name__ == '__main__':
    # view_filter_low_pass()
    vector_autoRegressive_model()
