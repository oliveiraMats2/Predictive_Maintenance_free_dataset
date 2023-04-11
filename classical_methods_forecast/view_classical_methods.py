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
    payload_hex = "../Datasets/dataset_TPV_sensors/hex/payloadHex.csv"

    df = pd.read_csv(payload_hex)

    split_test = 45000

    df = df.drop(labels=["Time"], axis=1)

    inlet_pressure = df["InletPressure"].tolist()

    fs = len(inlet_pressure)

    filters_avoid_low_freq = FiltersAvoidLowFreq(fs, value_cutoff_freq=0.05)
    df = filters_avoid_low_freq.filter_multiple_sensor(df, "InletPressure",
                                                       "OutletPressure",
                                                       "OutletTemperature",
                                                       "InverterSpeed")

    train = df[:-split_test]
    test = df[-split_test:]

    for i in range(20):
        model = VAR(train)
        results = model.fit(i)
        print('Order =', i)
        print('AIC: ', results.aic)
        print('BIC: ', results.bic)
        print()

    window = 500
    result = model.fit(window)
    # print(result.summary())

    lagged_Values = train.values[-window:]
    pred = result.forecast(y=lagged_Values, steps=split_test)

    # idx = pd.date_range('2015-01-01', periods=split_test, freq='MS')
    df_forecast = pd.DataFrame(data=pred, columns=['InletPressure_forecast',
                                                   'OutletPressure_forecast',
                                                   'OutletTemperature_forecast',
                                                   'InverterSpeed_forecast'])

    test_original = df[:-split_test]
    test_original.index = pd.to_datetime(test_original.index)
    # 'InletPressure',
    # 'OutletPressure',
    # 'OutletTemperature',
    # 'InverterSpeed'

    feature = 'OutletTemperature'

    print((test_original[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(test_original[feature].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.savefig(f'savefig_VAR/{feature}.png')

    feature = 'OutletPressure'

    print((test_original[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(test_original[feature].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.savefig(f'savefig_VAR/{feature}.png')

    feature = 'OutletTemperature'

    print((test_original[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(test_original[feature].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.savefig(f'savefig_VAR/{feature}.png')

    feature = 'InverterSpeed'

    print((test_original[feature].shape,
           df_forecast[f'{feature}_forecast'].shape))

    plt.figure()
    plt.plot(test_original[feature].tolist())
    plt.plot(df_forecast[f'{feature}_forecast'].tolist())
    plt.savefig(f'savefig_VAR/{feature}.png')

    plt.show()


if __name__ == '__main__':
    view_filter_low_pass()
    # vector_autoRegressive_model()
