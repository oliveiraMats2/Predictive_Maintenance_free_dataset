from utils_neural_prophet import AdjustDataFrameForTrain
from train_neural_prophet import TrainNeuralProphet
import pandas as pd
from datetime import datetime
from utils.utils import read_yaml
import argparse
from save_fig_forecast import SaveFigForecast
from generate_timestamp import GenerateTimestamp
import opcua_tools as op


def mono_variable_execute(feature="PhaseA-voltage"):
    parser = argparse.ArgumentParser(description="neuralProphet main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    start_date = datetime(2023, 1, 1, 0, 0, 0)
    end_date = datetime(2023, 7, 24, 23, 59, 59)

    machine = "compressor"

    df = op.get_historized_values(machine, feature, start_date, end_date)
    # save_fig_forecast = SaveFigForecast()

    configs["select_feature"] = feature

    adjust_dataframe_for_train = AdjustDataFrameForTrain(df, **configs)

    adjust_dataframe_for_train.eliminate_outliers(**configs["eliminate_outliers"])

    adjust_dataframe_for_train.eliminate_range_values(**configs["eliminate_range_outliers"])

    df_ = adjust_dataframe_for_train.get_data_frame(**configs["drop"])

    df_train, df_test = adjust_dataframe_for_train.dataset_split(df_, split=configs["train_test_split"])

    df_ = pd.concat([df_train, df_test], ignore_index=True)

    # ----------------------------------------------------------------
    train_model = TrainNeuralProphet(**configs["parameters_model"])

    metrics = adjust_dataframe_for_train.train_or_test(df_train, train_model, **configs)

    train_model.load(f'weighted_history/{configs["name"]}_{configs["select_feature"]}.np')

    m = train_model.neural_prophet

    df_test["y"] = 0

    df_test = GenerateTimestamp.generate_timestamps_delimiter(start=df["Time"].tolist()[-1],
                                                              end=configs["prediction_for_future"]["end"])

    df_eixo_time = df_test

    future = m.make_future_dataframe(df_eixo_time,
                                     periods=configs["predict_in_the_future"],
                                     n_historic_predictions=len(df_eixo_time))

    forecast = m.predict(df=future)

    y_hat = forecast["yhat1"].tolist()
    ds_test = forecast["ds"]
    # print(y_hat[:10])
    return y_hat


features = ["InletPressure", "InverterSpeed", "OAVelocity_x", "OAVelocity_y", "OAVelocity_z", "OutletPressure",
            "OutletTemperature", "phaseA_current", "phaseB_current", "phaseC_current", "phaseA_voltage",
            "phaseB_voltage", "phaseC_voltage"]

result = {}

for idx, feature in enumerate(features):
    result[feature] = mono_variable_execute(feature)
    print(feature)

