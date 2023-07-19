from utils_neural_prophet import AdjustDataFrameForTrain
from train_neural_prophet import TrainNeuralProphet
import pandas as pd
from utils.utils import read_yaml
import argparse
from save_fig_forecast import SaveFigForecast
from upcua_instants_value import UpcuaInstantValues
from generate_timestamp import GenerateTimestamp
def find_value_lower(a, b):
    if a < b:
        return a
    else:
        return b


def truncate_values(y_truth, yhat):
    truncate_value = find_value_lower(len(y_truth), len(yhat))
    y_truth = y_truth[:truncate_value]
    yhat = yhat[:truncate_value]

    return y_truth, yhat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="neuralProphet main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    upcua_instant_values = UpcuaInstantValues("PhaseA-voltage")

    save_fig_forecast = SaveFigForecast()

    df = upcua_instant_values.actual_dataframe(2)

    feature_ = configs["select_feature"]

    configs["select_feature"] = configs["select_feature_upcua"]

    adjust_dataframe_for_train = AdjustDataFrameForTrain(df, **configs)

    adjust_dataframe_for_train.eliminate_outliers(**configs["eliminate_outliers"])

    adjust_dataframe_for_train.eliminate_range_values(**configs["eliminate_range_outliers"])

    df_ = adjust_dataframe_for_train.get_data_frame(**configs["drop"])

    df_train, df_test = adjust_dataframe_for_train.dataset_split(df_, split=configs["train_test_split"])

    df_ = pd.concat([df_train, df_test], ignore_index=True)

    # ----------------------------------------------------------------

    configs["select_feature"] = feature_

    train_model = TrainNeuralProphet(**configs["parameters_model"])

    metrics = adjust_dataframe_for_train.train_or_test(df_train, train_model, **configs)

    train_model.load(f'weighted_history/{configs["name"]}_{configs["select_feature"]}.np')

    m = train_model.neural_prophet

    df_test["y"] = 0

    df_test = GenerateTimestamp.generate_timestamps_delimiter(start=df["Time"].tolist()[-1],
                                                              end="2023-08-10")

    df_eixo_time = df_test

    future = m.make_future_dataframe(df_eixo_time,
                                     periods=configs["predict_in_the_future"],
                                     n_historic_predictions=len(df_eixo_time))

    forecast = m.predict(df=future)

    y_truth = df_train["y"].tolist()
    y_hat = forecast["yhat1"].tolist()
    ds_test = forecast["ds"]
    ds_train = df_train["ds"]