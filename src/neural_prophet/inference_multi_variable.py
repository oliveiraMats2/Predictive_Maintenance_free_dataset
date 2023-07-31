from utils_neural_prophet import AdjustDataFrameForTrain
from train_neural_prophet import TrainNeuralProphet
import pandas as pd
from datetime import datetime
from utils.utils import read_yaml
import argparse
from generate_timestamp import GenerateTimestamp
import opcua_tools as op, requests
from front_tools import generate_json_future_anomaly
import tqdm
import json
from functools import wraps
import time
import numpy as np


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{total_time:.4f} seconds')
        return result

    return timeit_wrapper


def transform_result_df_prevision(ds, vet_future):
    return pd.DataFrame({"ds": ds, "yhat": vet_future})


@timeit
def mono_variable_execute(model, feature, **configs):
    start_date = datetime(2023, 1, 1, 0, 0, 0)
    end_date = datetime(2023, 7, 24, 23, 59, 59)

    machine = "compressor"

    # df = op.get_historized_values(machine, feature, start_date, end_date)
    #df de teste rapido
    configs["select_feature"] = feature
    df = pd.DataFrame({feature:  [12, 13], configs["time"]: [start_date, end_date]})
    # save_fig_forecast = SaveFigForecast()

    adjust_dataframe_for_train = AdjustDataFrameForTrain(df, **configs)

    adjust_dataframe_for_train.eliminate_outliers(**configs["eliminate_outliers"])

    adjust_dataframe_for_train.eliminate_range_values(**configs["eliminate_range_outliers"])

    df_ = adjust_dataframe_for_train.get_data_frame(**configs["drop"])

    df_train, df_test = adjust_dataframe_for_train.dataset_split(df_, split=configs["train_test_split"])

    df_ = pd.concat([df_train, df_test], ignore_index=True)

    m = model.neural_prophet

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
    return adjust_dataframe_for_train.df_, ds_test, y_hat


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="neuralProphet main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    features = ["InletPressure"
                , "OAVelocity_x"]

    dict_multi_variate_models = {}
    result_multi_variate_models = {}

    configs = read_yaml(args.config_file)

    for idx, feature in enumerate(features):
        dict_multi_variate_models[feature] = TrainNeuralProphet(**configs["parameters_model"])

        dict_multi_variate_models[feature].load(f'src/neural_prophet/weighted_history/{configs["name"]}_{feature}.np')
        # dict_multi_variate_models[feature].load(f'weighted_history/{configs["name"]}_{feature}.np')

    with tqdm.trange(len(features), desc='features') as progress_bar:
        for idx, feature in zip(progress_bar, features):
            result_multi_variate_models[feature] = mono_variable_execute(dict_multi_variate_models[feature], feature, **configs)
            progress_bar.set_postfix(
                desc=f"Prevision - [{feature}]"
            )

    for idx, feature in enumerate(features):
        df, ds, result = result_multi_variate_models[feature]

        df_truth, detection_timestamp = df, ds

        dict_details_json = {
            "name_model": "neural_prophet",
            "feature_name": feature,
            "detect_time": "future",
            "anomaly_type": "severe",
            "detection_timestamp_list": detection_timestamp,
            "detection_value_list": np.zeros(len(detection_timestamp)),
            "df_current": df_truth,
            "df_prevision": transform_result_df_prevision(ds, result)

        }

        json_data_future = generate_json_future_anomaly(**dict_details_json)

        with open(f'json_data_future_{feature}.json', 'w') as f:
            json.dump(json_data_future, f)

        with open(f"json_data_future_{feature}.json") as r:
            json_data = json.load(r)

        # Define the IP address and port of the server
        ip_address = "172.31.111.103"
        port = 447

        # Define the URL to send the POST request to
        url = f"http://{ip_address}:{port}/api/predictive-event"

        # Send the JSON data to the server
        response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

        # Check the response status
        if response.status_code == 201:
            print("JSON data sent successfully!")
        else:
            print("Error sending JSON data:", response.text)
