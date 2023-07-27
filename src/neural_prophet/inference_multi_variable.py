from utils_neural_prophet import AdjustDataFrameForTrain
from train_neural_prophet import TrainNeuralProphet
import pandas as pd
from datetime import datetime
from utils.utils import read_yaml
import argparse
from save_fig_forecast import SaveFigForecast
from generate_timestamp import GenerateTimestamp
import opcua_tools as op, requests
from front_tools import generate_json_future_anomaly
import json


def transform_result_df_prevision(vet_future):
    return pd.DataFrame({"ds": ds, "yhat": vet_future})


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
    return adjust_dataframe_for_train.df_, ds_test, y_hat


features = ["InletPressure"]
# , "InverterSpeed", "OAVelocity_x", "OAVelocity_y", "OAVelocity_z", "OutletPressure",
#         "OutletTemperature", "phaseA_current", "phaseB_current", "phaseC_current", "phaseA_voltage",
#         "phaseB_voltage", "phaseC_voltage"]

result = {}

for idx, feature in enumerate(features):
    df, ds, result[feature] = mono_variable_execute(feature)

    df_truth, detection_timestamp = df, ds

    dict_details_json = {
        "name_model": "neural_prophet",
        "feature_name": feature,
        "detect_time": "future",
        "anomaly_type": "severe",
        "detection_timestamp": detection_timestamp,
        "detection": 1,
        "df_current": df_truth,
        "df_prevision": transform_result_df_prevision(result[feature])

    }

    json_data_future = generate_json_future_anomaly(**dict_details_json)

    with open('json_data_future.json', 'w') as f:
        json.dump(json_data_future, f)

    with open("json_data_future.json") as r:
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

