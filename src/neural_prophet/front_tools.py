import pandas as pd
import json
import numpy as np
import os

def feature_name_parser(old_feature_name):
    """
    Returns the parsed feature name according to Front-end pattern.

            Parameters:
                    feature_name (string): Name of the old feature name

            Returns:
                    new_feature_name (string): parsed feature name according to feature name mapping
    """

    feature_name_map = {
        "InletPressure": "Pressure.InletPressure",
        "OutletPressure": "Pressure.OutletPressure",
        "phaseA_voltage": "PhaseA.Voltage",
        "phaseB_voltage": "PhaseB.Voltage",
        "phaseC_voltage": "PhaseC.Voltage",
        "phaseA_current": "PhaseA.Current",
        "phaseB_current": "PhaseB.Current",
        "phaseC_current": "PhaseC.Current",
        "OAVelocity_x": "HorizontalVibration.XVibrationAccelerationVelocityOa",
        "OAVelocity_y": "VerticalVibration.YVibrationAccelerationVelocityOa",
        "OAVelocity_z": "AxialVibration.ZVibrationAccelerationVelocityOa",
        "temperature": "Temperature.OutletTemperature",
    }

    new_feature_name = ""

    if old_feature_name in feature_name_map:
        new_feature_name = feature_name_map[old_feature_name]
    else:
        new_feature_name = old_feature_name

    return new_feature_name

def generate_json_current_anomaly(
    name_model,
    feature_name,
    detect_time,
    anomaly_type,
    detection_timestamp_list,
    detection_value_list,
    df_current,
):
    """
    Returns the formatted JSON string.

            Parameters:
                    name_model (string): name of the ML model
                    feature_name (string): name of the feature analysed
                    detect_time (string): detection time <current|future>
                    anomaly_type (string): anomaly type <none|light|medium|severe>
                    detection_timestamp_list (array [timestamp]): array containing the detection timestamp
                    detection_value_list (array [int]): array containing the detection values <0: normal|1: anomaly>
                    df_current (Dataframe): dataframe containing the input data

            Returns:
                    json_data (string): formatted JSON string
    """
    # Constants
    OUTPUT_FOLDER_NAME = 'output/'
    SAVE_JSON_FILE = True

    # Process dataframes to pattern
    current_data = []
    detection_data = []
    df_current = np.array(df_current)

    # Formats current_data to pattern
    for df in df_current[:, :]:
        timestamp = df[1]  # Get timestamp
        value = df[2]  # Get real value
        current_data.append({"timestamp": timestamp, "value": value})

    # Formats detection to pattern
    if len(detection_value_list) == len(detection_timestamp_list):
        for index, _ in enumerate(detection_timestamp_list):
            timestamp = detection_timestamp_list[index]  # Get timestamp
            value = detection_value_list[index]  # Get prediction yhat value
            detection_data.append({"timestamp": timestamp, "detection": value})
    else:
        detection_data.append(
            {
                "timestamp": "timestamp and detection value array size mismatch",
                "detection": "timestamp and detection value array size mismatch",
            }
        )

    # Define the JSON header and properties
    data = {
        "equipment_id": "648a15197e30d0e3725d9a6b",
        "origin_field": "predictive",
        "evaluation_criticality": True,
        "properties": [
            {
                "property": feature_name_parser(feature_name),
                # "value": 8,
                "current_data": current_data,
                "prevision_description": {
                    "name_model": name_model,
                    "feature_name": feature_name_parser(feature_name),
                    "detect_time": detect_time,
                    "anomaly_type": anomaly_type,
                    "detection": detection_data,
                },
            },
        ],
    }

    # Convert the JSON data to a string
    json_data = json.dumps(eval(data))

    # Save JSON file
    if SAVE_JSON_FILE:
        # Check if folder exists, then create it if needed
        if not(os.path.exists(os.path.join(OUTPUT_FOLDER_NAME))):
            os.mkdir(os.path.join(OUTPUT_FOLDER_NAME))

        first_timestamp = str(df_current[:, :][0][0])
        first_timestamp = "".join(ch for ch in first_timestamp if ch.isalnum())
        filename = f"{feature_name}_{first_timestamp}.json"
        filename = os.path.join(OUTPUT_FOLDER_NAME, filename)

        with open(filename, "w") as f:
            json.dump(json_data, f)

    return json_data


def generate_json_future_anomaly(
    name_model,
    feature_name,
    detect_time,
    anomaly_type,
    detection_timestamp_list,
    detection_value_list,
    df_current,
    df_prevision,
):
    """
    Returns the formatted JSON string.

            Parameters:
                    name_model (string): name of the ML model
                    feature_name (string): name of the feature analysed
                    detect_time (string): detection time <current|future>
                    anomaly_type (string): anomaly type <none|light|medium|severe>
                    detection_timestamp_list (array [timestamp]): array containing the detection timestamp
                    detection_value_list (array [int]): array containing the detection values <0: normal|1: anomaly>
                    df_current (Dataframe): dataframe containing the input data
                    df_prevision (Dataframe): dataframe containing the prevision data

            Returns:
                    json_data (string): formatted JSON string
    """
    # Constants
    OUTPUT_FOLDER_NAME = 'output/'
    SAVE_JSON_FILE = True

    # Process dataframes to pattern
    current_data = []
    prevision_data = []
    detection_data = []
    df_current = np.array(df_current)
    df_prevision = np.array(df_prevision)

    # Formats current_data to pattern
    for df in df_current[:, :]:
        timestamp = str(df[0])  # Get timestamp
        value = df[1]  # Get real value
        current_data.append({"timestamp": timestamp, "value": value})

    # Formats prevision_data to pattern
    for df in df_prevision[:, :]:
        timestamp = str(df[0])  # Get timestamp
        value = df[1]  # Get prediction yhat value
        prevision_data.append({"timestamp": timestamp, "value": value})

    # Formats detection to pattern
    if len(detection_value_list) == len(detection_timestamp_list):
        for index, _ in enumerate(detection_timestamp_list):
            timestamp = str(detection_timestamp_list[index])  # Get timestamp
            value = detection_value_list[index]  # Get prediction yhat value
            detection_data.append({"timestamp": timestamp, "detection": value})
    else:
        detection_data.append(
            {
                "timestamp": "timestamp and detection value array size mismatch",
                "detection": "timestamp and detection value array size mismatch",
            }
        )

    feature_new_name = feature_name_parser(feature_name)
    # Define the JSON header and properties
    data = {
        "equipment_id": "64909fc47e30d0e3725d9a9a",
        "origin_field": "predictive",
        "evaluation_criticality": True,
        "properties": [
            {
                "property": feature_new_name,
                "current_data": current_data,
                "prevision_data": prevision_data,
                "prevision_description": {
                    "name_model": name_model,
                    "feature_name": feature_new_name,
                    "detect_time": detect_time,
                    "anomaly_type": anomaly_type,
                    "detection": detection_data
                },
            },
        ],
    }

    # Convert the JSON data to a string
    # json_data = json.dumps(eval(data))
    json_data = json.dumps(data)
    # json_data = json.dumps(eval(json.loads(data)))
    # json_data = json_data.replace('"evaluation_criticality": true,', '"evaluation_criticality": True,')


    # Save JSON file
    if SAVE_JSON_FILE:
        # Check if folder exists, then create it if needed
        if not(os.path.exists(os.path.join(OUTPUT_FOLDER_NAME))):
            os.mkdir(os.path.join(OUTPUT_FOLDER_NAME))

        first_timestamp = str(df_prevision[:, :][0][0])
        first_timestamp = "".join(ch for ch in first_timestamp if ch.isalnum())
        filename = f"{feature_name}_{first_timestamp}.json"
        filename = os.path.join(OUTPUT_FOLDER_NAME, filename)

        with open(filename, "w") as f:
            json.dump(json_data, f)

    return json_data


if __name__ == "__main__":
    # Test json generation script

    # Load
    # name_model, feature_name, detect_time, anomaly_type, detection_timestamp, detection, df_current, df_prevision
    name_model = "modelo1"
    feature_name = "Temperature.InletTemperature"
    df_true = pd.read_csv(
        "/media/antonio/AllData/Workspace/git/general/ufam/Predictive_Maintenance_free_dataset/src/utils/df_train.csv"
    )
    df_pred = pd.read_csv(
        "/media/antonio/AllData/Workspace/git/general/ufam/Predictive_Maintenance_free_dataset/src/utils/df_pred.csv"
    )

    # Test generate json function
    detection_timestamps = df_true["ds"]
    detection_values = np.zeros(shape=detection_timestamps.size)
    json_data_current = generate_json_current_anomaly(
        name_model,
        feature_name,
        "current",
        "severe",
        detection_timestamps,
        detection_values,
        df_true,
    )

    detection_timestamps = df_pred["ds"]
    detection_values = np.zeros(shape=detection_timestamps.size)
    json_data_future = generate_json_future_anomaly(
        name_model,
        feature_name,
        "future",
        "severe",
        detection_timestamps,
        detection_values,
        df_true,
        df_pred,
    )

    # Save JSON file
    with open("json_data_current.json", "w") as f:
        json.dump(json_data_current, f)

    with open("json_data_future.json", "w") as f:
        json.dump(json_data_future, f)

    # Read JSON file
    # with open('json_data.json', 'r') as data_file:
    #     data_loaded = json.load(data_file)

    # print(json_data == data_loaded)
    # print(json_data)
