import pandas as pd
import json
import numpy as np


def generate_json_current_anomaly(name_model, feature_name, detect_time, anomaly_type, detection_timestamp_list,
                                  detection_value_list, df_current):
    '''
    Returns the formatted JSON string.

            Parameters:
                    name_model (string): Name of the ML model
                    feature_name (string): Name of the feature analysed
                    detect_time (string): Detection time <current|future>
                    anomaly_type (string): Anomaly type <none|light|medium|severe>
                    detection_timestamp_list (array [timestamp]): Array containing the detection timestamp
                    detection_value_list (array [int]): Array containing the detection values <0: normal|1: anomaly>
                    df_current (Dataframe): Dataframe containing the input data

            Returns:
                    json_data (string): formatted JSON string
    '''

    # Process dataframes to pattern
    current_data = []
    detection_data = []
    df_current = np.array(df_current)

    # Formats current_data to pattern
    for df in df_current[:, :]:
        timestamp = df[1]  # Get timestamp
        value = df[2]  # Get real value
        current_data.append({'timestamp': timestamp, 'value': value})

    # Formats Detection to pattern
    if len(detection_value_list) == len(detection_value_list):
        for index, _ in enumerate(detection_timestamp_list):
            timestamp = detection_timestamp_list[index]  # Get timestamp
            value = detection_value_list[index]  # Get prediction yhat value
            detection_data.append({'Timestamp': timestamp, 'Detection': value})
    else:
        detection_data.append({'Timestamp': 'Timestamp and Detection value array size mismatch',
                               'Detection': 'Timestamp and Detection value array size mismatch'})

    # Define the JSON header and properties
    data = {
        "equipment_id": "648a15197e30d0e3725d9a6b",
        "origin_field": "predictive",
        "evaluation_criticality": True,
        "properties": [
            {
                "property": feature_name,
                "value": 8,
                "current_data": current_data,
                "prevision_description": {
                    "Name_model": name_model,
                    "Feature_name": feature_name,
                    "Detect_time": detect_time,
                    "Anomaly_type": anomaly_type,
                    "Detection": detection_data
                },
            },
        ],
    }

    # Convert the JSON data to a string
    json_data = json.dumps(data)

    return json_data


def generate_json_future_anomaly(name_model, feature_name, detect_time, anomaly_type, detection_timestamp_list,
                                 detection_value_list, df_current, df_prevision):
    '''
    Returns the formatted JSON string.

            Parameters:
                    name_model (string): Name of the ML model
                    feature_name (string): Name of the feature analysed
                    detect_time (string): Detection time <current|future>
                    anomaly_type (string): Anomaly type <none|light|medium|severe>
                    detection_timestamp_list (array [timestamp]): Array containing the detection timestamp
                    detection_value_list (array [int]): Array containing the detection values <0: normal|1: anomaly>
                    df_current (Dataframe): Dataframe containing the input data
                    df_prevision (Dataframe): Dataframe containing the prevision data

            Returns:
                    json_data (string): formatted JSON string
    '''

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
        current_data.append({'timestamp': timestamp, 'value': value})

    # Formats prevision_data to pattern
    for df in df_prevision[:, :]:
        timestamp = str(df[0])  # Get timestamp
        value = df[1]  # Get prediction yhat value
        prevision_data.append({'timestamp': timestamp, 'value': value})

    # Formats Detection to pattern
    if len(detection_value_list) == len(detection_value_list):
        for index, _ in enumerate(detection_timestamp_list):
            timestamp = str(detection_timestamp_list[index])  # Get timestamp
            value = detection_value_list[index]  # Get prediction yhat value
            detection_data.append({'Timestamp': timestamp, 'Detection': value})
    else:
        detection_data.append({'Timestamp': 'Timestamp and Detection value array size mismatch',
                               'Detection': 'Timestamp and Detection value array size mismatch'})

    # Define the JSON header and properties
    data = {
        "equipment_id": "64909fc47e30d0e3725d9a9a",
        "origin_field": "predictive",
        "evaluation_criticality": True,
        "properties": [
            {
                "property": f"Temperature.InletTemperature",
                "value": 8,
                "current_data": current_data,
                "prevision_data": prevision_data,
                # "prevision_description": {
                #     "Name_model": name_model,
                #     "Feature_name": feature_name,
                #     "Detect_time": detect_time,
                #     "Anomaly_type": anomaly_type,
                #     "Detection": detection_data
                # },
            },
        ],
    }

    # Convert the JSON data to a string
    json_data = json.dumps(data)

    return json_data


if __name__ == '__main__':
    # Test json generation script

    # Load
    # name_model, feature_name, detect_time, anomaly_type, detection_timestamp, detection, df_current, df_prevision
    name_model = 'modelo1'
    feature_name = 'Temperature.InletTemperature'
    df_true = pd.read_csv(
        '/media/antonio/AllData/Workspace/git/general/ufam/Predictive_Maintenance_free_dataset/src/utils/df_train.csv')
    df_pred = pd.read_csv(
        '/media/antonio/AllData/Workspace/git/general/ufam/Predictive_Maintenance_free_dataset/src/utils/df_pred.csv')

    # Test generate json function
    detection_timestamps = df_true['ds']
    detection_values = np.zeros(shape=detection_timestamps.size)
    json_data_current = generate_json_current_anomaly(name_model, feature_name, 'current', 'severe',
                                                      detection_timestamps, detection_values, df_true)

    detection_timestamps = df_pred['ds']
    detection_values = np.zeros(shape=detection_timestamps.size)
    json_data_future = generate_json_future_anomaly(name_model, feature_name, 'future', 'severe', detection_timestamps,
                                                    detection_values, df_true, df_pred)

    # Save JSON file
    with open('json_data_current.json', 'w') as f:
        json.dump(json_data_current, f)

    with open('json_data_future.json', 'w') as f:
        json.dump(json_data_future, f)

    # Read JSON file
    # with open('json_data.json', 'r') as data_file:
    #     data_loaded = json.load(data_file)

    # print(json_data == data_loaded)
    # print(json_data)
