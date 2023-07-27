import pandas as pd
import json
import numpy as np


def generate_json_current_anomaly(name_model, feature_name, detect_time, anomaly_type, detection_timestamp, detection, df_current):
    # Process dataframes to pattern
    current_data = []
    df_current = np.array(df_current)

    for df in df_current[:, :]:
        timestamp = df[1]   # Get timestamp
        value = df[2]       # Get real value
        current_data.append({'timestamp': timestamp, 'value': value})

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
                    "Detection": {
                        "Timestamp": detection_timestamp,
                        "Detection": detection
                    },
                },
            },
        ],
    }

    # Convert the JSON data to a string
    json_data = json.dumps(data)

    return json_data


def generate_json_future_anomaly(name_model, feature_name, detect_time, anomaly_type, detection_timestamp, detection, df_current, df_prevision):
    # Process dataframes to pattern
    current_data = []
    prevision_data = []
    df_current = np.array(df_current)
    df_prevision = np.array(df_prevision)

    for df in df_current[:, :]:
        timestamp = df[1]   # Get timestamp
        value = df[2]       # Get real value
        current_data.append({'timestamp': timestamp, 'value': value})

    for df in df_prevision[:, :]:
        timestamp = df[1]   # Get timestamp
        value = df[4]       # Get prediction yhat value
        prevision_data.append({'timestamp': timestamp, 'value': value})

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
                "prevision_data": prevision_data,
                "prevision_description": {
                    "Name_model": name_model,
                    "Feature_name": feature_name,
                    "Detect_time": detect_time,
                    "Anomaly_type": anomaly_type,
                    "Detection": {
                        "Timestamp": detection_timestamp,
                        "Detection": detection
                    },
                },
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
    df_true = pd.read_csv('/media/antonio/AllData/Workspace/git/general/ufam/Predictive_Maintenance_free_dataset/src/utils/df_train.csv')
    df_pred = pd.read_csv('/media/antonio/AllData/Workspace/git/general/ufam/Predictive_Maintenance_free_dataset/src/utils/df_pred.csv')
    detection_timestamp = df_true['ds'][500]        # Choose timestamp for test
    json_data_current = generate_json_current_anomaly(name_model, feature_name, 'current', 'severe', detection_timestamp, 1, df_true)
    json_data_future = generate_json_future_anomaly(name_model, feature_name, 'future', 'severe', detection_timestamp, 1, df_true, df_pred)

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
