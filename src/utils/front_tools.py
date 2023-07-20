import pandas as pd
import json
import numpy as np

def generate_json(property_list, df_true_list, df_pred_list):
    # Read dataframes
    properties = []

    for idx, property in enumerate(property_list):
        
        current_data = []
        prevision_data = []
        df_true_list = np.array(df_true_list)
        df_pred_list = np.array(df_pred_list)

        for df in df_true_list[idx, :, :]:
            timestamp = df[1]   # Get timestamp
            value = df[2]       # Get real value
            current_data.append({'timestamp': timestamp, 'value': value})

        for df in df_pred_list[idx, :, :]:
            timestamp = df[1]   # Get timestamp
            value = df[4]       # Get prediction yhat value
            prevision_data.append({'timestamp': timestamp, 'value': value})
        
        # properties.append()
        # print('current_data')
        # print(current_data)
        # print('prevision_data')
        # print(prevision_data)

    # Define the minimum JSON for a notification
    data = {
        "equipment_id": "648a15197e30d0e3725d9a6b",
        "origin_field": "predictive",
        "properties": [
            {
                "property": "Temperature.InletTemperature",
                "value": 100,
                "current_data": current_data,
                "prevision_data": prevision_data
            },
            {
                "property": "Temperature.OutletTemperature",
                "value": 10,
                "current_data": current_data,
                "prevision_data": prevision_data
            },
            {
                "property": "Pressure.InletPressure",
                "value": 1,
                "current_data": current_data,
                "prevision_data": prevision_data
            }
        ]
    }

    # Convert the JSON data to a string
    json_data = json.dumps(data)

    return json_data

if __name__ == '__main__':

    # Test json generation script

    # Load 
    property_list = ['Temperature.InletTemperature', 'Temperature.OutletTemperature', 'Pressure.InletPressure']
    df_true = pd.read_csv('/home/gustavo/Documents/WileC/Predictive_Maintenance_free_dataset/src/utils/df_train.csv')
    df_pred = pd.read_csv('/home/gustavo/Documents/WileC/Predictive_Maintenance_free_dataset/src/utils/df_pred.csv')

    df_pred_list = [df_pred, df_pred, df_pred]
    df_true_list = [df_true, df_true, df_true]

    json_data = generate_json(property_list=property_list, df_true_list=df_true_list, df_pred_list=df_pred_list)
    print(json_data)
    # json.save()
    with open('json_data.json', 'w') as f:
        json.dump(json_data, f)

    # Read JSON file
    # with open('json_data.json', 'r') as data_file:
    #     data_loaded = json.load(data_file)

    # print(json_data == data_loaded)
    # print(data_loaded)
