list_keys = ['InletPressure',
         'OutletPressure',
         'OutletTemperature',
         'InverterSpeed',
         'phaseA_active',
         'phaseB_active',
         'phaseC_active',
         'phaseA_current',
         'phaseB_current',
         'phaseC_current',
         'OAVelocity_y',
         'OAVelocity_x',
         'OAVelocity_z']

def split_on_dict_gluon(tss, forecasts):
    dict_forecast = {}
    dict_ground_truth = {}
    for i, elements in enumerate(forecasts):
        dict_forecast[list_keys[i]] = forecasts[i].samples.mean(axis=0)
        dict_ground_truth[list_keys[i]] = tss[i][0].tolist() # 0 is key dataframe tss


    return dict_ground_truth, dict_forecast


def length_lower(dataframe1, dataframe2):
    tam_1 = dataframe1.shape[0]
    tam_2 = dataframe2.shape[0]

    if tam_1 < tam_2:
        return tam_1
    else:
        return tam_2

def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())