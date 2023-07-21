from trainer_auto_regression import VectorAutoRegressionModel
import pandas as pd
import numpy as np


def normalize_z(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


if __name__ == '__main__':
    dicio_resample = {"resample_1min":
                          ["../../Datasets/dataset_TPV/base_pump_23042023_A_resampled_1min",
                           "base_pump_23042023_A_resampled_1min.csv", "base_pump_23042023_A_resampled_1min"],
                      "resample_10min":
                          ["../../Datasets/dataset_TPV/base_pump_23042023_A_resampled_10min",
                           "base_pump_23042023_A_resampled_10min.csv", "base_pump_23042023_A_resampled_10min"]}

    # key = 'resample_1min'
    key = 'resample_10min'

    df = pd.read_csv(f"{dicio_resample[key][0]}/{dicio_resample[key][1]}")

    # Filter features.
    df = df[['InletPressure',
             'OutletPressure',
             'OutletTemperature',
             'InverterSpeed',
             'phaseA_active',
             'phaseB_active',
             'phaseC_active',
             'phaseA_current',
             'phaseB_current',
             'phaseC_current',
             'Time',
             'OAVelocity_y',
             'OAVelocity_x',
             'OAVelocity_z']]

    vector_auto_regression_model = VectorAutoRegressionModel()

    df_train, df_test = vector_auto_regression_model.train_test_split_hold_out(df)

    vector_auto_regression_model.fit(df, order=9)

    # df_train, df_test, window, steps, first_limiar, sub_path
    ground_truth, pred = vector_auto_regression_model.predict(df_train,
                                                              df_test,
                                                              window_input=50,
                                                              steps=200,
                                                              first_limiar=100,
                                                              sub_path=dicio_resample[key][2])

    # df_train, window_input, steps
    inference_for_microservice = vector_auto_regression_model.inference_for_microservice(df_train,
                                                                                         window_input=50,
                                                                                         steps=200)

    mean_abs, smape_loss, mean_square_error = vector_auto_regression_model.generate_metrics(ground_truth,
                                                                                            pred)

    df = pd.DataFrame([mean_abs, smape_loss, mean_square_error], index=["mean_abs",
                                                                        "smape_loss",
                                                                        "mean_square_error"]).T

    df.index.name = "variables"

    df.to_csv(f"result_multi_sensor_{dicio_resample[key][1]}")
