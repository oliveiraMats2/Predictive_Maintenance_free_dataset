from trainer_auto_regression import VectorAutoRegressionModel
import pandas as pd
import numpy as np

FILE_TMP = "tmp.csv"

if __name__ == '__main__':
    df_inference_tmp = pd.read_csv(FILE_TMP)

    # Filter features.
    df_inference_tmp = df_inference_tmp[['InletPressure',
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
                                         'OAVelocity_z']]

    vector_auto_regression_model = VectorAutoRegressionModel()

    results = vector_auto_regression_model.load_model("vector_auto_regressive.pickle")

    vector_auto_regression_model.result = results

    # df_train, window_input, steps
    inference_for_microservice = vector_auto_regression_model.inference_for_microservice(df_inference_tmp,
                                                                                         window_input=50,
                                                                                         steps=200)

    print()
