from train_varma import VARMA, train_test_split_hold_out
import pandas as pd


if __name__ == "__main__":
    varma = VARMA()

    dicio_resample = {"resample_1min":
                          ["../../Datasets/dataset_TPV/base_pump_23042023_A_resampled_1min",
                           "base_pump_23042023_A_resampled_1min.csv", "base_pump_23042023_A_resampled_1min"],
                      "resample_10min":
                          ["../../Datasets/dataset_TPV/base_pump_23042023_A_resampled_10min",
                           "base_pump_23042023_A_resampled_10min.csv", "base_pump_23042023_A_resampled_10min"]}

    # key = 'resample_1min'
    key = 'resample_10min'

    df = pd.read_csv(f"{dicio_resample[key][0]}/{dicio_resample[key][1]}")

    df = df[['InletPressure',
             'OutletPressure',
             'OutletTemperature',
             'InverterSpeed',
             # 'phaseA_active', # <<<<< algum problem nas phases <<
             # 'phaseB_active',
             # 'phaseC_active',
             'phaseA_current',
             'phaseB_current',
             'phaseC_current',
             'Time',
             'OAVelocity_y',
             'OAVelocity_x',
             'OAVelocity_z']]

    df = df[:800]

    df = df.dropna()

    df_train, df_test = train_test_split_hold_out(df)

    model_fit = varma.varma_fit(df_train)

    varma.save_model()

    model_fit = varma.load_model("modelo_varmax.pkl")

    yhat = varma.inference_microservice(length_prevision=200,
                                        model_fit=model_fit)

    dict_mae, dict_smape, dict_mse = varma.predict_metrics(yhat, df_test)

    print(f"dict mae {dict_mae}")
    print(f"dict dict_smape {dict_smape}")
    print(f"dict dict_mse {dict_mse}")


