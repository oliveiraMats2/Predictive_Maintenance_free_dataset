from utils_neural_prophet import AdjustDataFrameForTrain
from train_neural_prophet import TrainNeuralProphet
import pandas as pd
from utils.utils import read_yaml
import argparse
from save_fig_forecast import SaveFigForecast


def find_value_lower(a, b):
    if a < b:
        return a
    else:
        return b


def truncate_values(y_truth, yhat):
    truncate_value = find_value_lower(len(y_truth), len(yhat))
    y_truth = y_truth[:truncate_value]
    yhat = yhat[:truncate_value]

    return y_truth, yhat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="neuralProphet main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    df = pd.read_csv(f"{configs['base_dataset']}/base_pump_23042023_A_resampled_10min.csv")

    adjust_dataframe_for_train = AdjustDataFrameForTrain(df, **configs)

    adjust_dataframe_for_train.eliminate_outliers(**configs["eliminate_outliers"])

    adjust_dataframe_for_train.eliminate_range_values(**configs["eliminate_range_outliers"])

    df_ = adjust_dataframe_for_train.get_data_frame(**configs["drop"])

    save_fig_forecast = SaveFigForecast()

    df_train, df_test = adjust_dataframe_for_train.dataset_split(df_, split=configs["train_test_split"])

    df_ = pd.concat([df_train, df_test], ignore_index=True)

    dict_models = {"model":
                       {"InletPressure": None,
                        "InverterSpeed": None,
                        "OAVelocity_x": None,
                        "OAVelocity_y": None,
                        "OAVelocity_z": None,
                        "OutletPressure": None,
                        "OutletTemperature": None,
                        "phaseA_current": None,
                        "phaseA_voltage": None,
                        "phaseB_current": None,
                        "phaseB_voltage": None,
                        "phaseC_current": None,
                        "phaseC_voltage": None,
                        "temperature": None,
                        }
                   }

    for key in dict_models["model"].keys():
        dict_models["model"][key] = TrainNeuralProphet(**configs["parameters_model"])
        dict_models["model"][key].load(f'weighted_history/{configs["name"]}_{configs["select_feature"]}.np')

    dict_future = {}

    for key in dict_models["model"].keys():
        dict_future[key] = dict_models["model"][key].neural_prophet.make_future_dataframe(df_train,
                                                                                          periods=configs[
                                                                                              "predict_in_the_future"],
                                                                                          n_historic_predictions=len(
                                                                                              df_train))

    dict_forecast = {}

    for key in dict_models["model"].keys():
        dict_forecast[key] = dict_models["model"][key].neural_prophet.predict(df=dict_future[key])

    for key in dict_forecast.keys():
        y_truth = df_train["y"].tolist()
        y_hat = dict_forecast[key]["yhat1"].tolist()
        ds_test = dict_forecast[key]["ds"]
        ds_train = df_train["ds"]
        configs["plot_config"]["select_feature"] = key
        #
        # if configs["metrics"]:
        #     y_truth, y_hat = truncate_values(y_truth, y_hat)
        #
        #     avaliable_vector_auto_regressive_model(y_truth, y_hat, type_model="single")
        #
        # else:
        #     print(len(y_truth), len(y_hat), abs(len(y_truth) - len(y_hat)))
        #
        save_fig_forecast.plot_presentation(ds_train, ds_test,
                                            y_truth, y_hat, **configs["plot_config"])
