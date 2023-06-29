from utils_neural_prophet import AdjustDataFrameForTrain
from train_neural_prophet import TrainNeuralProphet
import pandas as pd
from utils.utils import read_yaml
import argparse
from save_fig_forecast import SaveFigForecast

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="neuralProphet main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    df = pd.read_csv(f"{configs['base_dataset']}/base_pump_23042023_A_resampled_10min.csv")

    adjust_dataframe_for_train = AdjustDataFrameForTrain(df, **configs)

    df_ = adjust_dataframe_for_train.eliminate_outliers(**configs["eliminate_outliers"])

    df_ = adjust_dataframe_for_train.get_data_frame(drop_zeros=True,
                                                    drop_constant=True)

    save_fig_forecast = SaveFigForecast()

    df_train, df_test = adjust_dataframe_for_train.dataset_split(df_, split=0.8)

    df_ = pd.concat([df_train, df_test], ignore_index=True)

    # ----------------------------------------------------------------

    train_model = TrainNeuralProphet(**configs["parameters_model"])

    metrics = adjust_dataframe_for_train.train_or_test(df_train, train_model, **configs)

    train_model.load(f'weighted_history/{configs["name"]}_{configs["select_feature"]}.np')

    m = train_model.neural_prophet

    future = m.make_future_dataframe(df_test,
                                     periods=configs["predict_in_the_future"],
                                     n_historic_predictions=len(df_test))

    forecast = m.predict(df=future)

    y_truth = df_train["y"].tolist()
    y_hat = forecast["yhat1"].tolist()
    ds_test = forecast["ds"]
    ds_train = df_train["ds"]

    print(len(y_truth), len(y_hat), abs(len(y_truth) - len(y_hat)))

    save_fig_forecast.plot_presentation(ds_train, ds_test,
                                        y_truth, y_hat, **configs["plot_config"])
