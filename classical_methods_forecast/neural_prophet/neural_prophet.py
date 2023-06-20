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

    df_ = adjust_dataframe_for_train.get_data_frame()

    # write algorithm--------------------------------------------
    df_train = df_[:3016]
    df_test = df_[7881:]
    # ----------------------------------------------------------------
    # m = NeuralProphet(**configs["parameters_model"])
    train_model = TrainNeuralProphet(**configs["parameters_model"])

    # metrics = train_model.neural_prophet.fit(df_train)
    # train_model.save(configs["name"])

    train_model.load(configs["name"])

    m = train_model.neural_prophet
    # forecast = m.predict(df_)
    future = m.make_future_dataframe(df_train,
                                     periods=configs["predict_in_the_future"],
                                     n_historic_predictions=len(df))

    forecast = m.predict(df=future)

    y_truth = df_["y"].tolist()
    y_hat = forecast["yhat1"].tolist()
    ds = forecast["ds"]

    save_fig_forecast = SaveFigForecast()
    save_fig_forecast(ds, y_truth, y_hat, **configs["plot_config"])
