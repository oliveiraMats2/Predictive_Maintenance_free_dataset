from utils_neural_prophet import AdjustDataFrameForTrain
from train_neural_prophet import TrainNeuralProphet
import pandas as pd
from utils.utils import read_yaml
import argparse
import matplotlib.pyplot as plt

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


#write algorithm--------------------------------------------
    df_train = df_[:3016]
    df_test = df_[7881:]
# ----------------------------------------------------------------
    # m = NeuralProphet(**configs["parameters_model"])
    train_model = TrainNeuralProphet(**configs["parameters_model"])

    train_model.save()
    train_model.load()

    metrics = train_model.neural_prophet.fit(df_train)

    forecast = train_model.neural_prophet.predict(df_test)

    x = forecast["y"].tolist()
    x_arange = list(range(len(x)))

    x_truth = forecast["yhat1"].tolist()
    x_arange_truth = list(range(len(x)))

    plt.scatter(x_arange_truth, x_truth)
    plt.scatter(x_arange, x)

    fig_forecast = train_model.neural_prophet.plot(forecast)
    fig_components = train_model.neural_prophet.plot_components(forecast)
    fig_model = train_model.neural_prophet.plot_parameters()


    future = train_model.neural_prophet.make_future_dataframe(df_,
                                                              periods=6,
                                                              n_historic_predictions=len(df))

    # forecast = m.predict(future)
    #
    # # plotar pontos na previsão.
    # fig_forecast = m.plot(forecast)
    # fig_forecast
    #
    # fig_components = m.plot_components(forecast)
    # fig_components