from utils_neural_prophet import AdjustDataFrameForTrain
from prophet import Prophet
import pandas as pd
from utils import read_yaml
import argparse
from save_fig_forecast import SaveFigForecast
from prophet.serialize import model_to_json, model_from_json

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
import numpy as np

def save_metrics(test, pred, **configs):
    sum_df = pd.DataFrame(index = ['MAE', 'MSE', 'SMAPE'])
    sum_mae = 0
    sum_mse = 0
    sum_smape = 0
    
    # for index, feature in enumerate(X_test.keys()):
    mae = mean_absolute_error(test['y'], pred['yhat'])
    sum_mae = sum_mae + mae
    
    mse = mean_squared_error(test['y'], pred['yhat'])
    sum_mse = sum_mse + mse
    
    smape = np.mean(2 * np.abs((pred['yhat'].values - test['y'].values)) / (abs(pred['yhat'].values) + np.abs(test['y'].values) + 1e-8))*100
    sum_smape = sum_smape + smape
    
    sum_df[configs["select_feature"]] = [mae, mse, smape]

    # sum_df['Média'] = [np.mean(sum_mae), np.mean(sum_mse), np.mean(sum_smape)]
    sum_df.to_csv(f'metrics/metrics_{configs["select_feature"]}.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="neuralProphet main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    df = pd.read_csv(f"{configs['base_dataset']}")
    
    # just for WISE
    if 'OAVelocity' in configs["select_feature"]:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df[df['Time'] < pd.Timestamp(day=6, month=3, year=2023)]

    adjust_dataframe_for_train = AdjustDataFrameForTrain(df, **configs)
    
    if 'eliminate_outliers' in configs.keys():
        print('Eliminating outliers...')
        df_ = adjust_dataframe_for_train.eliminate_outliers(**configs["eliminate_outliers"])

    df_ = adjust_dataframe_for_train.get_data_frame(drop_zeros=True,
                                                    drop_constant=True)

    save_fig_forecast = SaveFigForecast()

    df_train, df_test = adjust_dataframe_for_train.dataset_split(df_, split=0.8)

    df_ = pd.concat([df_train, df_test], ignore_index=True)

    # ----------------------------------------------------------------

    train_model = Prophet()

    metrics = adjust_dataframe_for_train.train_or_test(df_train, train_model, is_train=True, **configs)

    with open(f'weighted_history/{configs["name"]}_{configs["select_feature"]}.json', 'r') as fin:
        train_model = model_from_json(fin.read())  # Load model

    m = train_model

    # print('configs')
    # print(configs["predict_in_the_future"])
    # future = m.make_future_dataframe(df_test,
    #                                  periods=configs["predict_in_the_future"],
    #                                  n_historic_predictions=len(df_test))

    forecast = m.predict(df=df_test)

    y_truth = df_train["y"].tolist()
    y_hat = forecast["yhat"].tolist()
    ds_test = forecast["ds"]
    ds_train = df_train["ds"]

    print(len(y_truth), len(y_hat), abs(len(y_truth) - len(y_hat)))

    save_fig_forecast.plot_presentation(ds_train, ds_test,
                                        y_truth, y_hat, **configs["plot_config"])
    
    save_metrics(df_test, forecast, **configs["plot_config"])
