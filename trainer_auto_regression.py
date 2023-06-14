import pandas as pd
import matplotlib.pyplot as plt
from utils.read_dataset import ReadDatasets
from statsmodels.iolib.smpickle import load_pickle
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from figures_save import SaveFigures
from metrics import avaliable_vector_auto_regressive_model


class VectorAutoRegressionModel:

    def __init__(self, name_model="vector_auto_regressive"):
        self.name_model = name_model

    def define_df(self, vector_series):
        return pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),  # )
                             "ds": np.arange(len(vector_series))})

    def train_test_split_hold_out(self, df):
        print(f"samples: {df.shape[0]}, begin: {df.Time[0]}, end: {df.Time[df.shape[0] - 1]}")
        print(f"samples test: {df.shape[0] // 4} | samples train: {3 * (df.shape[0] // 4)}")

        df = df.drop('Time', axis=1)

        one_quarter_base = len(df) // 4
        inv_one_quarter_base = len(df) - len(df) // 4

        df_train = df[:inv_one_quarter_base]
        df_test = df[inv_one_quarter_base:]

        return df_train, df_test

    def inference_for_microservice(self, df_train, window_input, steps):

        lagged_Values = df_train.values[-window_input:]
        pred = self.result.forecast(y=lagged_Values, steps=steps)

        col_predicted = [f"{x}_forecast" for x in df_train.keys()]

        df_forecast = pd.DataFrame(data=pred, columns=col_predicted)

        return df_forecast

    def predict(self, df_train, df_test, window_input, steps, first_limiar, sub_path):
        lagged_Values = df_train.values[-window_input:]
        pred = self.result.forecast(y=lagged_Values, steps=steps)

        col_predicted = [f"{x}_forecast" for x in df_train.keys()]

        df_forecast = pd.DataFrame(data=pred, columns=col_predicted)

        df_test.index = pd.to_datetime(df_test.index)

        # SaveFigures.save(df_test=df_test,
        #                  df_pred=df_forecast,
        #                  lim_end=steps,
        #                  first_limiar=first_limiar,
        #                  sub_path=sub_path)

        ground_truth, pred = df_test, df_forecast

        pred_slice = pred.shape[0]
        ground_truth = ground_truth[:pred_slice]

        return ground_truth, pred

    def fit(self, df, order=50):

        df_train, df_test = self.train_test_split_hold_out(df)

        model = VAR(df_train)

        self.result = model.fit(order)
        self.__save_model()

    def __save_model(self):
        self.result.save(f"{self.name_model}.pickle")

    def load_model(self, dir_model=None):
        if dir_model is None:
            results = load_pickle(f"{self.name_model}.pickle")

        else:
            results = load_pickle(dir_model)

        return results

    def generate_metrics(self, ground_truth, pred):
        mean_abs, smape_loss, mean_square_error = avaliable_vector_auto_regressive_model(ground_truth,
                                                                                         pred,
                                                                                         "multiple")

        return mean_abs, smape_loss, mean_square_error