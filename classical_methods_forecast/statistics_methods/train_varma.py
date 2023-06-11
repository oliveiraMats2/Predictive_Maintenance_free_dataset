import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
import pickle
from metrics import avaliable_vector_auto_regressive_model


def train_test_split_hold_out(df):
    print(f"samples: {df.shape[0]}, begin: {df.Time[0]}, end: {df.Time[df.shape[0] - 1]}")
    print(f"samples test: {df.shape[0] // 4} | samples train: {3 * (df.shape[0] // 4)}")

    df = df.drop('Time', axis=1)

    one_quarter_base = len(df) // 4
    inv_one_quarter_base = len(df) - len(df) // 4

    df_train = df[:inv_one_quarter_base]
    df_test = df[inv_one_quarter_base:]

    return df_train, df_test


class VARMA:
    def __init__(self):
        self.model_fit = None

    def varma_fit(self, train: pd.DataFrame):
        model = VARMAX(train, order=(0, 1, 2))
        self.model_fit = model.fit(disp=False)
        return self.model_fit

    def inference_microservice(self, length_prevision, model_fit=None):
        if model_fit is None:
            yhat = self.model_fit.forecast(steps=length_prevision)
        else:
            yhat = model_fit.forecast(steps=length_prevision)
        return yhat

    def predict_metrics(self, yhat, df_test):
        dict_mae, dict_smape, dict_mse = avaliable_vector_auto_regressive_model(df_test, yhat, type_model="multiple")
        return dict_mae, dict_smape, dict_mse

    def save_model(self, filename="modelo_varmax.pkl", model_fit=None):
        if model_fit is None:
            with open(filename, 'wb') as file:
                pickle.dump(self.model_fit, file)
        else:
            with open(filename, 'wb') as file:
                pickle.dump(model_fit, file)

    def load_model(self, filename):

        with open(filename, 'rb') as file:
            model_fit = pickle.load(file)

        return model_fit
