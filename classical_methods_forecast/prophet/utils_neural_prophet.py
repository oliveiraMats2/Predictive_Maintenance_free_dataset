import pandas as pd
import numpy as np
from prophet.serialize import model_to_json, model_from_json


class AdjustDataFrameForTrain:
    def __init__(self, df_: pd.DataFrame, **configs) -> None:
        list_select_feature = df_[configs["select_feature"]].tolist()
        list_time = df_[configs["time"]].tolist()
        list_select_feature_len = len(list_select_feature)

        self.x = list(range(list_select_feature_len))
        self.y = list_select_feature
        self.x_time = list_time

        self.df_ = pd.DataFrame({'ds': self.x_time, 'y': self.y})

    def eliminate_outliers(self, apply=False, inferior=0.05, superior=0.95):
        if apply:
            return None

        # Define a threshold for outlier detection (e.g., z-score > 3)
        Q1 = self.df_['y'].quantile(inferior)
        Q3 = self.df_['y'].quantile(superior)

        # Calculate the IQR (Interquartile Range)
        IQR = Q3 - Q1

        # Define the upper and lower bounds for outlier detection
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find the outliers
        # outliers = self.df_[(self.df_['y'] < lower_bound) | (self.df_['y'] > upper_bound)]

        # Discard the outliers
        self.df_ = self.df_[(self.df_['y'] >= lower_bound) & (self.df_['y'] <= upper_bound)]

    def get_data_frame(self, drop_zeros, drop_constant):
        if drop_zeros:
            self.df_ = self.df_.replace(0.0, np.nan)
            self.df_ = self.df_.dropna()

        if drop_constant:
            self.df_ = self.df_.replace(0.15, np.nan)
            self.df_ = self.df_.dropna()

        return self.df_

    def dataset_split(self, df_: pd.DataFrame, split: int = 0.8) -> (pd.DataFrame, pd.DataFrame):

        max_size = len(df_)
        slice_choose = int((max_size) * split)

        df_train = df_[:slice_choose]
        df_test = df_[slice_choose:]

        print(f"length train: {df_train.shape[0]}")
        print(f"length test: {df_test.shape[0]}")

        return df_train, df_test

    @staticmethod
    def train_or_test(df_train, train_model, is_train=True, **configs):
        if is_train:
            # Train Prophet model
            metrics = train_model.fit(df_train) 

            # Save model trained
            with open(f'weighted_history/{configs["name"]}_{configs["select_feature"]}.json', 'w') as fout:
                fout.write(model_to_json(train_model))

            return metrics
        else:
            # Not used
            train_model.load(f'weighted_history/{configs["name"]}_{configs["select_feature"]}.np')
