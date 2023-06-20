import pandas as pd
import numpy as np


class AdjustDataFrameForTrain:
    def __init__(self, df_: pd.DataFrame, **configs) -> None:
        list_select_feature = df_[configs["select_feature"]].tolist()
        list_time = df_[configs["time"]].tolist()
        list_select_feature_len = len(list_select_feature)

        self.x = list(range(list_select_feature_len))
        self.y = list_select_feature
        self.x_time = list_time

    def get_data_frame(self, drop_zeros=True):
        df_ = pd.DataFrame({'ds': self.x_time, 'y': self.y})
        if drop_zeros:
            df_ = df_.replace(0.0, np.nan)
            df_ = df_.dropna()

        return df_
