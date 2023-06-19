import pandas as pd

class AdjustDataFrameForTrain:
    def __init__(self, df_:pd.DataFrame, **configs) -> None:

        list_select_feature = df_[configs["select_feature"]].tolist()
        list_time = df_[configs["time"]].tolist()
        list_select_feature_len = len(list_select_feature)

        self.x = list(range(list_select_feature_len))
        self.y = list_select_feature
        self.x_time = list_time

    def get_data_frame(self):
        return pd.DataFrame({'ds': self.x_time, 'y': self.y})