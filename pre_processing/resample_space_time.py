import pandas as pd


class ResampleSpaceTime:
    def __init__(self,
                 init_time: str = "00:00:10",
                 delta_time: str = "00:00:11"):

        self.temp_init = pd.Timedelta(init_time)
        self.delta_time = pd.Timedelta(delta_time)

    def resample_up_sampling(self, data:pd.DataFrame):
        print(f"Samples before interpolate {data.shape[0]}")
        data = data.reset_index()

        data["index_time"] = data.index * self.delta_time + self.temp_init

        data = data.set_index("index_time")

        self.data_resampled = data.resample(self.delta_time).ffill()

        self.data_resampled['diff'] = self.delta_time

        df_resampled = self.data_resampled.reset_index()

        df_resampled["Time"] = df_resampled["index_time"]

        df_resampled = df_resampled.drop(columns=['index_time', 'index', 'diff'])

        print(f"Samples after interpolate {df_resampled.shape[0]}")

        return df_resampled
