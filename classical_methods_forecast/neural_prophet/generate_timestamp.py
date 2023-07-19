import pandas as pd


class GenerateTimestamp:

    @staticmethod
    def generate_timestamps_delimiter(start, end, freq="10S"):
        timestamps = pd.date_range(start=start, end=end, freq=freq)

        data = {"y": [0] * len(timestamps), "ds": timestamps}
        return pd.DataFrame(data)
