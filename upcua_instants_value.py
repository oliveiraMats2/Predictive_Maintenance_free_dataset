import time
import pandas as pd
from src.utils import opcua_tools as op


class UpcuaInstantValues:
    def __init__(self, feature: str = "PhaseA-voltage") -> None:
        self.feature = feature

    def get_on_upcua(self, range_of_features: int) -> list[list, list]:
        results = []
        date_time = []

        for i in range(range_of_features):
            output = op.get_single(self.feature)
            results.append(output[0])
            date_time.append(output[1])

            print(f"Cycle {i}: value {output[0]} at {output[1]}")

            # Wait for 1 minute
            time.sleep(1)

        return results, date_time

    def actual_dataframe(self, range_of_samples: int = 100):
        results, date_time = self.get_on_upcua(range_of_samples)
        return pd.DataFrame({"Time": date_time, self.feature: results})
