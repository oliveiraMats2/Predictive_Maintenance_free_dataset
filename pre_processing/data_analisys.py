import pandas as pd
import matplotlib.pyplot as plt


def plot_frequency(indices: list, frequencias: list, name: str, color: str) -> None:
    plt.bar(indices, frequencias, color=color)

    # Configurações do gráfico
    plt.xlabel('Diferença')
    plt.ylabel('Frequência')
    plt.title(f'Frequência de Diferenças de Tempo {name}')
    plt.xticks(rotation=45)
    plt.grid(True)
    for i, v in enumerate(frequencias):
        plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    # Exibir o gráfico
    plt.show()
    return None


class DataAnalisysTimeStamp:
    def __init__(self,
                 df: pd.DataFrame,
                 col_time: str = "Time",
                 sort_index_data: bool = True) -> None:

        if type(df[col_time][0]) is str:
            df[col_time] = pd.to_datetime(df[col_time])

        df['diff'] = df[col_time].diff().abs()

        df_value_counts = df["diff"].value_counts()

        if sort_index_data is True:
            df_value_counts.sort_index(inplace=True)

        self.time_diff = [str(x) for x in list(df_value_counts.index)]

        self.frequency_count = list(df_value_counts.values)

    def filter_by_frequency(self, slices):
        self.time_diff = self.time_diff[:slices]
        self.frequency_count = self.frequency_count[:slices]

    def plot_frequency_timestamp(self, name_sensor: str, color="blue"):
        index_s = self.time_diff
        frequency = self.frequency_count

        plot_frequency(index_s, frequency, name_sensor, color=color)
