import pandas as pd
import numpy as np


def create_dataframe(start_date, end_date, interval):
    # Cria uma sequência de datas no intervalo especificado
    dates = pd.date_range(start=start_date, end=end_date, freq=interval)

    # Cria um DataFrame com as colunas 'ds' e 'y'
    df = pd.DataFrame({'ds': dates, 'y': np.zeros(len(dates))})

    return df


# Exemplo de uso
start_date = '2023-04-22 20:40:00'
end_date = '2023-08-01 00:10:00'
interval = '10min'

df = create_dataframe(start_date, end_date, interval)
print(df)
