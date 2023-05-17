import pandas as pd


def filter_dataframe(df, start_date, end_date):
    df = df[1:]# remover Timestamp
    df['Time'] = pd.to_datetime(df['Time'])  # Converte a coluna Time para datetime
    mask = (df['Time'] >= start_date) & (df['Time'] <= end_date)  # Cria a máscara para filtrar as datas
    filtered_df = df.loc[mask]  # Filtra o DataFrame
    return filtered_df
