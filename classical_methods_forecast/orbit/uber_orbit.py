from orbit.utils.dataset import load_iclaims
from orbit.models import DLT
from orbit.diagnostics.plot import plot_predicted_data
import pandas as pd

df_1 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/x.csv')
df_2 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/y.csv')
df_3 = pd.read_csv('../../Datasets/dataset_TPV_sensors/wise/z.csv')

# Combina as colunas relevantes dos DataFrames
df = pd.DataFrame()
df["OAVelocity_x"] = df_1["OAVelocity"].astype('float64')
df["OAVelocity_y"] = df_2["OAVelocity"].astype('float64')
df["OAVelocity_z"] = df_3["OAVelocity"].astype('float64')
df["ds"] = pd.to_datetime(df_1["Time"])

# Remove as datas duplicadas
df = df.drop_duplicates(subset=["ds"])

# Ordena o DataFrame pelo índice de datas
df = df.sort_values('ds')

# Redefine o índice do DataFrame
df = df.reset_index(drop=True)

# Imprime o DataFrame atualizado
print("DataFrame com datas organizadas:")
print(df)

splitter = 10000

train_df = df[:splitter]
test_df = df[splitter:]

dlt = DLT(
  response_col='OAVelocity_x', date_col='ds',
  regressor_col=['OAVelocity_y', 'OAVelocity_z'],
  seasonality=52,
)
dlt.fit(df=train_df)

# outcomes data frame
predicted_df = dlt.predict(df=train_df)

plot_predicted_data(
  training_actual_df=train_df, predicted_df=predicted_df,
  date_col=dlt.date_col, actual_col=dlt.response_col,
  test_actual_df=test_df
)