from neuralprophet import NeuralProphet
import pandas as pd

# Cria um DataFrame de exemplo com variáveis independentes (X1, X2) e a variável alvo (y)
from neuralprophet import NeuralProphet
import pandas as pd

# Cria um DataFrame de exemplo com variáveis independentes (X1, X2) e a variável alvo (y)
data = {
    'ds': pd.date_range(start='2023-01-01', periods=10),
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'y': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
}
df = pd.DataFrame(data)

# Renomeia as colunas independentes para 'regressor1' e 'regressor2'
df.rename(columns={'X1': 'regressor1', 'X2': 'regressor2'}, inplace=True)

# Cria o modelo NeuralProphet
model = NeuralProphet(
    n_forecasts=3,  # Horizonte de previsão
    n_lags=3  # Número de lags (atrasos) a serem considerados nas variáveis independentes
)

# Treina o modelo com os dados
model.fit(df, freq='D')

# Gera as previsões para o horizonte especificado
future = model.make_future_dataframe(df, periods=3)
forecast = model.predict(future)

# Imprime as previsões

# Imprime as previsões
print(forecast[['ds', 'yhat', 'yhat1_lower', 'yhat1_upper', 'yhat2_lower', 'yhat2_upper', 'yhat3_lower', 'yhat3_upper']])
