
import pandas as pd
import statsmodels.api as sm

# Cria um DataFrame de exemplo com variáveis independentes (X1, X2) e a variável alvo (y)
data = {
    'Data': pd.date_range(start='2023-01-01', periods=10),
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'y': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
}
df = pd.DataFrame(data)

# Define as variáveis independentes e a variável alvo
X = df[['X1', 'X2']]
y = df['y']

# Adiciona uma constante aos dados independentes
X = sm.add_constant(X)

# Cria o modelo de regressão multivariada
model = sm.OLS(y, X)

# Treina o modelo
results = model.fit()

# Imprime os resultados
print(results.summary())

# Gera as previsões
future_data = {
    'Data': pd.date_range(start='2023-01-11', periods=3),
    'X1': [11, 12, 13],
    'X2': [21, 22, 23]
}
future_df = pd.DataFrame(future_data)
future_X = sm.add_constant(future_df[['X1', 'X2']])
predictions = results.predict(future_X)

# Imprime as previsões
forecast = future_df.copy()
forecast['yhat'] = predictions
print(forecast[['Data', 'yhat']])
