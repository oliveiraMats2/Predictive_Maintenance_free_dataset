import pandas as pd
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster

# Cria um DataFrame de exemplo com variáveis independentes (X) e a variável alvo (y)
data = {
    'Data': pd.date_range(start='2023-01-01', periods=10),
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'y': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
}
df = pd.DataFrame(data)

# Define as configurações de previsão
forecast_horizon = 3  # Horizonte de previsão
config = ForecastConfig(
    model_template="SILVERKITE",
    forecast_horizon=forecast_horizon,
    coverage=0.95,
    metadata_param=MetadataParam(
        time_col="Data",  # Coluna de data/hora
        value_col="y",  # Coluna da variável alvo
        freq="D"  # Frequência dos dados (no exemplo, diário)
    )
)

# Cria um objeto Forecaster e realiza a previsão
forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=config
)

# Obtém os resultados da previsão
forecast = result.forecast
print(forecast)
