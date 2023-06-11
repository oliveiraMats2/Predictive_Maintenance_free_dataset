from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum

# Defines inputs
df = DataLoader().load_bikesharing().tail(24*90)  # Input time series (pandas.DataFrame)
config = ForecastConfig(
     metadata_param=MetadataParam(time_col="ts", value_col="count"),  # Column names in `df`
     model_template=ModelTemplateEnum.AUTO.name,  # AUTO model configuration
     forecast_horizon=24,   # Forecasts 24 steps ahead
     coverage=0.95,         # 95% prediction intervals
 )

# Creates forecasts
forecaster = Forecaster()
result = forecaster.run_forecast_config(df=df, config=config)

# Accesses results
result.forecast     # Forecast with metrics, diagnostics
result.backtest     # Backtest with metrics, diagnostics
result.grid_search  # Time series CV result
result.model        # Trained model
result.timeseries   # Processed time series with plotting functions