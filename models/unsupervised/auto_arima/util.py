import pandas as pd
import xgboost as xgb

# Carrega os dados de treinamento
train_data = pd.read_csv('../../../Datasets/dataset_TPV_sensors/hex/payloadHex.csv', index_col=0)

# Define as variáveis preditoras (features) e a variável de resposta (target)
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']

# Define os parâmetros do modelo
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'n_estimators': 100,
    'learning_rate': 0.1
}

# Cria o modelo XGBoost
xgb_model = xgb.XGBRegressor(**params)

# Ajusta o modelo com os dados de treinamento
xgb_model.fit(X_train, y_train)

# Carrega os dados de teste
# test_data = pd.read_csv('test_data.csv', index_col=0)
#
# # Realiza previsões para novos dados
# X_test = test_data.drop('target', axis=1)
# y_pred = xgb_model.predict(X_test)
#
# print(y_pred)  # imprime as previsões para cada período futuro
