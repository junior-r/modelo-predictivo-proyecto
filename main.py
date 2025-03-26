import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Cargar datos simulados de ventas
data = {
    'fecha': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'ventas': np.random.randint(50, 500, 100),
    'precio': np.random.uniform(10, 100, 100),
    'publicidad': np.random.uniform(1000, 10000, 100),
    'stock_disponible': np.random.randint(100, 1000, 100)
}
df = pd.DataFrame(data)

# Convertir fecha en variable numérica
df['dia'] = df['fecha'].dt.dayofyear
df.drop(columns=['fecha'], inplace=True)

# Crear nuevas características
df['precio_x_stock'] = df['precio'] * df['stock_disponible']
df['log_publicidad'] = np.log1p(df['publicidad'])

# Separar variables predictoras y objetivo
X = df[['dia', 'precio', 'publicidad', 'stock_disponible', 'precio_x_stock', 'log_publicidad']]
y = df['ventas']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimización de hiperparámetros con GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']  # Cambié 'auto' por valores válidos
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Mostrar los mejores parámetros
print(f"Mejores parámetros: {grid_search.best_params_}")

# Obtener el modelo entrenado con los mejores parámetros
best_model = grid_search.best_estimator_

# Predicciones con el mejor modelo
y_pred_rf = best_model.predict(X_test_scaled)

# Evaluación del modelo
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print(f"MAE (Random Forest): {mae_rf:.2f}, RMSE (Random Forest): {rmse_rf:.2f}")

# Usar XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predicciones con XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluación del modelo XGBoost
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

print(f"MAE (XGBoost): {mae_xgb:.2f}, RMSE (XGBoost): {rmse_xgb:.2f}")

# Enfoque de Stacking
base_learners = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42))
]

stacking_model = StackingRegressor(estimators=base_learners, final_estimator=RandomForestRegressor())
stacking_model.fit(X_train_scaled, y_train)

# Predicciones con Stacking
y_pred_stacking = stacking_model.predict(X_test_scaled)

# Evaluación del modelo Stacking
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
rmse_stacking = np.sqrt(mse_stacking)

print(f"MAE (Stacking): {mae_stacking:.2f}, RMSE (Stacking): {rmse_stacking:.2f}")

# Visualizar resultados de Random Forest
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel("Ventas Reales")
plt.ylabel("Ventas Predichas (RF)")
plt.title("Comparación de Ventas Reales vs. Predichas - Random Forest")
plt.show()

# Visualizar resultados de XGBoost
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_xgb)
plt.xlabel("Ventas Reales")
plt.ylabel("Ventas Predichas (XGBoost)")
plt.title("Comparación de Ventas Reales vs. Predichas - XGBoost")
plt.show()

# Visualizar resultados de Stacking
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_stacking)
plt.xlabel("Ventas Reales")
plt.ylabel("Ventas Predichas (Stacking)")
plt.title("Comparación de Ventas Reales vs. Predichas - Stacking")
plt.show()
