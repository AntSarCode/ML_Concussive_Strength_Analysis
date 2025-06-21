#imports
import sqlite3
import pandas as pd
import numpy as np
#MachineLearning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

#SqLite Bridge
conn = sqlite3.connect(r'C:/Users/takis/PycharmProjects/Machine_Learning/concrete_strength.db')

#SQL Import
query = """
SELECT * FROM concrete_strength;
"""
#Dataframe
df = pd.read_sql_query(query, conn)
df = df.dropna(subset=["compressive_strength"])

#Variables - Features & Target
X = df.drop(columns=["compressive_strength"]) #features
y = df["compressive_strength"] #target

#Training/Test Split
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42))

# -- LINEAR REGRESSION TUNING -- (Included only for comparison)
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
rmse_lr = np.sqrt(mean_squared_error(y_test, lr_model.predict(X_test)))

# -- RANDOM FOREST REGRESSOR TUNING --
from sklearn.ensemble import RandomForestRegressor

fr_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
fr_grid = GridSearchCV(
    RandomForestRegressor(random_state=42), fr_params, cv=5,
    scoring='neg_root_mean_squared_error', n_jobs=-1)
fr_grid.fit(X_train, y_train)
best_fr = fr_grid.best_estimator_
rmse_fr = np.sqrt(mean_squared_error(y_test, best_fr.predict(X_test)))

# -- GRADIENT BOOSTING REGRESSOR TUNING --
from sklearn.ensemble import GradientBoostingRegressor

gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5],
}
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42), gb_params, cv=5,
    scoring='neg_root_mean_squared_error', n_jobs=-1)
gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_
rmse_gb = np.sqrt(mean_squared_error(y_test, best_gb.predict(X_test)))

# -- TUNING RESULTS SUMMARY --
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest (Tuned)',
              'Gradient Boosting (Tuned)'],
    'RMSE': [rmse_lr, rmse_fr, rmse_gb],
})
results = results.sort_values(by='RMSE')
print(results.to_string(index=False))
print("\nBest Parameters:")
print('Random Forest Regressor:', fr_grid.best_params_)
print('Gradient Boosting Regressor:', gb_grid.best_params_)

conn.close()