#imports
import sqlite3
import pandas as pd
import numpy as np
#MachineLearning
from sklearn.model_selection import train_test_split

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

#Training/Test split
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42))

# -- DUMMY REGRESSOR & RMSE TEST --
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

#Model
dum_model = DummyRegressor(strategy="mean")  # or "median"
dum_model.fit(X_train, y_train)
#Prediction & Comparison
y_pred_dum = dum_model.predict(X_test)
rmse_dum = np.sqrt(mean_squared_error(y_test, y_pred_dum))


# -- RANDOM FOREST REGRESSOR IMPORTANCE & RMSE TEST --
from sklearn.ensemble import RandomForestRegressor

#Model
fr_model = RandomForestRegressor(n_estimators=100, random_state=42)
fr_model.fit(X_train, y_train)
#Prediction & Comparison
y_pred_fr = fr_model.predict(X_test)
rmse_fr = np.sqrt(mean_squared_error(y_test, y_pred_fr))

# -- LINEAR REGRESSION COEFFICIENT IMPORTANCE + RMSE TEST --
from sklearn.linear_model import LinearRegression

#Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
#Prediction & Comparison
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# -- GRADIENT BOOSTING REGRESSOR IMPORTANCE + RMSE TEST --
from sklearn.ensemble import GradientBoostingRegressor

#Model
gb_model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
#Prediction & Comparison
y_pred_gb = gb_model.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

#Outputs
print('RMSE Scores for Evaluated Models:')
print(f"Linear Regression RMSE: {rmse_lr:.2f}")
print(f"Random Forest Regressor RMSE: {rmse_fr:.2f}")
print(f"Gradient Boosting Regressor RMSE: {rmse_gb:.2f}")

#Output Integration
results = pd.DataFrame({
    "Model": ["Dummy Regression", "Linear Regression", "Random Forest", "Gradient Boosting"],
    "RMSE": [rmse_dum, rmse_lr, rmse_fr, rmse_gb],
})
results = results.sort_values(by="RMSE")
print(results.to_string(index=False))

#Output Condition & Baseline Performance
best_model = results.iloc[0]
theory_baseline_rmse = rmse_dum
practice_baseline_rmse = rmse_lr
theory_improvement = 100 * (theory_baseline_rmse - best_model["RMSE"]) / theory_baseline_rmse
practice_improvement = 100 * (practice_baseline_rmse - best_model["RMSE"]) / practice_baseline_rmse
print(f"\nThe Best Model is {best_model['Model']} with RMSE: {best_model['RMSE']:.2f}")
print(f"Performance Improvement vs Theoretical Baseline: {theory_improvement:.2f}%")
print(f"Performance Improvement vs Practical Baseline: {practice_improvement:.2f}%")

#Plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x="Model", y="RMSE", data=results, order=results['Model'])
plt.title("Model RMSE Comparison")
plt.tight_layout()
plt.xlabel("Model")
plt.ylabel("Root Mean Squared Error")
plt.savefig("rmse_comparison.png", dpi=300)
plt.show()

conn.close()