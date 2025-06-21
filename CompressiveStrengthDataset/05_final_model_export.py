#imports
import joblib
import sqlite3
import pandas as pd
import numpy as np
#MachineLearning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
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

#Tuned Final Model
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5],
}
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42), gb_params, cv=5,
    scoring='neg_root_mean_squared_error', n_jobs=-1)
gb_grid.fit(X_train, y_train)
final_model = gb_grid.best_estimator_

#Final RMSE Evaluation
rmse = np.sqrt(mean_squared_error(y_test, final_model.predict(X_test)))
print(f"Final RMSE on test set: {rmse:.2f}")

#Model Save
joblib.dump(final_model, 'final_model.pkl')

#Feature Columns Save
joblib.dump(list(X.columns), 'final_model_features.pkl')

#RMSE & Parameters Save
with open('final_model_info.txt', 'w') as f:
    f.write(f"Gradient Boosting Final Model\n")
    f.write(f"Best Parameters: {gb_grid.best_params_}\n")
    f.write(f"RMSE: {rmse:.2f}\n")

conn.close()

#Reload Function
def load_final_model(model_path='final_model.pkl', features_path='final_model_features.pkl'):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features