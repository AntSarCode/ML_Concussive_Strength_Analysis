#imports
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
print(df.dtypes)
print(df.head())

#Variables - Features & Target
X = df.drop(columns=["compressive_strength"]) #features
y = df["compressive_strength"] #target

#Training Split
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42))

# -- RANDOM FOREST REGRESSOR IMPORTANCE --
#Model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Model Prediction
importances = model.feature_importances_
features = X.columns
rf_importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
rf_importances_df = rf_importances_df.sort_values(by='Importance', ascending=False)

#Plot
plt.figure(figsize = (10,6))
sns.barplot(x="Feature", y="Importance", data=rf_importances_df)
plt.title('Random Forest Regressor Strengths')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -- LINEAR REGRESSION IMPORTANCE --

#Model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#Model Prediction
importances = lr_model.coef_
features = X.columns
lr_coeffs_df = pd.DataFrame({'Feature': features, 'Coefficient': importances})
lr_coeffs_df['AbsCoeff'] = lr_coeffs_df['Coefficient'].abs()
lr_coeffs_df = lr_coeffs_df.sort_values(by='AbsCoeff', ascending=False)

#Plot
sns.barplot(x="AbsCoeff", y="Feature", data=lr_coeffs_df)
plt.title("Linear Regression Coefficient Strengths")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# -- GRADIENT BOOSTING REGRESSOR IMPORTANCE --

#Model
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

#Model Prediction
importances = gb_model.feature_importances_
features = X.columns
gb_importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
gb_importances_df['AbsCoeff'] = gb_importances_df['Importance'].abs()
gb_importances_df = gb_importances_df.sort_values(by='AbsCoeff', ascending=False)

#Plot
sns.barplot(x="AbsCoeff", y="Feature", data=gb_importances_df)
plt.title("Gradient Boosting Regressor Strengths")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

conn.close()


