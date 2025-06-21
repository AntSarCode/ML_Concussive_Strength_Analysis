# ml_utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def plot_feature_importance(model, X, title):
    """
    Extracts and plots feature importances or coefficients from a fitted model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        label = "Importance"
    elif hasattr(model, 'coef_'):
        importances = model.coef_
        label = "Coefficient"
    else:
        raise ValueError("Model does not support feature importance or coefficient extraction.")

    features = X.columns
    fi_df = pd.DataFrame({'Feature': features, label: importances})

    if label == "Coefficient":
        fi_df["Abs"] = fi_df[label].abs()
        fi_df = fi_df.sort_values(by="Abs", ascending=False)
        x_col = "Abs"
    else:
        fi_df = fi_df.sort_values(by=label, ascending=False)
        x_col = label

    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y="Feature", data=fi_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def evaluate_default_model(model, name, X_train, X_test, y_train, y_test):
    """
    Fits a default model and returns name and RMSE.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return name, rmse

def tune_model(model, param_grid, name, X_train, X_test, y_train, y_test, cv=5):
    """
    Performs GridSearchCV tuning, returns name, RMSE, and best params.
    """
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return name, rmse, grid.best_params_
