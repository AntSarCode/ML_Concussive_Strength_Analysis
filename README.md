# ML_Concussive_Strength_Analysis

This project focuses on predicting the compressive strength of concrete using supervised machine learning techniques. It follows a complete modeling pipeline including data preparation, feature engineering, model comparison, hyperparameter tuning, final model export, and deployment preparation. The goal is to simulate a practical data science workflow in a structured and reproducible way.

## Technologies and Tools

- Python (pandas, numpy, matplotlib, seaborn, joblib)
- scikit-learn (regression, model selection, metrics)
- SQLite (data source)
- Jupyter Notebook and/or standalone Python scripts

## Project Structure

ML_Concussive_Strength_Analysis/
├── CompressiveStrengthDataset/
│ ├── 01_feature_importance.py
│ ├── 02_baseline_model_comparison.py
│ ├── 03_model_tuning.py
│ ├── 04_tuned_model_comparison.py
│ ├── 05_final_model_export.py
│ ├── 06_deployment_prep.py
│ ├── concrete_strength.db
│ ├── final_model.pkl
│ ├── final_model_features.pkl
│ ├── final_model_info.txt
│ ├── identifier.sqlite
│ ├── ml_utils.py
│ ├── rmse_comparison.png
│ └── other supporting files
├── README.md

## Workflow Overview

1. **Data Source**: Reads from a SQLite database of compressive strength measurements.
2. **Feature Engineering**: Includes feature importance ranking and variable selection.
3. **Baseline Modeling**: Evaluates multiple regressors (e.g., Linear Regression, Decision Tree, Random Forest).
4. **Model Tuning**: Applies GridSearchCV to optimize model performance.
5. **Final Model Export**: Saves the trained model and related metadata for deployment.
6. **Deployment Prep**: Loads model and prediction schema for use in production or analysis contexts.

## How to Run the Project

  1. Clone the repository:

     ```bash
     git clone https://github.com/AntSarCode/ML_Concussive_Strength_Analysis.git

  2. Install dependencies:
    pip install -r requirements.txt

  3. Run the scripts sequentially (01 through 06) or open the matching Jupyter notebooks to follow along with the process.

  4. Review visual output and final model files in the project directory.

## Purpose

This project was developed to:

  - Demonstrate the end-to-end process of supervised model development

  - Practice working with structured SQLite data

  - Build reproducible pipelines for use in deployment or presentation contexts

## License

This project is licensed under the MIT License. You are free to use, adapt, and distribute it for educational or professional purposes.
