#imports
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = 'final_model.pkl'
FEATURES_PATH = 'final_features.pkl'

def load_model(model_path=MODEL_PATH, features_path=FEATURES_PATH):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

def preprocess_input(df, feature_columns):
    """
    Validates DataFrame and training structure match.
    Extra columns are dropped; missing columns filled with 0.
    """
    df = df.copy()
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

def predict_from_file(input_csv, output_csv='predictions.csv'):
    model, features = load_model()
    input_df = pd.read_csv(input_csv)
    input_df_processed = preprocess_input(input_df, features)
    preds = model.predict(input_df_processed)
    input_df['prediction'] = np.round(preds, 2)
    input_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

def predict_from_dict(input_dict):
    model, features = load_model()
    input_df = pd.DataFrame([input_dict])
    input_df_processed = preprocess_input(input_df, features)
    pred = model.predict(input_df_processed)
    return round(pred, 2)

if __name__ == '__main__':
    # Example usage:

    # CSV mode
    # predict_from_file("new_data.csv")

    # Dictionary mode
    example_input = {
        "cement": 540.0,
        "slag": 0.0,
        "fly_ash": 0.0,
        "water": 162.0,
        "superplasticizer": 2.5,
        "coarse_aggregate": 1040.0,
        "fine_aggregate": 676.0,
        "age": 28
    }
    prediction = predict_from_dict(example_input)
    print(f"Predicted Compressive Strength: {prediction} MPa")