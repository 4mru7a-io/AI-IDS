"""
ai_ids.py
Training and utility script for AI-Based Network Intrusion Detection System (IDS).

Usage:
- Place NSL-KDD CSV file named "nsl_kdd.csv" in the same folder, or change the path.
- Run training:
    python ai_ids.py --train
- Export model (model.pkl) will be saved to disk.
- Example prediction (single sample) is provided in the predict_sample() function.

This script uses scikit-learn, pandas, and joblib.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

MODEL_FILE = "model.pkl"

def load_dataset(path="nsl_kdd.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Download NSL-KDD and save as nsl_kdd.csv")
    df = pd.read_csv(path)
    return df

def preprocess(df, scale=True):
    df = df.copy()
    # Basic cleaning: drop rows with NA
    df.dropna(inplace=True)
    # Convert label to binary: 'normal' -> 0, others -> 1
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 0 if str(x).lower().strip() in ['normal', 'normal.'] else 1)
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        le_dict[c] = le
    X = df.drop(['label'], axis=1)
    y = df['label']
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, y, le_dict, scaler

def train(path="nsl_kdd.csv", test_size=0.2, random_state=42, dump_model=True):
    df = load_dataset(path)
    X, y, le_dict, scaler = preprocess(df, scale=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    print("Training model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
    if dump_model:
        joblib.dump({
            'model': model,
            'label_encoders': le_dict,
            'scaler': scaler
        }, MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")
    return model, le_dict, scaler

def predict_sample(sample: dict, model_bundle_path=MODEL_FILE):
    \"\"\"Predict a single sample given as a dict {feature: value}.
    This function expects the same features (columns) as the training CSV (except 'label').
    \"\"\"
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError("Model not found. Train first and save model as model.pkl")
    bundle = joblib.load(model_bundle_path)
    model = bundle['model']
    scaler = bundle.get('scaler', None)
    # Note: This function assumes categorical features have been encoded using saved label encoders.
    # For a robust deployment, you'll need to transform categorical features using the saved label encoders.
    df = pd.DataFrame([sample])
    # Simple numeric-only approach: fill missing columns with 0
    for c in model.feature_names_in_:
        if c not in df.columns:
            df[c] = 0
    df = df[model.feature_names_in_]
    X = df.values
    if scaler is not None:
        X = scaler.transform(X)
    pred = model.predict(X)
    return int(pred[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI IDS utilities")
    parser.add_argument("--train", action="store_true", help="Train model (requires nsl_kdd.csv)")
    parser.add_argument("--predict-sample", action="store_true", help="Run a sample prediction (requires model.pkl)")
    args = parser.parse_args()
    if args.train:
        train()
    elif args.predict_sample:
        # Example: minimal sample. Replace with real feature values matching dataset.
        sample = {}
        try:
            r = predict_sample(sample)
            print("Prediction:", r)
        except Exception as e:
            print("Error during prediction:", e)
    else:
        print("No arguments supplied. Use --train to train or --predict-sample to test model.")