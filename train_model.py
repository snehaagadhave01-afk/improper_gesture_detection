"""
train_model.py
---------------
Loads the CSV produced by collect_data.py, trains a RandomForest
classifier on the hand-landmark coordinates, and saves the trained
model + label encoder to the model/ folder.

Usage:
    python train_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


def main():
    if not os.path.isfile(config.DATA_CSV):
        print(f"No data found at {config.DATA_CSV}. Run collect_data.py first.")
        return

    df = pd.read_csv(config.DATA_CSV)
    print(f"Loaded {len(df)} samples across labels: {df['label'].unique().tolist()}")

    if df["label"].nunique() < 2:
        print("Need at least 2 different labels to train a classifier.")
        return

    X = df.drop("label", axis=1).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.2%}\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(encoder, config.LABEL_ENCODER_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Label encoder saved to {config.LABEL_ENCODER_PATH}")
    print("\nSet USE_ML_MODEL = True in config.py to use it in main.py.")


if __name__ == "__main__":
    main()