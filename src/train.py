# Not meant to run locally. Upload to S3. Run by SageMaker.

import pandas as pd
import numpy as np
import os
import joblib
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    args = parser.parse_args()

    # Load all CSV files in training directory
    csv_files = glob(os.path.join(args.train, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the training directory.")

    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} samples from {len(csv_files)} files.")

    # Feature selection
    X = df[["sensor_1", "sensor_2", "sensor_3"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
