#Creating the models (Linear Regression & Random Forest)

import os
import argparse
import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def clean_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace(",", "")
    )


def train_single_model(model_name, crop, X_train, X_test, y_train, y_test, out_dir):
    """Train, evaluate, and save a specific model type for a crop."""
    os.makedirs(out_dir, exist_ok=True)

    if model_name == "linear":
        model = Ridge(alpha=1.0)
    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    crop_clean = clean_name(crop)
    model_file = os.path.join(out_dir, f"{crop_clean}_model.pkl")
    metrics_file = os.path.join(out_dir, f"{crop_clean}_metrics.pkl")

    joblib.dump(model, model_file)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }

    joblib.dump(metrics, metrics_file)

    print(f"  [{model_name.upper()}] Saved model   : {model_file}")
    print(f"  [{model_name.upper()}] Saved metrics : {metrics_file} (R² = {r2:.4f}, MAE = {mae:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Train Crop Yield Prediction Models (Linear Regression & Random Forest)"
    )
    parser.add_argument(
        "--model",
        choices=["linear", "rf", "both", "all"],
        default="both",
        help="Type of model to train: 'linear', 'rf', or 'both'/'all' (default: both)"
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Optional: Train for a specific crop name only (e.g. 'Wheat')"
    )
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(BASE_DIR, "final_merged_and_cleaned_dataset_3.csv")

    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Path to the dataset CSV file"
    )

    args = parser.parse_args()

    selected_models = []
    if args.model in ["linear", "both", "all"]:
        selected_models.append("linear")
    if args.model in ["rf", "both", "all"]:
        selected_models.append("rf")

    print(f"Loading dataset from: {args.dataset}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset {args.dataset} not found.")

    data = pd.read_csv(args.dataset)

    # Fill missing values
    for col in data.columns:
        if data[col].dtype in ["int64", "float64"]:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])

    if args.crop:
        matching_crops = [c for c in data["crop_name"].unique() if args.crop.lower() in c.lower()]
        if not matching_crops:
            print(f"No crop matching '{args.crop}' found.")
            return
        crop_list = sorted(matching_crops)
    else:
        crop_list = sorted(data["crop_name"].unique())

    print(f"Total crops found: {len(crop_list)}")
    print(f"Training models ({', '.join(selected_models)}) for: {crop_list}")

    for crop in crop_list:
        print(f"\nTraining for crop: {crop}")

        crop_data = data[(data["crop_name"] == crop) & (data["yield"] > 0)].copy()

        if len(crop_data) < 50:
            print(f"Skipping {crop} (not enough data: {len(crop_data)} records)")
            continue

        X = crop_data.drop(columns=["yield", "yield_unit", "year", "district_code", "crop_code"])
        y = crop_data["yield"]

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for m in selected_models:
            out_dir = os.path.join(BASE_DIR, "models", m)
            train_single_model(m, crop, X_train, X_test, y_train, y_test, out_dir)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
