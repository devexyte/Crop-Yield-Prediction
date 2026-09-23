import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def clean_name(name):
    """Normalize crop name for consistent file paths."""
    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace(",", "")
    )


def train_single_model(model_type, crop, X_train, X_test, y_train, y_test, out_dir):
    """Fit, evaluate, and serialize a regression model for a specific crop."""
    os.makedirs(out_dir, exist_ok=True)

    if model_type in ["linear", "ridge"]:
        model = Ridge(alpha=1.0)
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    crop_slug = clean_name(crop)
    model_file = os.path.join(out_dir, f"{crop_slug}_model.pkl")
    metrics_file = os.path.join(out_dir, f"{crop_slug}_metrics.pkl")

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

    type_label = "RIDGE" if model_type in ["linear", "ridge"] else "RF"
    print(f"  [{type_label}] Saved model: {model_file}")
    print(f"  [{type_label}] Saved metrics: {metrics_file} (R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Train Crop Yield Prediction Models (Random Forest Regressor & Ridge Regression)"
    )
    parser.add_argument(
        "--model",
        choices=["linear", "ridge", "rf", "both", "all"],
        default="both",
        help="Model architecture: 'rf', 'ridge' (linear L2), or 'both' (default: both)"
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Optional single crop name filter (e.g., 'Wheat')"
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(base_dir, "final_merged_and_cleaned_dataset_3.csv")

    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Path to cleaned dataset CSV"
    )

    args = parser.parse_args()

    selected_models = []
    if args.model in ["linear", "ridge", "both", "all"]:
        selected_models.append("linear")
    if args.model in ["rf", "both", "all"]:
        selected_models.append("rf")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found at: {args.dataset}")

    print(f"Loading dataset: {args.dataset}")
    data = pd.read_csv(args.dataset)

    # Impute missing values
    for col in data.columns:
        if data[col].dtype in ["int64", "float64"]:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])

    if args.crop:
        matching_crops = [c for c in data["crop_name"].unique() if args.crop.lower() in c.lower()]
        if not matching_crops:
            print(f"No crop matching '{args.crop}' found in dataset.")
            return
        crop_list = sorted(matching_crops)
    else:
        crop_list = sorted(data["crop_name"].unique())

    print(f"Found {len(crop_list)} crops. Training targets: {', '.join(selected_models)}")

    for crop in crop_list:
        crop_data = data[(data["crop_name"] == crop) & (data["yield"] > 0)].copy()

        if len(crop_data) < 50:
            print(f"Skipping {crop} (insufficient records: {len(crop_data)} < 50)")
            continue

        print(f"\nTraining models for: {crop} ({len(crop_data)} records)")

        drop_cols = ["yield", "yield_unit", "year", "district_code", "crop_code"]
        X = crop_data.drop(columns=[c for c in drop_cols if c in crop_data.columns])
        y = crop_data["yield"]

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for m in selected_models:
            out_dir = os.path.join(base_dir, "models", m)
            train_single_model(m, crop, X_train, X_test, y_train, y_test, out_dir)

    print("\nModel training pipeline complete.")


if __name__ == "__main__":
    main()
