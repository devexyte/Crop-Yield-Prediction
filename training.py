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

    normalized_type = "ridge" if model_type in ["linear", "ridge"] else "random_forest"

    if normalized_type == "ridge":
        from sklearn.linear_model import RidgeCV
        model = RidgeCV(alphas=[0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    elif normalized_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=350,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features=0.7,
            bootstrap=True,
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

    # Maintain legacy folders (rf / linear) for full backward compatibility
    legacy_folder_name = "linear" if normalized_type == "ridge" else "rf"
    legacy_dir = os.path.join(os.path.dirname(out_dir), legacy_folder_name)
    os.makedirs(legacy_dir, exist_ok=True)
    joblib.dump(model, os.path.join(legacy_dir, f"{crop_slug}_model.pkl"))
    joblib.dump(metrics, os.path.join(legacy_dir, f"{crop_slug}_metrics.pkl"))

    type_label = "RIDGE" if normalized_type == "ridge" else "RANDOM_FOREST"
    print(f"  [{type_label}] Saved model: {model_file}")
    print(f"  [{type_label}] Saved metrics: {metrics_file} (R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Train Crop Yield Prediction Models (Random Forest Regressor & Ridge Regression)"
    )
    parser.add_argument(
        "--model",
        choices=["linear", "ridge", "rf", "random_forest", "both", "all"],
        default="both",
        help="Model architecture: 'random_forest'/'rf', 'ridge'/'linear', or 'both' (default: both)"
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

    # Categorize models into dedicated folders inside 'models/'
    selected_categories = []
    if args.model in ["rf", "random_forest", "both", "all"]:
        selected_categories.append("random_forest")
    if args.model in ["linear", "ridge", "both", "all"]:
        selected_categories.append("ridge")

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

    print(f"Found {len(crop_list)} crops. Training targets: {', '.join(selected_categories)}")
    print(f"Target model directories: {', '.join([os.path.join('models', cat) for cat in selected_categories])}")

    for crop in crop_list:
        crop_data = data[(data["crop_name"] == crop) & (data["yield"] > 0)].copy()

        if len(crop_data) < 50:
            print(f"Skipping {crop} (insufficient records: {len(crop_data)} < 50)")
            continue

        # Sanitize extreme typo outliers (> 99.5th percentile when severely disconnected from 99th)
        q99 = crop_data["yield"].quantile(0.99)
        q995 = crop_data["yield"].quantile(0.995)
        max_y = crop_data["yield"].max()
        if max_y > 2.5 * q99 and max_y > 10.0:
            trimmed_count = int((crop_data["yield"] > q995).sum())
            print(f"  [Sanitization] Trimming {trimmed_count} extreme outlier typo entries (> {q995:.2f} t/ha, max was {max_y:.2f})")
            crop_data = crop_data[crop_data["yield"] <= q995].copy()

        print(f"\nTraining models for: {crop} ({len(crop_data)} records)")

        # Retain 'year' to capture technological and agronomic progress across decades
        drop_cols = ["yield", "yield_unit", "district_code", "crop_code"]
        X = crop_data.drop(columns=[c for c in drop_cols if c in crop_data.columns])
        y = crop_data["yield"]

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for category in selected_categories:
            out_dir = os.path.join(base_dir, "models", category)
            train_single_model(category, crop, X_train, X_test, y_train, y_test, out_dir)

    print("\nModel training pipeline complete.")


if __name__ == "__main__":
    main()
