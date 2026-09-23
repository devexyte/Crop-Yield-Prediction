# ============================================================
# CROP YIELD PREDICTION - DETAILED REPORT
# ============================================================

import sys
import numpy as np
import pandas as pd
import joblib
import os

# Compatibility shim for NumPy 2.x pickles loaded in NumPy 1.x environments
if int(np.__version__.split(".")[0]) < 2 and "numpy._core" not in sys.modules:
    try:
        import numpy.core as _core
        sys.modules["numpy._core"] = _core
        sys.modules["numpy._core.numeric"] = _core.numeric
        sys.modules["numpy._core.multiarray"] = _core.multiarray
        import numpy.core._multiarray_umath as _umath
        sys.modules["numpy._core._multiarray_umath"] = _umath
    except Exception:
        pass
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# LOAD DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "final_merged_and_cleaned_dataset_3.csv")

data = pd.read_csv(DATASET)

# Fill missing values
for col in data.columns:

    if data[col].dtype in ["int64", "float64"]:

        data[col] = data[col].fillna(data[col].median())

    else:

        data[col] = data[col].fillna(data[col].mode()[0])


# ============================================================
# MAIN FUNCTION
# ============================================================

def detailed_prediction(state, district, crop, season, model_type="rf"):

    # Normalize model_type
    model_type = "linear" if "linear" in str(model_type).lower() else "rf"

    # --------------------------------------------------------
    # Filter selected crop
    # --------------------------------------------------------

    crop_data = data[
        (data["crop_name"].str.lower() == crop.lower()) &
        (data["yield"] > 0)
    ].copy()

    if crop_data.empty:
        raise ValueError("No records found for selected crop.")

    # --------------------------------------------------------
    # Prepare ML data
    # --------------------------------------------------------
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
    crop_file = clean_name(crop)

    # Check model directory: models/<model_type>/ first, fallback to models/
    subfolder = "linear" if model_type == "linear" else "rf"
    model_path = os.path.join(BASE_DIR, "models", subfolder, f"{crop_file}_model.pkl")
    metrics_path = os.path.join(BASE_DIR, "models", subfolder, f"{crop_file}_metrics.pkl")

    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "models", f"{crop_file}_model.pkl")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(BASE_DIR, "models", f"{crop_file}_metrics.pkl")

    if not os.path.exists(model_path):
        model_name_str = "Linear Regression" if model_type == "linear" else "Random Forest"
        raise FileNotFoundError(f"{model_name_str} model not found for {crop}")

    if not os.path.exists(metrics_path):
        model_name_str = "Linear Regression" if model_type == "linear" else "Random Forest"
        raise FileNotFoundError(f"{model_name_str} metrics file not found for {crop}")

    # Load trained model
    model = joblib.load(model_path)

    # Get feature names stored inside the model
    expected_columns = model.feature_names_in_

    # Load saved performance metrics
    metrics = joblib.load(metrics_path)

    mae = metrics["mae"]
    mse = metrics["mse"]
    rmse = metrics["rmse"]
    r2 = metrics["r2"]

    training_samples = metrics["training_samples"]
    testing_samples = metrics["testing_samples"]
    # ============================================================
    # FILTER HISTORICAL RECORDS
    # ============================================================

    filtered = crop_data[
        (crop_data["state_name"].str.lower() == state.lower()) &
        (crop_data["district_name"].str.lower() == district.lower()) &
        (crop_data["season"].str.lower() == season.lower())
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No historical records found for the selected inputs."
        )

    latest_year = filtered["year"].max()

    filtered = filtered[
        filtered["year"].between(
            latest_year - 4,
            latest_year
        )
    ]

    latest = filtered.sort_values("year").iloc[-1]

    # ============================================================
    # MODEL VALIDATION
    # ============================================================

    sample = (
        filtered
        .sort_values("year", ascending=False)
        .iloc[[0]]
        .copy()
    )

    actual_yield = sample["yield"].iloc[0]

    sample = sample.drop(columns=[
        "yield",
        "yield_unit",
        "year",
        "district_code",
        "crop_code"
    ])

    sample = pd.get_dummies(sample)

    sample = sample.reindex(
        columns=expected_columns,
        fill_value=0
    )

    predicted_sample = model.predict(sample)[0]

    prediction_error = abs(
        actual_yield - predicted_sample
    )

    # ============================================================
    # REPRESENTATIVE FEATURES
    # ============================================================

    numeric_features = [

        "boron",
        "copper",
        "electrical_conductivity",
        "iron",
        "manganese",
        "nitrogen",
        "organic_carbon",
        "phosphorus",
        "potassium",
        "soil_ph",
        "sulphur",
        "zinc",
        "average_rainfall",
        "average_temperature"

    ]

    prediction_row = {}

    for feature in numeric_features:

        prediction_row[feature] = filtered[
            feature
        ].median()

    prediction_row["state_name"] = latest["state_name"]

    prediction_row["district_name"] = latest["district_name"]

    prediction_row["crop_name"] = latest["crop_name"]

    prediction_row["season"] = latest["season"]

    prediction_row["crop_type"] = latest["crop_type"]

    prediction_df = pd.DataFrame([prediction_row])

    prediction_df = pd.get_dummies(prediction_df)

    prediction_df = prediction_df.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # ============================================================
    # FINAL PREDICTION
    # ============================================================

    predicted = model.predict(
        prediction_df
    )[0]

    # ============================================================
    # HISTORICAL DATA
    # ============================================================

    history = filtered[
        [
            "year",
            "yield",
            "average_rainfall",
            "average_temperature"
        ]
    ].copy()

    history.columns = [
        "Year",
        "Yield",
        "Rainfall (mm)",
        "Temperature (°C)"
    ]

    # ============================================================
    # WEATHER & SOIL DATA
    # ============================================================

    weather = filtered[
        [
            "year",
            "average_rainfall",
            "average_temperature",
            "nitrogen",
            "phosphorus",
            "potassium",
            "soil_ph",
            "organic_carbon",
            "zinc",
            "iron",
            "copper",
            "boron",
            "manganese",
            "sulphur",
            "yield"
        ]
    ].copy()

    weather.columns = [
        "Year",
        "Rainfall",
        "Temperature",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Soil pH",
        "Organic Carbon",
        "Zinc",
        "Iron",
        "Copper",
        "Boron",
        "Manganese",
        "Sulphur",
        "Yield"
    ]
    # ============================================================
    # DATASET SUMMARY
    # ============================================================

    summary = {

        "total_records": int(len(data)),

        "crop_records": int(len(crop_data)),

        "historical_records": int(len(filtered)),

        "training_samples": training_samples,

        "testing_samples": testing_samples,

        "yield_min": float(data["yield"].min()),

        "yield_max": float(data["yield"].max()),

        "yield_mean": float(data["yield"].mean()),

        "historical_mean": float(filtered["yield"].mean()),

        "historical_median": float(filtered["yield"].median())

    }

    # ============================================================
    # REPRESENTATIVE FEATURES TABLE
    # ============================================================

    feature_table = pd.DataFrame(
        list(prediction_row.items()),
        columns=["Feature", "Value"]
    )

    # ============================================================
    # RETURN EVERYTHING TO GUI
    # ============================================================

    return {

        "prediction": {

            "state": state,

            "district": district,

            "crop": crop,

            "season": season,

            "model_type": "Linear Regression" if model_type == "linear" else "Random Forest",

            "predicted_yield": float(predicted),

            "actual_yield": float(actual_yield),

            "validation_prediction": float(predicted_sample),

            "prediction_error": float(prediction_error)

        },

        "performance": {

            "mae": float(mae),

            "mse": float(mse),

            "rmse": float(rmse),

            "r2": float(r2)

        },

        "history": history,

        "weather": weather,

        "features": feature_table,

        "summary": summary

    }

# ============================================================
# SIMPLE PREDICTION FUNCTION FOR MAIN GUI
# ============================================================

def predict_crop(state, district, crop, season, model_type="rf"):
    """
    Lightweight wrapper for the main GUI.
    Returns the prediction and metrics for either 'linear' or 'rf'.
    """

    result = detailed_prediction(state, district, crop, season, model_type=model_type)

    return {
        "predicted_yield": result["prediction"]["predicted_yield"],
        "model_type": result["prediction"]["model_type"],
        "mae": result["performance"]["mae"],
        "mse": result["performance"]["mse"],
        "rmse": result["performance"]["rmse"],
        "r2": result["performance"]["r2"],
        "training_samples": result["summary"]["training_samples"],
        "testing_samples": result["summary"]["testing_samples"]
    }
# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    result = detailed_prediction(

        "Maharashtra",

        "Kolhapur",

        "Wheat",

        "Rabi"

    )

    print(result["prediction"])

    print(result["performance"])

    print(result["summary"])
