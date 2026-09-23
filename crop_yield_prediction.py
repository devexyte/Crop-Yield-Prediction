import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

# Suppress minor version pickle warnings from scikit-learn
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "final_merged_and_cleaned_dataset_3.csv")

# Global caches for lazy loading and rapid sub-millisecond inference
_DATASET_CACHE = None
_MODEL_CACHE = {}


def clean_name(name):
    """Normalize crop/feature name for model file lookup."""
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


def get_dataset():
    """Lazy load and cache the cleaned dataset."""
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.strip()

        # Fill missing values: median for numeric, mode for categorical
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].astype(str).str.strip().fillna(df[col].mode()[0])

        _DATASET_CACHE = df
    return _DATASET_CACHE


def get_trained_crops():
    """Return a set of clean crop slugs that have both RF and Ridge models available."""
    rf_dirs = [
        os.path.join(BASE_DIR, "models", "random_forest"),
        os.path.join(BASE_DIR, "models", "random forest"),
        os.path.join(BASE_DIR, "models", "rf"),
    ]
    ridge_dirs = [
        os.path.join(BASE_DIR, "models", "ridge"),
        os.path.join(BASE_DIR, "models", "linear"),
    ]

    rf_models = set()
    for d in rf_dirs:
        if os.path.exists(d):
            rf_models.update(f.replace("_model.pkl", "") for f in os.listdir(d) if f.endswith("_model.pkl"))

    ridge_models = set()
    for d in ridge_dirs:
        if os.path.exists(d):
            ridge_models.update(f.replace("_model.pkl", "") for f in os.listdir(d) if f.endswith("_model.pkl"))

    return rf_models & ridge_models


def load_model_and_metrics(crop, model_type="rf"):
    """
    Retrieve trained model and metrics from memory cache or disk.
    Supported model_type values: 'rf' / 'random_forest' or 'linear' / 'ridge'.
    """
    m_str = str(model_type).lower().strip()
    if "lin" in m_str or "ridge" in m_str:
        subfolder = "ridge"
        candidate_dirs = [
            os.path.join(BASE_DIR, "models", "ridge"),
            os.path.join(BASE_DIR, "models", "linear"),
        ]
        model_display = "Ridge Regression"
    else:
        subfolder = "random_forest"
        candidate_dirs = [
            os.path.join(BASE_DIR, "models", "random_forest"),
            os.path.join(BASE_DIR, "models", "random forest"),
            os.path.join(BASE_DIR, "models", "rf"),
        ]
        model_display = "Random Forest"

    crop_file = clean_name(crop)
    cache_key = f"{subfolder}:{crop_file}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_path = None
    metrics_path = None
    for d in candidate_dirs:
        m_p = os.path.join(d, f"{crop_file}_model.pkl")
        met_p = os.path.join(d, f"{crop_file}_metrics.pkl")
        if os.path.exists(m_p) and os.path.exists(met_p):
            model_path = m_p
            metrics_path = met_p
            break

    if not model_path or not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "models", f"{crop_file}_model.pkl")
    if not metrics_path or not os.path.exists(metrics_path):
        metrics_path = os.path.join(BASE_DIR, "models", f"{crop_file}_metrics.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_display} model not found for crop: {crop}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"{model_display} metrics file not found for crop: {crop}")

    model = joblib.load(model_path)
    metrics = joblib.load(metrics_path)

    _MODEL_CACHE[cache_key] = (model, metrics, subfolder)
    return model, metrics, subfolder


def detailed_prediction(state, district, crop, season, model_type="rf"):
    """
    Generate comprehensive crop yield prediction, historical analysis,
    and validation metrics for the specified location and crop.
    """
    data = get_dataset()

    # Filter crop records with positive yield
    crop_data = data[
        (data["crop_name"].str.lower() == str(crop).strip().lower()) &
        (data["yield"] > 0)
    ].copy()

    if crop_data.empty:
        raise ValueError(f"No records found for crop '{crop}' with positive yield.")

    # Sanitize extreme typo outliers so charts and baseline summary stats are realistic
    q99 = crop_data["yield"].quantile(0.99)
    q995 = crop_data["yield"].quantile(0.995)
    max_y = crop_data["yield"].max()
    if max_y > 2.5 * q99 and max_y > 10.0:
        crop_data = crop_data[crop_data["yield"] <= q995].copy()

    # Load model & metrics (supports RF and Ridge)
    model, metrics, resolved_type = load_model_and_metrics(crop, model_type)
    expected_columns = getattr(model, "feature_names_in_", None)

    # Filter historical records matching state, district, season
    filtered = crop_data[
        (crop_data["state_name"].str.lower() == str(state).strip().lower()) &
        (crop_data["district_name"].str.lower() == str(district).strip().lower()) &
        (crop_data["season"].str.lower() == str(season).strip().lower())
    ].copy()

    if filtered.empty:
        raise ValueError(f"No historical records found for {crop} in {district}, {state} ({season} season).")

    # Select the most recent 5-year historical window and sort chronologically
    latest_year = filtered["year"].max()
    filtered = (
        filtered[filtered["year"].between(latest_year - 4, latest_year)]
        .sort_values("year")
        .reset_index(drop=True)
    )

    latest = filtered.iloc[-1]

    # Predict for each historical year using that year's recorded features
    historical_predictions = []
    historical_residuals = []

    drop_cols = ["yield", "yield_unit", "district_code", "crop_code"]
    for _, hist_row in filtered.iterrows():
        sample_df = pd.DataFrame([hist_row.to_dict()]).drop(columns=[c for c in drop_cols if c in hist_row])
        sample_encoded = pd.get_dummies(sample_df)
        if expected_columns is not None:
            sample_encoded = sample_encoded.reindex(columns=expected_columns, fill_value=0)
        pred_val = max(0.0, float(model.predict(sample_encoded)[0]))
        historical_predictions.append(pred_val)
        historical_residuals.append(float(hist_row["yield"] - pred_val))

    # Single-point validation on latest year
    actual_yield = float(latest["yield"])
    validation_prediction = historical_predictions[-1]
    prediction_error = abs(actual_yield - validation_prediction)

    # Environmental numeric features
    numeric_features = [
        "boron", "copper", "electrical_conductivity", "iron", "manganese",
        "nitrogen", "organic_carbon", "phosphorus", "potassium", "soil_ph",
        "sulphur", "zinc", "average_rainfall", "average_temperature"
    ]

    # Build representative feature profile (5-year median conditions)
    prediction_row = {}
    prediction_row["year"] = int(latest["year"]) + 1
    for feat in numeric_features:
        if feat in filtered.columns:
            prediction_row[feat] = float(filtered[feat].median())
        else:
            prediction_row[feat] = 0.0

    prediction_row["state_name"] = latest["state_name"]
    prediction_row["district_name"] = latest["district_name"]
    prediction_row["crop_name"] = latest["crop_name"]
    prediction_row["season"] = latest["season"]
    prediction_row["crop_type"] = latest.get("crop_type", "")

    prediction_df = pd.DataFrame([prediction_row])
    prediction_df = pd.get_dummies(prediction_df)
    if expected_columns is not None:
        prediction_df = prediction_df.reindex(columns=expected_columns, fill_value=0)

    predicted_yield = max(0.0, float(model.predict(prediction_df)[0]))

    # Chronologically sorted history table with genuine model predictions and residuals
    history = pd.DataFrame({
        "Year": filtered["year"].astype(int),
        "Yield": filtered["yield"].round(3),
        "Predicted": [round(p, 3) for p in historical_predictions],
        "Residual": [round(r, 3) for r in historical_residuals],
        "Rainfall (mm)": filtered["average_rainfall"].round(1),
        "Temperature (°C)": filtered["average_temperature"].round(1)
    })

    # Weather & Soil full parameter profile table
    weather_cols = [
        "year", "average_rainfall", "average_temperature",
        "nitrogen", "phosphorus", "potassium", "soil_ph", "organic_carbon",
        "zinc", "iron", "copper", "boron", "manganese", "sulphur", "yield"
    ]
    avail_weather_cols = [c for c in weather_cols if c in filtered.columns]
    weather = filtered[avail_weather_cols].copy()

    rename_map = {
        "year": "Year",
        "average_rainfall": "Rainfall (mm)",
        "average_temperature": "Temp (°C)",
        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "soil_ph": "Soil pH",
        "organic_carbon": "Organic C (%)",
        "zinc": "Zinc (Zn)",
        "iron": "Iron (Fe)",
        "copper": "Copper (Cu)",
        "boron": "Boron (B)",
        "manganese": "Manganese (Mn)",
        "sulphur": "Sulphur (S)",
        "yield": "Yield (t/ha)"
    }
    weather.rename(columns=rename_map, inplace=True)

    # Feature breakdown table
    feature_display_names = {
        "state_name": "State",
        "district_name": "District",
        "crop_name": "Crop Name",
        "season": "Cultivation Season",
        "crop_type": "Crop Category",
        "average_rainfall": "Average Rainfall (mm)",
        "average_temperature": "Average Temperature (°C)",
        "nitrogen": "Soil Nitrogen (N)",
        "phosphorus": "Soil Phosphorus (P)",
        "potassium": "Soil Potassium (K)",
        "soil_ph": "Soil pH Level",
        "organic_carbon": "Organic Carbon (%)",
        "zinc": "Zinc (Zn, ppm)",
        "iron": "Iron (Fe, ppm)",
        "copper": "Copper (Cu, ppm)",
        "boron": "Boron (B, ppm)",
        "manganese": "Manganese (Mn, ppm)",
        "sulphur": "Sulphur (S, ppm)",
        "electrical_conductivity": "Electrical Conductivity (dS/m)"
    }

    feature_rows = []
    for k, v in prediction_row.items():
        label = feature_display_names.get(k, k.replace("_", " ").title())
        val_str = f"{v:.3f}" if isinstance(v, float) else str(v)
        feature_rows.append({"Feature": label, "Representative Value": val_str})

    feature_table = pd.DataFrame(feature_rows)

    model_display_name = "Ridge Regression" if resolved_type in ["linear", "ridge"] else "Random Forest Regressor"

    summary = {
        "total_records": int(len(data)),
        "crop_records": int(len(crop_data)),
        "historical_records": int(len(filtered)),
        "training_samples": int(metrics.get("training_samples", 0)),
        "testing_samples": int(metrics.get("testing_samples", 0)),
        "yield_min": float(data["yield"].min()),
        "yield_max": float(data["yield"].max()),
        "yield_mean": float(data["yield"].mean()),
        "historical_mean": float(filtered["yield"].mean()),
        "historical_median": float(filtered["yield"].median()),
        "historical_std": float(filtered["yield"].std()) if len(filtered) > 1 else 0.0
    }

    return {
        "prediction": {
            "state": state,
            "district": district,
            "crop": crop,
            "season": season,
            "model_type": model_display_name,
            "model_key": resolved_type,
            "predicted_yield": predicted_yield,
            "actual_yield": actual_yield,
            "validation_prediction": validation_prediction,
            "prediction_error": prediction_error,
            "historical_predictions": historical_predictions,
            "historical_residuals": historical_residuals,
            "latest_year": int(latest_year)
        },
        "performance": {
            "mae": float(metrics.get("mae", 0.0)),
            "mse": float(metrics.get("mse", 0.0)),
            "rmse": float(metrics.get("rmse", 0.0)),
            "r2": float(metrics.get("r2", 0.0))
        },
        "history": history,
        "weather": weather,
        "features": feature_table,
        "summary": summary,
        "crop_yield_series": crop_data["yield"].dropna().values
    }


def predict_crop(state, district, crop, season, model_type="rf"):
    """Lightweight inference function for GUI display."""
    res = detailed_prediction(state, district, crop, season, model_type=model_type)
    return {
        "predicted_yield": res["prediction"]["predicted_yield"],
        "model_type": res["prediction"]["model_type"],
        "model_key": res["prediction"]["model_key"],
        "mae": res["performance"]["mae"],
        "mse": res["performance"]["mse"],
        "rmse": res["performance"]["rmse"],
        "r2": res["performance"]["r2"],
        "actual_yield": res["prediction"]["actual_yield"],
        "validation_prediction": res["prediction"]["validation_prediction"],
        "prediction_error": res["prediction"]["prediction_error"],
        "training_samples": res["summary"]["training_samples"],
        "testing_samples": res["summary"]["testing_samples"]
    }


def compare_models(state, district, crop, season):
    """
    Run both Random Forest and Ridge Regression simultaneously
    and return a structured comparison.
    """
    rf_res = detailed_prediction(state, district, crop, season, model_type="rf")
    lin_res = detailed_prediction(state, district, crop, season, model_type="linear")

    rf_yield = rf_res["prediction"]["predicted_yield"]
    lin_yield = lin_res["prediction"]["predicted_yield"]
    diff = abs(rf_yield - lin_yield)
    avg_yield = (rf_yield + lin_yield) / 2.0

    return {
        "rf": {
            "predicted_yield": rf_yield,
            "r2": rf_res["performance"]["r2"],
            "rmse": rf_res["performance"]["rmse"],
            "mae": rf_res["performance"]["mae"],
            "actual_yield": rf_res["prediction"]["actual_yield"],
            "validation_prediction": rf_res["prediction"]["validation_prediction"]
        },
        "ridge": {
            "predicted_yield": lin_yield,
            "r2": lin_res["performance"]["r2"],
            "rmse": lin_res["performance"]["rmse"],
            "mae": lin_res["performance"]["mae"],
            "actual_yield": lin_res["prediction"]["actual_yield"],
            "validation_prediction": lin_res["prediction"]["validation_prediction"]
        },
        "consensus_yield": avg_yield,
        "difference": diff,
        "relative_diff_pct": (diff / avg_yield * 100) if avg_yield > 0 else 0.0,
        "state": state,
        "district": district,
        "crop": crop,
        "season": season
    }


if __name__ == "__main__":
    test_result = detailed_prediction("Maharashtra", "Kolhapur", "Wheat", "Rabi", model_type="rf")
    print(f"Prediction: {test_result['prediction']['predicted_yield']:.3f} t/ha")
    print(f"Metrics (RF): R2={test_result['performance']['r2']:.3f}, RMSE={test_result['performance']['rmse']:.3f}")

    comp = compare_models("Maharashtra", "Kolhapur", "Wheat", "Rabi")
    print(f"Comparison: RF={comp['rf']['predicted_yield']:.3f} vs Ridge={comp['ridge']['predicted_yield']:.3f}")
