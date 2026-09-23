# Crop Yield Prediction

A machine learning desktop application that predicts crop yields across Indian states and districts using historical climate, soil nutrient data, and past harvest records.

The application allows users to select a region, crop, and season to estimate yields using either **Random Forest** or **Ridge Regression**, compare model performance side-by-side, inspect visual analytics, and export advisory reports to PDF or Word.

---

## Key Features

- **Dual Model Support**:
  - **Random Forest**: Primary non-linear model capturing complex relationships between soil chemistry, weather, and yield trends.
  - **Ridge Regression (L2)**: Regularized linear model used as an interpretable baseline.
  - **Model Comparison**: View predictions, error margins ($R^2$, RMSE, MAE), and consensus estimates side-by-side.

- **Desktop Interface**:
  - Built using Python and Tkinter with high-DPI display support.
  - Cascading dropdowns (`State` &rarr; `District` &rarr; `Crop` &rarr; `Season`) that automatically filter available options.
  - Instant unit conversion between **Tonnes / Hectare**, **Quintals / Acre**, and **Kilograms / Hectare**.

- **Visual Analytics & Diagnostics**:
  - **Yield Over Time**: Historical harvest trajectory plotted alongside the upcoming model forecast.
  - **Regional Distribution**: Forecast benchmarked against the crop's historical yield distribution in the region.
  - **Prediction Residuals**: Error margins ($Actual - Fitted$) across past harvest seasons.
  - **Productivity Tiers**: Donut chart breaking down historical yields into High, Moderate, and Low production tiers.

- **Report Export**:
  - Export comprehensive reports directly to **PDF** (via ReportLab) or **Microsoft Word (.docx)**.
  - Includes input parameters, baseline soil and climate averages, historical records, and embedded high-resolution analytics plots.

---

## Project Structure

```
Crop-Yield-Prediction/
├── final_gui.py                            # Main application window and user interface
├── detailed_view.py                        # Analytics dashboard, data tables, and Matplotlib charts
├── report_exporter.py                      # PDF and Word document exporter
├── crop_yield_prediction.py                # Core prediction engine and model loader
├── training.py                             # Script to train and serialize models
├── final_merged_and_cleaned_dataset_3.csv  # Agricultural, soil, and climate dataset
├── requirements.txt                        # Python dependencies
└── models/
    ├── random_forest/                      # Trained Random Forest models and metrics
    └── ridge/                              # Trained Ridge Regression models and metrics
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9 or newer
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/devexyte/Crop-Yield-Prediction.git
cd Crop-Yield-Prediction
```

### 3. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Launch the Application
```bash
python final_gui.py
```

1. Select **State**, **District**, **Crop**, and **Season** using the dropdowns.
2. Choose your preferred model (**Random Forest** or **Ridge Regression**).
3. Click **Predict Yield** (or press `Enter`).
4. Click **Detailed Analytics** (or press `F1`) to explore charts and export reports.
5. Click **Compare Models** (or press `Shift + Enter`) to view a side-by-side performance breakdown.

### Keyboard Shortcuts
| Shortcut | Action |
| :--- | :--- |
| `Enter` | Run prediction with selected model |
| `Shift + Enter` | Open model comparison dialog |
| `F1` or `Ctrl + D` | Open detailed analytics and charts |
| `Escape` | Reset selection form |

---

## Model Training

Pre-trained models for all supported crops are included in the `models/` directory. If you want to retrain models on updated data:

```bash
# Train both Random Forest and Ridge models for all crops
python training.py --model both

# Train only Random Forest models
python training.py --model random_forest

# Train models for a specific crop (e.g. Wheat)
python training.py --crop "Wheat" --model both
```

Trained models and their evaluation metrics (`.pkl`) will be saved into `models/random_forest/` and `models/ridge/`.

---

## License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for more details.
