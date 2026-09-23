# Crop Yield Prediction System

A desktop application that predicts seasonal crop yields using historical climate and soil data. The system supports two models: **Random Forest** and **Ridge Regression**, providing predictions alongside performance metrics and visual analytics.

---

## Features

- **Two Machine Learning Models**:
  - **Random Forest**: Captures non-linear relationships across environmental variables.
  - **Ridge Regression (L2)**: Linear regularized model for stable baseline comparisons.
  - **Side-by-Side Comparison**: Directly compare predictions and accuracy between both models.

- **Desktop Interface**:
  - Built with Python and Tkinter with high-DPI scaling.
  - Dropdown selection: State $\rightarrow$ District $\rightarrow$ Crop $\rightarrow$ Season.
  - Dropdown values cascade so only available combinations are selectable.
  - Unit conversions: Tonnes/Hectare, Quintals/Acre, and kg/Hectare.

- **Detailed Analytics Window**:
  - **Yield Over Time**: Historical recorded yield vs. model fitted values with upcoming harvest projection.
  - **Yield Distribution**: Histogram and normal density curve benchmarking the forecast against historical yields.
  - **Prediction Residuals**: Bar chart showing historical error margins ($Actual - Predicted$) for past harvest years.
  - **Yield Categories**: Pie diagram illustrating the distribution of historical harvests across productivity tiers.
  - **Data Tables & Executive Export**: Interactive tables for historical harvests, weather, soil readings, and one-click export to **Executive PDF** or **Microsoft Word (.docx)** with embedded statistical charts.

---

## Project Structure

```
Crop-Yield-Prediction/
├── final_gui.py                            # Main desktop application interface
├── detailed_view.py                        # Analytics dashboard, tables, and Matplotlib plots
├── report_exporter.py                      # Executive PDF and Microsoft Word document generator
├── crop_yield_prediction.py                # Core ML inference engine and caching layer
├── training.py                             # Offline model training pipeline
├── final_merged_and_cleaned_dataset_3.csv  # Cleaned national agricultural dataset
├── requirements.txt                        # Python package dependencies
└── models/
    ├── rf/                                 # Trained Random Forest models & evaluation metrics
    └── linear/                             # Trained Ridge Regression models & evaluation metrics
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9 or higher (tested up to Python 3.14)
- Git

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Launch GUI
```bash
python final_gui.py
```

### Keyboard Shortcuts
- **Enter**: Run prediction with the selected model
- **Shift + Enter**: Open dual-model comparison dialog
- **F1** or **Ctrl + D**: Open detailed analytics report & graphs
- **Esc**: Reset parameter form

---

## Retraining Models (Optional)

To train or update models from the dataset:

```bash
# Train both Random Forest and Ridge models for all crops
python training.py --model both

# Train for a specific crop (e.g. Wheat)
python training.py --crop "Wheat" --model both
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
