# Generating Figures for LaTeX Report

This directory contains a script to automatically generate all figures needed for the LaTeX report.

## Usage

### Option 1: Run the script directly

```bash
cd mlmed2026
python generate_figures.py
```

### Option 2: Run from notebook

You can also run the script cells from your Jupyter notebook if you prefer.

## Requirements

Make sure you have the following installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

## What the script does:

1. **Loads the data** from `prac1/data/` directory
2. **Preprocesses** the data (standardization)
3. **Trains or loads** the Random Forest model
   - If a trained model exists, it loads it (faster)
   - If not, it trains a new model (takes a few minutes)
4. **Generates 5 figures**:
   - `class_distribution.png` - Bar chart showing class imbalance
   - `ecg_samples.png` - Sample ECG signals for each class
   - `confusion_matrix.png` - Confusion matrix heatmap
   - `confusion_normalized.png` - Normalized confusion matrix (recall)
   - `feature_importance.png` - Top 30 most important features

## Output

All figures are saved in the `figures/` directory with 300 DPI resolution, suitable for publication.

## Note

- The script automatically detects if you're running on Kaggle or locally
- If the model is already trained and saved, it will load it instead of retraining
- All figures are saved with high resolution (300 DPI) for print quality

