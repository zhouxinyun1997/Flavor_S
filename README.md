# Soy Sauce Aroma Modeling Toolkit

## Overview

This repository contains two scripts that support data-driven analysis of sauce-aroma Soy Sauce chemical compounds.

## Files
- `ml_build_plots.py`: trains multiple regression and classification models, tunes hyper-parameters, and exports the top-performing models for each sensory target.
- `model_predictor.py`: loads the serialized models and scalers produced by the training script.

## Prerequisites
- Python 3.8+
- Required libraries: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, seaborn, matplotlib, statsmodels, openpyxl

## Quick Start
1. Install dependencies inside a virtual environment.
2. Run `python ml_build_plots.py` with your training spreadsheet (default: `Flavor_OAV_Sensory_Training_v3.xlsx`).
3. Load the saved models via `FlavorPredictor` from `model_predictor.py` for downstream prediction scripts.

## License
Copyright  2025. All rights reserved. For research and educational use only.
