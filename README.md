# Nairobi House Price Prediction

This project builds a simple end-to-end workflow for predicting Nairobi house prices using a small dataset of property listings. It demonstrates data cleaning, feature engineering, exploratory data analysis (EDA), baseline modeling, and comparison of multiple regression models.

The focus of this project is workflow structure and practical implementation rather than dataset size or production-level performance.

---

PROJECT OBJECTIVES

- Clean and standardize raw property listing data
- Engineer useful predictive features
- Train baseline and improved regression models
- Compare models using proper evaluation metrics
- Structure the project like a real data workflow

---

PROJECT STRUCTURE

Nairobi-House-Price-Prediction/

data/

- raw_listings.csv (original raw dataset)
- clean_listings.csv (cleaned + engineered dataset)

notebooks/

- 03_eda_baseline.ipynb (EDA + baseline linear regression)

src/

- clean_features.py (data cleaning + feature engineering script)
- train_compare_models.py (model training + comparison script)

reports/ (optional outputs)
requirements.txt
README.md

---

DATA FEATURES

Core Columns:

- location
- property_type
- bedrooms
- bathrooms
- size_sqm
- amenities
- price_kes

Engineered Features:

- amenity_score (number of listed amenities)
- price_per_sqm (price divided by size)
- month (extracted from listing_date if available)

Target Variable:

- price_kes

---

HOW TO RUN THE PROJECT

1. Create a virtual environment (recommended)

python -m venv .venv
source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Run data cleaning + feature engineering

python src/clean_features.py

This generates:
data/clean_listings.csv

4. Train and compare regression models

python src/train_compare_models.py

Models Compared:

- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation Metrics:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Explained Variance Score)

Example Output:

Model Comparison (lower MAE/RMSE is better):

Model MAE RMSE R2
RandomForest 3.22e+06 6.75e+06 0.967
GradientBoosting 3.99e+06 7.33e+06 0.961
LinearRegression 5.10e+06 7.87e+06 0.955

---

KEY TAKEAWAYS

- Random Forest performed best on this small dataset.
- High R² may indicate overfitting due to small sample size.
- The main value of this project is the structured workflow:
  cleaning → feature engineering → modeling → evaluation.

---

LIMITATIONS

- Dataset is small (demo scale).
- Results may not generalize to larger real-world datasets.
- No cross-validation implemented (can be added as improvement).

---

POSSIBLE IMPROVEMENTS

- Add cross-validation
- Save best model using joblib
- Increase dataset size
- Add feature importance visualization
- Deploy as simple prediction API

---

Author:
Allan Wamai
