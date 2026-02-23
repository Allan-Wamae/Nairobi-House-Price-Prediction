import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    df = pd.read_csv("data/clean_listings.csv")

    features = ["location", "property_type", "bedrooms", "bathrooms", "size_sqm", "amenity_score"]
    target = "price_kes"

    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["location", "property_type"]),
            ("num", "passthrough", ["bedrooms", "bathrooms", "size_sqm", "amenity_score"]),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    results = []
    best_name = None
    best_pipe = None
    best_mae = float("inf")

    for name, reg in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("reg", reg)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae, rmse, r2 = evaluate(y_test, preds)
        results.append((name, mae, rmse, r2))

        # Choose best model based on lowest MAE
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_pipe = pipe

    results_df = (
        pd.DataFrame(results, columns=["model", "MAE", "RMSE", "R2"])
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    print("\n Model Comparison (lower MAE is better):")
    print(results_df.to_string(index=False))

    # Save results CSV
    results_df.to_csv("reports/model_results.csv", index=False)
    print("\n Saved results to reports/model_results.csv")

    # Save best model
    joblib.dump(best_pipe, "models/best_model.pkl")
    print(f" Saved best model to models/best_model.pkl ({best_name})")


if __name__ == "__main__":
    main()