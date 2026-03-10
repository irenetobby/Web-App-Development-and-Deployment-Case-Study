import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATA
# ============================================================
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "data", "beer-servings.csv")

df = pd.read_csv(file_path)

# ============================================================
# DATA CLEANING
# ============================================================
# 1. Remove ID column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# 2. CRITICAL FIX: Remove rows where the TARGET is missing
# Scikit-learn cannot impute the target variable (y) during fit()
df = df.dropna(subset=["total_litres_of_pure_alcohol"])

# ============================================================
# FEATURE / TARGET SPLIT
# ============================================================
# We drop 'country' because it has too many unique values (high cardinality)
X = df.drop(columns=["total_litres_of_pure_alcohol", "country"])
y = df["total_litres_of_pure_alcohol"]

categorical_cols = ["continent"]
numerical_cols = ["beer_servings", "spirit_servings", "wine_servings"]

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
# We handle imputation INSIDE the pipeline to ensure consistency
numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ============================================================
# TRAIN TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# MODELS & HYPERPARAMETERS
# ============================================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

param_grids = {
    "Ridge Regression": {"regressor__alpha": [0.01, 0.1, 1, 10]},
    "Lasso Regression": {"regressor__alpha": [0.001, 0.01, 0.1, 1]},
    "Random Forest": {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 10, 20]
    },
    "Gradient Boosting": {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.01, 0.1],
        "regressor__max_depth": [3, 5]
    }
}

# ============================================================
# MODEL TRAINING
# ============================================================
results = {}
best_model = None
best_score = -np.inf
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")

    # Build the full pipeline for this specific model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    if name in param_grids:
        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=5,
            scoring="r2",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
    else:
        pipeline.fit(X_train, y_train)

    # Evaluation
    y_test_pred = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results[name] = test_r2
    print(f"{name} Test R2: {test_r2:.4f}")

    if test_r2 > best_score:
        best_score = test_r2
        best_model = pipeline
        best_model_name = name

# ============================================================
# SAVE BEST MODEL
# ============================================================
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))

print(f"\nSuccess! Best Model: {best_model_name} (R2: {best_score:.4f})")
