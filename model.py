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

print("\nDataset Info:")
print(df.info())

print("\nFirst rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# DATA CLEANING
# ============================================================

print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Remove useless column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Handle missing values in numerical columns
numerical_cols = ["beer_servings", "spirit_servings", "wine_servings"]

for col in numerical_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled {missing} missing values in {col} with median: {median_val}")

# Handle categorical columns
categorical_cols = ["continent"]

for col in categorical_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        df[col].fillna("Unknown", inplace=True)
        print(f"Filled {missing} missing values in {col} with 'Unknown'")

# Remove rows where target is missing
df = df.dropna(subset=["total_litres_of_pure_alcohol"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ============================================================
# FEATURE / TARGET SPLIT
# ============================================================

X = df.drop(columns=["total_litres_of_pure_alcohol", "country"])
y = df["total_litres_of_pure_alcohol"]

categorical_cols = ["continent"]
numerical_cols = ["beer_servings", "spirit_servings", "wine_servings"]

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

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

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================================
# MODELS
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

print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

results = {}
best_model = None
best_score = -np.inf
best_model_name = ""

for name, model in models.items():

    print(f"\nTraining {name}...")

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

        print("Best params:", grid.best_params_)

    else:
        pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    results[name] = test_r2

    if test_r2 > best_score:
        best_score = test_r2
        best_model = pipeline
        best_model_name = name

# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)

print("Best Model:", best_model_name)
print("Best Test R2:", round(best_score, 4))

# ============================================================
# SAVE MODEL
# ============================================================

models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "best_model.pkl")

joblib.dump(best_model, model_path)

print(f"\nModel saved to: {model_path}")

print("\nModel training completed successfully!")
