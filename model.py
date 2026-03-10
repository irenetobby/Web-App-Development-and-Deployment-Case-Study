"""
Model training script for alcohol consumption prediction
"""

import sys
import subprocess
import time

# Ensure pandas is installed and imported first
try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__} imported successfully")
except ImportError as e:
    print(f"⚠️ Pandas import error: {e}")
    print("Installing pandas...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

# Now import other packages
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Small delay to ensure all dependencies are loaded
time.sleep(1)

print("\n" + "="*60)
print("ALCOHOL CONSUMPTION PREDICTION MODEL TRAINING")
print("="*60)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the data
try:
    df = pd.read_csv('data/drinks.csv')
    print(f"\n✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("\n❌ Error: data/drinks.csv not found!")
    print("Please ensure the data file exists in the correct location.")
    sys.exit(1)

# Display dataset info
print("\n" + "="*60)
print("DATASET INFO")
print("="*60)
print(f"\nShape: {df.shape}")
print(f"\nColumns:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Drop Unnamed: 0 column if it exists (it's usually just an index)
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("\n✅ Dropped 'Unnamed: 0' column")

# Data Cleaning
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Check for missing values in target variable
target_col = 'total_litres_of_pure_alcohol'
initial_rows = len(df)

# Drop rows with missing target values (only 1 row, so it's safe)
if df[target_col].isnull().sum() > 0:
    df = df.dropna(subset=[target_col])
    print(f"✅ Dropped {initial_rows - len(df)} row(s) with missing target values")

# Handle missing values in feature columns
feature_cols = ['beer_servings', 'spirit_servings', 'wine_servings']
for col in feature_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✅ Filled {df[col].isnull().sum()} missing values in {col} with median: {median_val}")

# Verify no missing values remain
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Prepare features and target
# For this problem, we'll predict total alcohol consumption from the three serving types
X = df[feature_cols]
y = df[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create preprocessing pipeline
# For this simple case, we'll just scale the features
preprocessor = StandardScaler()

# Dictionary of models to train
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}  # No hyperparameters to tune
    },
    'Ridge Regression': {
        'model': Ridge(random_state=42),
        'params': {'regressor__alpha': [0.1, 1.0, 10.0]}
    },
    'Lasso Regression': {
        'model': Lasso(random_state=42, max_iter=10000),
        'params': {'regressor__alpha': [0.001, 0.01, 0.1]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [5, 10, None]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.01, 0.1]
        }
    }
}

# Store results
results = {}
best_model = None
best_score = -float('inf')
best_name = ""

print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

for name, config in models.items():
    print(f"\n🔍 Training {name}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', preprocessor),
        ('regressor', config['model'])
    ])
    
    # Check if model has hyperparameters to tune
    if config['params']:
        # Perform grid search
        grid = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=5, 
            scoring='r2',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
        print(f"   Best params: {grid.best_params_}")
    else:
        # Train without hyperparameter tuning
        pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    
    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    
    # Store results
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'rmse': rmse,
        'mae': mae,
        'pipeline': pipeline
    }
    
    # Print results
    print(f"   Train R2: {train_r2:.4f}")
    print(f"   Test R2: {test_r2:.4f}")
    print(f"   CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    # Track best model
    if test_r2 > best_score:
        best_score = test_r2
        best_model = pipeline
        best_name = name

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)
print(f"\n🏆 Best Model: {best_name}")
print(f"   Test R2: {best_score:.4f}")

# Save the best model
model_path = 'models/best_model.pkl'
joblib.dump(best_model, model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save all results for reference
results_df = pd.DataFrame({
    name: {
        'Train R2': f"{res['train_r2']:.4f}",
        'Test R2': f"{res['test_r2']:.4f}",
        'CV Mean': f"{res['cv_mean']:.4f}",
        'CV Std': f"{res['cv_std']:.4f}",
        'RMSE': f"{res['rmse']:.4f}",
        'MAE': f"{res['mae']:.4f}"
    }
    for name, res in results.items()
}).T

results_df.to_csv('models/model_results.csv')
print(f"📊 Results saved to: models/model_results.csv")

# Feature importance analysis for tree-based models
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

if 'Random Forest' in results:
    rf_model = results['Random Forest']['pipeline'].named_steps['regressor']
    importances = rf_model.feature_importances_
    print("\nRandom Forest Feature Importances:")
    for feature, importance in zip(feature_cols, importances):
        print(f"   {feature}: {importance:.4f}")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
