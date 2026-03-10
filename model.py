"""
Model training script for alcohol consumption prediction
FINAL VERSION - Ready for deployment
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("🍺 ALCOHOL CONSUMPTION PREDICTION MODEL TRAINING")
print("="*60)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# =============================================================================
# FIND AND LOAD DATA FILE
# =============================================================================
print("\n" + "="*60)
print("📂 LOCATING DATA FILE")
print("="*60)

# Look for the data file in common locations
data_path = None
possible_files = [
    'data/beer-servings.csv',  # From your logs
    'data/drinks.csv',
    'beer-servings.csv',
    'drinks.csv',
    './data/beer-servings.csv',
    './data/drinks.csv'
]

for file_path in possible_files:
    if os.path.exists(file_path):
        data_path = file_path
        print(f"✅ Found data file: {file_path}")
        break

if data_path is None:
    print(f"❌ Error: Could not find any data file!")
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"\nFiles in current directory:")
    for f in os.listdir('.'):
        print(f"  - {f}")
    
    if os.path.exists('data'):
        print(f"\nFiles in data directory:")
        for f in os.listdir('data'):
            print(f"  - {f}")
    sys.exit(1)

# =============================================================================
# LOAD THE DATA
# =============================================================================
print("\n" + "="*60)
print("📊 LOADING DATA")
print("="*60)

try:
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded successfully!")
    print(f"   - Rows: {df.shape[0]:,}")
    print(f"   - Columns: {df.shape[1]}")
    print(f"   - File size: {os.path.getsize(data_path) / 1024:.1f} KB")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# =============================================================================
# DATA EXPLORATION
# =============================================================================
print("\n" + "="*60)
print("🔍 DATA EXPLORATION")
print("="*60)

print(f"\nColumn names and types:")
for col in df.columns:
    dtype = df[col].dtype
    nulls = df[col].isnull().sum()
    print(f"  - {col}: {dtype} ({nulls} missing)")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nBasic statistics:")
print(df.describe())

# =============================================================================
# IDENTIFY TARGET AND FEATURE COLUMNS
# =============================================================================
print("\n" + "="*60)
print("🎯 IDENTIFYING TARGET COLUMN")
print("="*60)

# Based on the dataset, we know the target is total_litres_of_pure_alcohol
# But let's make it robust
target_col = None
if 'total_litres_of_pure_alcohol' in df.columns:
    target_col = 'total_litres_of_pure_alcohol'
    print(f"✅ Found target column: {target_col}")
else:
    # Look for alcohol-related columns
    alcohol_keywords = ['alcohol', 'total_litres', 'litres', 'pure_alcohol']
    for col in df.columns:
        col_lower = col.lower()
        for keyword in alcohol_keywords:
            if keyword in col_lower:
                target_col = col
                print(f"✅ Found target column: {target_col}")
                break
        if target_col:
            break

if target_col is None:
    # If no obvious target column, use the last float column
    float_cols = df.select_dtypes(include=['float64']).columns.tolist()
    if float_cols:
        target_col = float_cols[-1]
        print(f"⚠️ Using {target_col} as target column")
    else:
        print(f"❌ Could not identify target column!")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

# Identify feature columns (numeric columns except target)
feature_cols = []
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if col != target_col and 'unnamed' not in col.lower():
        feature_cols.append(col)

print(f"\n🔧 Feature columns: {feature_cols}")

if not feature_cols:
    print(f"❌ No feature columns found!")
    sys.exit(1)

# =============================================================================
# DATA CLEANING
# =============================================================================
print("\n" + "="*60)
print("🧹 DATA CLEANING")
print("="*60)

print(f"\nMissing values before cleaning:")
print(df[feature_cols + [target_col]].isnull().sum())

# Drop rows with missing target values (CRITICAL STEP)
initial_rows = len(df)
df = df.dropna(subset=[target_col])
rows_dropped = initial_rows - len(df)
if rows_dropped > 0:
    print(f"✅ Dropped {rows_dropped} row(s) with missing target values")

# Fill missing feature values with median
for col in feature_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✅ Filled {missing_count} missing values in {col} with median: {median_val:.2f}")

print(f"\nMissing values after cleaning:")
print(df[feature_cols + [target_col]].isnull().sum())

# Final check - ensure no NaN values remain
if df[target_col].isnull().sum() > 0:
    print(f"❌ Target column still has NaN values! This should not happen.")
    sys.exit(1)

# =============================================================================
# PREPARE DATA FOR MODELING
# =============================================================================
print("\n" + "="*60)
print("📈 PREPARING DATA FOR MODELING")
print("="*60)

X = df[feature_cols]
y = df[target_col]

print(f"\nFeatures matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

print(f"\nTarget statistics:")
print(f"  - Mean: {y.mean():.2f}")
print(f"  - Std:  {y.std():.2f}")
print(f"  - Min:  {y.min():.2f}")
print(f"  - Max:  {y.max():.2f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples")

# =============================================================================
# TRAIN MODELS
# =============================================================================
print("\n" + "="*60)
print("🤖 TRAINING MODELS")
print("="*60)

# Scale features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}
best_model = None
best_score = -float('inf')
best_name = ""
best_scaler = None

# 1. Linear Regression
print("\n🔍 Training Linear Regression...")
try:
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_pred)
    results['Linear Regression'] = test_r2
    print(f"   ✅ Test R²: {test_r2:.4f}")
    if test_r2 > best_score:
        best_score = test_r2
        best_model = lr
        best_scaler = scaler
        best_name = 'Linear Regression'
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Ridge Regression
print("\n🔍 Training Ridge Regression...")
try:
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_pred)
    results['Ridge Regression'] = test_r2
    print(f"   ✅ Test R²: {test_r2:.4f}")
    if test_r2 > best_score:
        best_score = test_r2
        best_model = ridge
        best_scaler = scaler
        best_name = 'Ridge Regression'
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Lasso Regression
print("\n🔍 Training Lasso Regression...")
try:
    lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_pred)
    results['Lasso Regression'] = test_r2
    print(f"   ✅ Test R²: {test_r2:.4f}")
    if test_r2 > best_score:
        best_score = test_r2
        best_model = lasso
        best_scaler = scaler
        best_name = 'Lasso Regression'
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Random Forest
print("\n🔍 Training Random Forest...")
try:
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)  # Random Forest doesn't need scaled data
    y_pred = rf.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    results['Random Forest'] = test_r2
    print(f"   ✅ Test R²: {test_r2:.4f}")
    if test_r2 > best_score:
        best_score = test_r2
        best_model = rf
        best_scaler = None
        best_name = 'Random Forest'
except Exception as e:
    print(f"   ❌ Error: {e}")

# 5. Gradient Boosting
print("\n🔍 Training Gradient Boosting...")
try:
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)  # Gradient Boosting doesn't need scaled data
    y_pred = gb.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    results['Gradient Boosting'] = test_r2
    print(f"   ✅ Test R²: {test_r2:.4f}")
    if test_r2 > best_score:
        best_score = test_r2
        best_model = gb
        best_scaler = None
        best_name = 'Gradient Boosting'
except Exception as e:
    print(f"   ❌ Error: {e}")

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
print("\n" + "="*60)
print("📊 RESULTS SUMMARY")
print("="*60)

# Sort results by score
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\nModel Performance (Test R²):")
for name, score in sorted_results:
    medal = "🥇" if score == sorted_results[0][1] else "🥈" if score == sorted_results[1][1] else "🥉" if score == sorted_results[2][1] else "  "
    print(f"  {medal} {name:20s}: {score:.4f}")

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_name}")
print(f"   Test R² Score: {best_score:.4f}")
print("="*60)

# =============================================================================
# SAVE MODEL
# =============================================================================
print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

model_data = {
    'name': best_name,
    'model': best_model,
    'scaler': best_scaler,
    'score': best_score,
    'features': feature_cols,
    'target': target_col,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

model_path = 'models/best_model.pkl'
try:
    joblib.dump(model_data, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Verify save
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024
        print(f"   File size: {size:.2f} KB")
        
        # Test load
        test_load = joblib.load(model_path)
        print(f"   ✓ Model verified (can be loaded)")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# Save results to CSV
results_df = pd.DataFrame([
    {'Model': name, 'Test R²': score}
    for name, score in sorted_results
])
results_path = 'models/model_results.csv'
try:
    results_df.to_csv(results_path, index=False)
    print(f"📊 Results saved to: {results_path}")
except Exception as e:
    print(f"❌ Error saving results: {e}")

# =============================================================================
# FEATURE IMPORTANCE (for tree-based models)
# =============================================================================
if best_name in ['Random Forest', 'Gradient Boosting'] and hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*60)
    print("📊 FEATURE IMPORTANCE")
    print("="*60)
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop features from {best_name}:")
    for i in range(min(5, len(feature_cols))):
        idx = indices[i]
        print(f"  {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nBest model: {best_name} (R² = {best_score:.4f})")
print(f"Model saved to: models/best_model.pkl")
print(f"Results saved to: models/model_results.csv")
