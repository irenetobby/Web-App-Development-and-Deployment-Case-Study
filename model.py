import pandas as pd
print(f"✓ Pandas imported successfully, version: {pd.__version__}")
import numpy as np
import os
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
print("ALCOHOL CONSUMPTION PREDICTION MODEL TRAINING")
print("="*60)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

data_path = 'data/drinks.csv'

# Check if file exists
if not os.path.exists(data_path):
    print(f"❌ Error: {data_path} not found!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files: {os.listdir('.')}")
    if os.path.exists('data'):
        print(f"Files in data: {os.listdir('data')}")
    sys.exit(1)

# Load the data
try:
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# =============================================================================
# DATA CLEANING
# =============================================================================
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Drop Unnamed: 0 if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("✅ Dropped 'Unnamed: 0' column")

# Define target and features
target_col = 'total_litres_of_pure_alcohol'
feature_cols = ['beer_servings', 'spirit_servings', 'wine_servings']

# Check if columns exist
for col in [target_col] + feature_cols:
    if col not in df.columns:
        print(f"❌ Column '{col}' not found!")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

# Handle missing values - THIS IS THE KEY FIX
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# First, drop rows with missing target values
initial_rows = len(df)
df = df.dropna(subset=[target_col])
print(f"✅ Dropped {initial_rows - len(df)} rows with missing target values")

# Then fill missing feature values with median
for col in feature_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✅ Filled missing {col} with median: {median_val}")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# =============================================================================
# PREPARE DATA
# =============================================================================
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

# =============================================================================
# TRAIN MODELS
# =============================================================================
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}
best_model = None
best_score = -float('inf')
best_name = ""

# 1. Linear Regression
print("\n🔍 Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_pred)
results['Linear Regression'] = test_r2
print(f"   Test R²: {test_r2:.4f}")
if test_r2 > best_score:
    best_score = test_r2
    best_model = ('Linear Regression', lr, scaler)
    best_name = 'Linear Regression'

# 2. Ridge Regression
print("\n🔍 Training Ridge Regression...")
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_pred)
results['Ridge Regression'] = test_r2
print(f"   Test R²: {test_r2:.4f}")
if test_r2 > best_score:
    best_score = test_r2
    best_model = ('Ridge Regression', ridge, scaler)
    best_name = 'Ridge Regression'

# 3. Lasso Regression
print("\n🔍 Training Lasso Regression...")
lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_pred)
results['Lasso Regression'] = test_r2
print(f"   Test R²: {test_r2:.4f}")
if test_r2 > best_score:
    best_score = test_r2
    best_model = ('Lasso Regression', lasso, scaler)
    best_name = 'Lasso Regression'

# 4. Random Forest
print("\n🔍 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # Random Forest doesn't need scaled data
y_pred = rf.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
results['Random Forest'] = test_r2
print(f"   Test R²: {test_r2:.4f}")
if test_r2 > best_score:
    best_score = test_r2
    best_model = ('Random Forest', rf, None)  # No scaler needed
    best_name = 'Random Forest'

# 5. Gradient Boosting
print("\n🔍 Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)  # Gradient Boosting doesn't need scaled data
y_pred = gb.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
results['Gradient Boosting'] = test_r2
print(f"   Test R²: {test_r2:.4f}")
if test_r2 > best_score:
    best_score = test_r2
    best_model = ('Gradient Boosting', gb, None)
    best_name = 'Gradient Boosting'

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "="*60)
print("RESULTS")
print("="*60)
for name, score in results.items():
    print(f"{name:20s}: {score:.4f}")

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)
print(f"🏆 {best_name} with R² = {best_score:.4f}")

# =============================================================================
# SAVE MODEL
# =============================================================================
model_path = 'models/best_model.pkl'
model_data = {
    'name': best_name,
    'model': best_model[1],
    'scaler': best_model[2],
    'score': best_score,
    'features': feature_cols
}

try:
    joblib.dump(model_data, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Verify save
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024
        print(f"   File size: {size:.2f} KB")
except Exception as e:
    print(f"\n❌ Error saving model: {e}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)
