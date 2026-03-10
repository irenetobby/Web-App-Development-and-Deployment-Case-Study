import sys
import subprocess
import time
import os

# =============================================================================
# CRITICAL: Ensure pandas is imported FIRST with error handling
# =============================================================================
print("="*60)
print("INITIALIZING MODEL TRAINING")
print("="*60)

def ensure_package(package_name, import_name=None):
    """Ensure a package is installed and imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name} {version} imported successfully")
        return module
    except ImportError:
        print(f"⚠️ {package_name} not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name} {version} installed and imported successfully")
            return module
        except Exception as e:
            print(f"❌ Failed to install {package_name}: {e}")
            sys.exit(1)

# First, ensure pandas is installed
pd = ensure_package('pandas')

# Now import everything else
np = ensure_package('numpy')
joblib = ensure_package('joblib')

# Handle scikit-learn imports separately (they're submodules)
try:
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    print("✓ scikit-learn modules imported successfully")
except ImportError as e:
    print(f"⚠️ scikit-learn import error: {e}")
    print("Installing scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    print("✓ scikit-learn installed and imported successfully")

import warnings
warnings.filterwarnings('ignore')

# Small delay to ensure all dependencies are loaded
time.sleep(1)

print("\n" + "="*60)
print("ALCOHOL CONSUMPTION PREDICTION MODEL TRAINING")
print("="*60)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# =============================================================================
# DATA LOADING WITH COMPREHENSIVE ERROR HANDLING
# =============================================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

data_path = 'data/drinks.csv'

# Check if file exists
if not os.path.exists(data_path):
    print(f"❌ Error: {data_path} not found!")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Check if data directory exists
    if os.path.exists('data'):
        print(f"Files in data directory: {os.listdir('data')}")
    else:
        print("⚠️ 'data' directory doesn't exist, creating it...")
        os.makedirs('data', exist_ok=True)
        print("Please place your drinks.csv file in the data directory")
    sys.exit(1)

# Load the data
try:
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# =============================================================================
# DATA EXPLORATION
# =============================================================================
print("\n" + "="*60)
print("DATASET INFO")
print("="*60)
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Drop Unnamed: 0 column if it exists (it's usually just an index)
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("\n✅ Dropped 'Unnamed: 0' column")

# =============================================================================
# DATA CLEANING
# =============================================================================
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Define target and feature columns
target_col = 'total_litres_of_pure_alcohol'
feature_cols = ['beer_servings', 'spirit_servings', 'wine_servings']

# Check if all required columns exist
missing_cols = []
for col in [target_col] + feature_cols:
    if col not in df.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"❌ Missing required columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

# Check for missing values in target variable
initial_rows = len(df)
target_missing = df[target_col].isnull().sum()
if target_missing > 0:
    print(f"⚠️ Found {target_missing} missing values in target column")
    # Drop rows with missing target values
    df = df.dropna(subset=[target_col])
    print(f"✅ Dropped {initial_rows - len(df)} row(s) with missing target values")

# Handle missing values in feature columns
for col in feature_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✅ Filled {missing_count} missing values in {col} with median: {median_val}")

# Verify no missing values remain
print("\nMissing values after cleaning:")
missing_after = df.isnull().sum()
print(missing_after)
if missing_after.sum() > 0:
    print("❌ Warning: Still have missing values!")
    # Drop any remaining rows with missing values
    df = df.dropna()
    print(f"✅ Dropped remaining rows with missing values, new shape: {df.shape}")
else:
    print("✅ No missing values remaining")

# =============================================================================
# PREPARE FEATURES AND TARGET
# =============================================================================
X = df[feature_cols]
y = df[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget statistics:")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# =============================================================================
# MODEL TRAINING
# =============================================================================
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Create preprocessing pipeline
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
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
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

for name, config in models.items():
    print(f"\n🔍 Training {name}...")
    
    try:
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', preprocessor),
            ('regressor', config['model'])
        ])
        
        # Check if model has hyperparameters to tune
        if config['params']:
            # Perform grid search
            print(f"   Tuning hyperparameters...")
            grid = GridSearchCV(
                pipeline, 
                config['params'], 
                cv=5, 
                scoring='r2',
                n_jobs=-1,
                verbose=0
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
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        # Track best model
        if test_r2 > best_score:
            best_score = test_r2
            best_model = pipeline
            best_name = name
            
    except Exception as e:
        print(f"   ❌ Error training {name}: {e}")
        import traceback
        traceback.print_exc()
        continue

# =============================================================================
# RESULTS AND SAVING
# =============================================================================
if best_model is None:
    print("\n❌ No models trained successfully!")
    sys.exit(1)

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)
print(f"\n🏆 Best Model: {best_name}")
print(f"   Test R²: {best_score:.4f}")

# Save the best model
model_path = 'models/best_model.pkl'
try:
    joblib.dump(best_model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Verify the model was saved
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / 1024  # KB
        print(f"   File size: {file_size:.2f} KB")
    else:
        print("   ⚠️ File not found after saving!")
        
except Exception as e:
    print(f"\n❌ Error saving model: {e}")

# Save all results for reference
try:
    results_df = pd.DataFrame({
        name: {
            'Train R²': f"{res['train_r2']:.4f}",
            'Test R²': f"{res['test_r2']:.4f}",
            'CV Mean': f"{res['cv_mean']:.4f}",
            'CV Std': f"{res['cv_std']:.4f}",
            'RMSE': f"{res['rmse']:.4f}",
            'MAE': f"{res['mae']:.4f}"
        }
        for name, res in results.items()
    }).T
    
    results_path = 'models/model_results.csv'
    results_df.to_csv(results_path)
    print(f"📊 Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(results_df.to_string())
    
except Exception as e:
    print(f"\n❌ Error saving results: {e}")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================
if 'Random Forest' in results:
    try:
        rf_model = results['Random Forest']['pipeline'].named_steps['regressor']
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            print("\n" + "="*60)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("="*60)
            print("\nRandom Forest Feature Importances:")
            for feature, importance in zip(feature_cols, importances):
                print(f"   {feature}: {importance:.4f}")
            
            # Sort features by importance
            sorted_idx = np.argsort(importances)[::-1]
            print("\n   Ranked:")
            for i, idx in enumerate(sorted_idx):
                print(f"   {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    except Exception as e:
        print(f"\n⚠️ Could not extract feature importances: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"✅ Total models trained: {len(results)}")
print(f"✅ Best model: {best_name} (R² = {best_score:.4f})")
print(f"✅ Model saved to: {model_path}")
print(f"✅ Results saved to: models/model_results.csv")
print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
