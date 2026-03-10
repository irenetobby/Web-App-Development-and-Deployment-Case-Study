import sys
import os
import pandas as pd
import numpy as np
import joblib
import warnings
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_DIR = 'models'
DATA_DIR = 'data'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("\n" + "="*70)
print("🍺 ALCOHOL CONSUMPTION PREDICTION MODEL TRAINING")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================
def find_and_load_data():
    """
    Find and load the dataset from common locations
    Returns: DataFrame if found, None otherwise
    """
    print("\n📂 LOCATING DATA FILE")
    print("-" * 40)
    
    # Look for the data file in common locations
    possible_files = [
        'data/beer-servings.csv',
        'data/drinks.csv',
        'beer-servings.csv',
        'drinks.csv',
        './data/beer-servings.csv',
        './data/drinks.csv',
        '../data/beer-servings.csv',
        '../data/drinks.csv'
    ]
    
    # Also look for any CSV in data directory
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith('.csv'):
                possible_files.append(os.path.join(DATA_DIR, file))
    
    data_path = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            data_path = file_path
            print(f"✅ Found data file: {file_path}")
            break
    
    if data_path is None:
        print(f"❌ Error: Could not find any data file!")
        print(f"\nCurrent working directory: {os.getcwd()}")
        
        if os.path.exists(DATA_DIR):
            print(f"\nFiles in {DATA_DIR} directory:")
            for f in os.listdir(DATA_DIR):
                print(f"  - {f}")
        else:
            print(f"\nFiles in current directory:")
            for f in os.listdir('.'):
                if f.endswith('.csv'):
                    print(f"  - {f}")
        
        return None
    
    # Load the data
    try:
        df = pd.read_csv(data_path)
        file_size = os.path.getsize(data_path) / 1024
        print(f"✅ Data loaded successfully!")
        print(f"   - Rows: {df.shape[0]:,}")
        print(f"   - Columns: {df.shape[1]}")
        print(f"   - File size: {file_size:.1f} KB")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# =============================================================================
# DATA EXPLORATION AND VALIDATION
# =============================================================================
def explore_data(df):
    """Display basic information about the dataset"""
    print("\n🔍 DATA EXPLORATION")
    print("-" * 40)
    
    print(f"\nColumn names and types:")
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        null_pct = (nulls / len(df)) * 100
        print(f"  - {col}: {dtype} | Missing: {nulls} ({null_pct:.1f}%)")
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

# =============================================================================
# IDENTIFY TARGET AND FEATURE COLUMNS
# =============================================================================
def identify_columns(df):
    """
    Identify target column and feature columns
    Returns: target_col, feature_cols
    """
    print("\n🎯 IDENTIFYING TARGET COLUMN")
    print("-" * 40)
    
    # Common target column names
    target_candidates = [
        'total_litres_of_pure_alcohol',
        'total_alcohol',
        'alcohol_consumption',
        'pure_alcohol',
        'litres_of_pure_alcohol',
        'total_litres'
    ]
    
    target_col = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target_col = candidate
            print(f"✅ Found target column: {target_col}")
            break
    
    if target_col is None:
        # Look for alcohol-related columns
        alcohol_keywords = ['alcohol', 'litres', 'pure', 'drink']
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
        float_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if float_cols:
            # Exclude ID-like columns
            for col in float_cols:
                if col.lower() not in ['id', 'index', 'unnamed:0']:
                    target_col = col
                    break
            if target_col is None:
                target_col = float_cols[-1]
            print(f"⚠️ Using '{target_col}' as target column")
        else:
            print(f"❌ Could not identify target column!")
            print(f"Available columns: {list(df.columns)}")
            return None, None
    
    # Identify feature columns (all numeric columns except target)
    feature_cols = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if col != target_col and 'unnamed' not in col.lower() and col.lower() not in ['id', 'index']:
            feature_cols.append(col)
    
    print(f"\n🔧 Feature columns ({len(feature_cols)}): {feature_cols}")
    
    if not feature_cols:
        print(f"❌ No feature columns found!")
        return None, None
    
    return target_col, feature_cols

# =============================================================================
# DATA CLEANING
# =============================================================================
def clean_data(df, target_col, feature_cols):
    """
    Clean the dataset by handling missing values
    Returns: Cleaned DataFrame
    """
    print("\n🧹 DATA CLEANING")
    print("-" * 40)
    
    initial_rows = len(df)
    
    # Check for missing values
    missing_before = df[feature_cols + [target_col]].isnull().sum()
    print(f"\nMissing values before cleaning:")
    print(missing_before[missing_before > 0] if any(missing_before > 0) else "  No missing values found")
    
    # Drop rows with missing target values (CRITICAL STEP)
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
            print(f"✅ Filled {missing_count} missing values in '{col}' with median: {median_val:.2f}")
    
    # Check for infinite values
    for col in feature_cols + [target_col]:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"⚠️ Found {inf_count} infinite values in '{col}'. Replacing with NaN...")
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            # Fill with median again
            df[col].fillna(df[col].median(), inplace=True)
    
    # Final check
    missing_after = df[feature_cols + [target_col]].isnull().sum()
    print(f"\nMissing values after cleaning:")
    print(missing_after[missing_after > 0] if any(missing_after > 0) else "  No missing values remain")
    
    # Validate target column has no NaN
    if df[target_col].isnull().sum() > 0:
        print(f"❌ Target column still has NaN values! This should not happen.")
        return None
    
    print(f"\n✅ Data cleaning complete. Final shape: {df.shape}")
    return df

# =============================================================================
# MODEL TRAINING FUNCTION
# =============================================================================
def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """
    Train multiple regression models and return the best one
    """
    print("\n🤖 TRAINING MODELS")
    print("-" * 40)
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store results
    results = {}
    models = {}
    best_model = None
    best_score = -float('inf')
    best_name = ""
    best_scaler = None
    
    # 1. Linear Regression
    print("\n📈 Linear Regression...")
    try:
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred = lr.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        results['Linear Regression'] = test_r2
        models['Linear Regression'] = lr
        print(f"   ✅ R²: {test_r2:.4f} | RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}")
        if test_r2 > best_score:
            best_score = test_r2
            best_model = lr
            best_scaler = scaler
            best_name = 'Linear Regression'
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Ridge Regression with hyperparameter tuning
    print("\n📈 Ridge Regression...")
    try:
        ridge = Ridge(random_state=RANDOM_STATE)
        ridge_params = {'alpha': [0.1, 1.0, 10.0]}
        ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2')
        ridge_grid.fit(X_train_scaled, y_train)
        best_ridge = ridge_grid.best_estimator_
        y_pred = best_ridge.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        results['Ridge Regression'] = test_r2
        models['Ridge Regression'] = best_ridge
        print(f"   ✅ R²: {test_r2:.4f} | RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}")
        print(f"      Best alpha: {ridge_grid.best_params_['alpha']}")
        if test_r2 > best_score:
            best_score = test_r2
            best_model = best_ridge
            best_scaler = scaler
            best_name = 'Ridge Regression'
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Lasso Regression with hyperparameter tuning
    print("\n📈 Lasso Regression...")
    try:
        lasso = Lasso(random_state=RANDOM_STATE, max_iter=10000)
        lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0]}
        lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='r2')
        lasso_grid.fit(X_train_scaled, y_train)
        best_lasso = lasso_grid.best_estimator_
        y_pred = best_lasso.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        results['Lasso Regression'] = test_r2
        models['Lasso Regression'] = best_lasso
        print(f"   ✅ R²: {test_r2:.4f} | RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}")
        print(f"      Best alpha: {lasso_grid.best_params_['alpha']}")
        if test_r2 > best_score:
            best_score = test_r2
            best_model = best_lasso
            best_scaler = scaler
            best_name = 'Lasso Regression'
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. Random Forest with hyperparameter tuning
    print("\n🌲 Random Forest...")
    try:
        rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
        rf_grid.fit(X_train, y_train)  # Random Forest doesn't need scaled data
        best_rf = rf_grid.best_estimator_
        y_pred = best_rf.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        results['Random Forest'] = test_r2
        models['Random Forest'] = best_rf
        print(f"   ✅ R²: {test_r2:.4f} | RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}")
        print(f"      Best params: {rf_grid.best_params_}")
        if test_r2 > best_score:
            best_score = test_r2
            best_model = best_rf
            best_scaler = None
            best_name = 'Random Forest'
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Gradient Boosting with hyperparameter tuning
    print("\n🚀 Gradient Boosting...")
    try:
        gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
        gb_params = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='r2', n_jobs=-1)
        gb_grid.fit(X_train, y_train)  # Gradient Boosting doesn't need scaled data
        best_gb = gb_grid.best_estimator_
        y_pred = best_gb.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        results['Gradient Boosting'] = test_r2
        models['Gradient Boosting'] = best_gb
        print(f"   ✅ R²: {test_r2:.4f} | RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}")
        print(f"      Best params: {gb_grid.best_params_}")
        if test_r2 > best_score:
            best_score = test_r2
            best_model = best_gb
            best_scaler = None
            best_name = 'Gradient Boosting'
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return results, models, best_model, best_scaler, best_name, best_score

# =============================================================================
# SAVE MODEL FUNCTION
# =============================================================================
def save_model(model_data, model_dir=MODEL_DIR):
    """Save the trained model and metadata"""
    print("\n💾 SAVING MODEL")
    print("-" * 40)
    
    model_path = os.path.join(model_dir, 'best_model.pkl')
    
    try:
        joblib.dump(model_data, model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Verify save
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024
            print(f"   File size: {size:.2f} KB")
            
            # Test load
            test_load = joblib.load(model_path)
            print(f"   ✓ Model verified (can be loaded successfully)")
            
            # Save model info as text for easy reading
            info_path = os.path.join(model_dir, 'model_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"Model: {model_data['name']}\n")
                f.write(f"R² Score: {model_data['score']:.4f}\n")
                f.write(f"Target: {model_data['target']}\n")
                f.write(f"Features: {', '.join(model_data['features'])}\n")
                f.write(f"Training samples: {model_data['training_samples']}\n")
                f.write(f"Test samples: {model_data['test_samples']}\n")
                f.write(f"Trained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"📄 Model info saved to: {info_path}")
            
            return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function"""
    
    # Step 1: Load data
    df = find_and_load_data()
    if df is None:
        print("\n❌ Failed to load data. Exiting...")
        return False
    
    # Step 2: Explore data
    df = explore_data(df)
    
    # Step 3: Identify columns
    target_col, feature_cols = identify_columns(df)
    if target_col is None or feature_cols is None:
        print("\n❌ Failed to identify columns. Exiting...")
        return False
    
    # Step 4: Clean data
    df = clean_data(df, target_col, feature_cols)
    if df is None:
        print("\n❌ Failed to clean data. Exiting...")
        return False
    
    # Step 5: Prepare data for modeling
    print("\n📈 PREPARING DATA FOR MODELING")
    print("-" * 40)
    
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
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")
    
    # Step 6: Train models
    results, models, best_model, best_scaler, best_name, best_score = train_models(
        X_train, X_test, y_train, y_test, feature_cols
    )
    
    # Step 7: Results summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    # Sort results by score
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nModel Performance (Test R²):")
    for i, (name, score) in enumerate(sorted_results):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"  {medal} {name:20s}: {score:.4f}")
    
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"   Test R² Score: {best_score:.4f}")
    print("="*70)
    
    # Step 8: Save model
    model_data = {
        'name': best_name,
        'model': best_model,
        'scaler': best_scaler,
        'score': best_score,
        'features': feature_cols,
        'target': target_col,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'all_results': sorted_results,
        'timestamp': datetime.now().isoformat()
    }
    
    if save_model(model_data):
        # Save results to CSV
        results_df = pd.DataFrame([
            {'Model': name, 'Test R²': score}
            for name, score in sorted_results
        ])
        results_path = os.path.join(MODEL_DIR, 'model_results.csv')
        try:
            results_df.to_csv(results_path, index=False)
            print(f"📊 Results saved to: {results_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
        
        # Feature importance for tree-based models
        if best_name in ['Random Forest', 'Gradient Boosting'] and hasattr(best_model, 'feature_importances_'):
            print("\n" + "="*70)
            print("📊 FEATURE IMPORTANCE")
            print("="*70)
            
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\nTop features from {best_name}:")
            for i in range(min(5, len(feature_cols))):
                idx = indices[i]
                print(f"  {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nBest model: {best_name} (R² = {best_score:.4f})")
    print(f"All files saved in: {MODEL_DIR}/")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
