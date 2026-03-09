import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "data", "beer-servings.csv")

df = pd.read_csv(file_path)

# Display basic info
print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Prepare features and target
X = df.drop('total_litres_of_pure_alcohol', axis=1)
y = df['total_litres_of_pure_alcohol']

# Identify categorical and numerical columns
categorical_cols = ['country', 'continent']
numerical_cols = ['beer_servings', 'spirit_servings', 'wine_servings']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Define models to try
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Hyperparameter grids for tuning
param_grids = {
    'Ridge Regression': {'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
    'Lasso Regression': {'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'Random Forest': {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7]
    }
}

# Train and evaluate models
results = {}
best_model = None
best_score = -float('inf')
best_model_name = ""

print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    if name in param_grids:
        # Perform hyperparameter tuning
        print(f"  Performing hyperparameter tuning...")
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_pipeline = grid_search.best_estimator_
        print(f"  Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)
        
        # Calculate scores
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'best_params': grid_search.best_params_,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': best_pipeline
        }
        
        # Check if this is the best model based on test R2
        if test_r2 > best_score:
            best_score = test_r2
            best_model = best_pipeline
            best_model_name = name
            
    else:
        # For models without hyperparameter tuning (like Linear Regression)
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate scores
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        
        if test_r2 > best_score:
            best_score = test_r2
            best_model = pipeline
            best_model_name = name
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    
    results[name]['mse'] = mse
    results[name]['mae'] = mae
    results[name]['rmse'] = rmse
    
    print(f"  Train R2: {train_r2:.4f}")
    print(f"  Test R2: {test_r2:.4f}")
    print(f"  CV R2 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

# Display summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
summary_df = pd.DataFrame({
    name: {
        'Train R2': f"{results[name]['train_r2']:.4f}",
        'Test R2': f"{results[name]['test_r2']:.4f}",
        'CV R2': f"{results[name]['cv_mean']:.4f} ± {results[name]['cv_std']:.4f}",
        'RMSE': f"{results[name]['rmse']:.4f}",
        'MAE': f"{results[name]['mae']:.4f}"
    }
    for name in results
}).T

print(summary_df)

print(f"\nBest Model: {best_model_name}")
print(f"Best Test R2 Score: {best_score:.4f}")

# Save the best model
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\nBest model saved to 'models/best_model.pkl'")

# Save preprocessor for feature names info
joblib.dump(preprocessor, 'models/preprocessor.pkl')

# Create a simple feature importance analysis if applicable
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    try:
        # Extract feature names
        feature_names = (numerical_cols + 
                        list(best_model.named_steps['preprocessor']
                             .named_transformers_['cat']
                             .get_feature_names_out(categorical_cols)))
        
        # Get feature importances
        importances = best_model.named_steps['regressor'].feature_importances_
        
        # Create dataframe
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(feat_imp.head(10))
        
        # Save feature importances
        feat_imp.to_csv('models/feature_importances.csv', index=False)
    except:
        print("Could not extract feature importances")

print("\nModel training completed successfully!")

