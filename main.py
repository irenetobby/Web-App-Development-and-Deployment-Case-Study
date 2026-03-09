"""
Alcohol Consumption Predictor - Main Application
A machine learning web app to predict total alcohol consumption based on beer, spirit, and wine servings.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Alcohol Consumption Predictor",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'df' not in st.session_state:
    st.session_state.df = None

# Create data directory if it doesn't exist
Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# Sample data creation function
@st.cache_data
def create_sample_data():
    """Create sample beer servings dataset"""
    data = {
        'country': ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 
                    'Australia', 'Austria', 'Bahamas', 'Bangladesh', 'Belarus', 'Belgium', 'Brazil',
                    'Canada', 'China', 'Cuba', 'Czech Republic', 'Denmark', 'Egypt', 'Finland', 
                    'France', 'Germany', 'Ghana', 'Greece', 'Hungary', 'Iceland', 'India', 
                    'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan',
                    'Kenya', 'Lebanon', 'Mexico', 'Morocco', 'Netherlands', 'New Zealand', 'Nigeria',
                    'Norway', 'Pakistan', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania',
                    'Russia', 'Saudi Arabia', 'South Africa', 'South Korea', 'Spain', 'Sweden',
                    'Switzerland', 'Thailand', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom',
                    'United States', 'Uruguay', 'Vatican City', 'Venezuela', 'Vietnam', 'Yemen',
                    'Zambia', 'Zimbabwe'],
        'beer_servings': [0, 89, 25, 245, 217, 193, 21, 261, 279, 122, 0, 142, 295, 245, 240, 79,
                          93, 361, 224, 6, 263, 277, 346, 31, 133, 234, 177, 9, 5, 0, 9, 313, 63,
                          185, 82, 202, 58, 20, 238, 12, 251, 203, 42, 169, 0, 163, 71, 343, 194,
                          297, 247, 0, 225, 141, 284, 152, 233, 99, 51, 45, 206, 219, 249, 115, 0,
                          333, 111, 6, 32, 64],
        'spirit_servings': [0, 132, 0, 138, 57, 25, 179, 72, 75, 176, 0, 373, 85, 145, 122, 192,
                            137, 170, 110, 4, 133, 81, 117, 3, 112, 215, 61, 4, 1, 0, 0, 118, 76,
                            42, 97, 202, 16, 54, 115, 6, 88, 83, 2, 71, 0, 110, 114, 215, 137, 222,
                            326, 0, 78, 16, 157, 60, 100, 258, 22, 9, 237, 126, 158, 35, 0, 100, 2,
                            0, 11, 18],
        'wine_servings': [0, 54, 14, 312, 45, 221, 11, 212, 191, 51, 0, 42, 212, 16, 100, 8, 5,
                          134, 278, 0, 97, 370, 175, 9, 218, 185, 78, 0, 0, 0, 0, 168, 9, 238, 9,
                          16, 1, 16, 28, 10, 190, 175, 2, 73, 0, 21, 5, 56, 339, 168, 79, 0, 81,
                          9, 212, 186, 280, 2, 7, 1, 45, 205, 84, 78, 0, 3, 1, 0, 2, 4],
        'continent': ['Asia', 'Europe', 'Africa', 'Europe', 'Africa', 'Americas', 'Asia', 'Oceania',
                      'Europe', 'Americas', 'Asia', 'Europe', 'Europe', 'Americas', 'Americas',
                      'Asia', 'Americas', 'Europe', 'Europe', 'Africa', 'Europe', 'Europe',
                      'Europe', 'Africa', 'Europe', 'Europe', 'Europe', 'Asia', 'Asia', 'Asia',
                      'Asia', 'Europe', 'Asia', 'Europe', 'Americas', 'Asia', 'Africa', 'Asia',
                      'Americas', 'Africa', 'Europe', 'Oceania', 'Africa', 'Europe', 'Asia',
                      'Americas', 'Asia', 'Europe', 'Europe', 'Europe', 'Europe', 'Asia', 'Africa',
                      'Asia', 'Europe', 'Europe', 'Europe', 'Asia', 'Asia', 'Africa', 'Europe',
                      'Europe', 'Americas', 'Americas', 'Europe', 'Americas', 'Asia', 'Asia',
                      'Africa', 'Africa'],
        'total_litres_of_pure_alcohol': [0.0, 4.9, 0.7, 12.4, 5.9, 8.3, 3.8, 10.4, 9.7, 6.3, 0.0,
                                         14.4, 10.5, 7.2, 8.9, 4.4, 4.1, 14.1, 10.4, 0.2, 10.0,
                                         11.8, 11.3, 0.8, 8.3, 11.3, 6.6, 0.3, 0.1, 0.0, 0.2, 11.4,
                                         2.9, 7.7, 3.7, 7.0, 1.4, 1.7, 5.5, 0.5, 9.4, 9.3, 0.9,
                                         6.0, 0.0, 5.3, 4.2, 10.8, 11.0, 11.2, 11.5, 0.0, 6.4, 3.3,
                                         10.0, 7.2, 10.3, 5.8, 1.4, 1.1, 8.9, 10.4, 8.7, 4.5, 0.0,
                                         7.9, 2.2, 0.1, 0.8, 1.6]
    }
    return pd.DataFrame(data)

def load_or_create_data():
    """Load existing data or create sample data"""
    data_path = Path("data/beer-servings.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    else:
        df = create_sample_data()
        df.to_csv(data_path, index=False)
        return df

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple regression models"""
    
    # Identify categorical and numerical columns
    categorical_cols = ['country', 'continent']
    numerical_cols = ['beer_servings', 'spirit_servings', 'wine_servings']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    # Define models to try
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Hyperparameter grids
    param_grids = {
        'Ridge Regression': {'regressor__alpha': [0.01, 0.1, 1.0, 10.0]},
        'Lasso Regression': {'regressor__alpha': [0.001, 0.01, 0.1, 1.0]},
        'Random Forest': {
            'regressor__n_estimators': [50, 100],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'regressor__n_estimators': [50, 100],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 5]
        }
    }
    
    results = {}
    best_model = None
    best_score = -float('inf')
    best_model_name = ""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        if name in param_grids:
            # Perform hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline, 
                param_grids[name], 
                cv=5, 
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_pipeline = grid_search.best_estimator_
            y_train_pred = best_pipeline.predict(X_train)
            y_test_pred = best_pipeline.predict(X_test)
            
            # Calculate scores
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'best_params': grid_search.best_params_,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'pipeline': best_pipeline
            }
            
            if test_r2 > best_score:
                best_score = test_r2
                best_model = best_pipeline
                best_model_name = name
        else:
            # For models without hyperparameter tuning
            pipeline.fit(X_train, y_train)
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
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
        y_test_pred = results[name]['pipeline'].predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        
        results[name]['mse'] = mse
        results[name]['mae'] = mae
        results[name]['rmse'] = rmse
        
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("Training completed!")
    progress_bar.empty()
    
    return results, best_model, best_model_name

def main():
    # Header
    st.markdown('<h1 class="main-header">🍺 Alcohol Consumption Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict total alcohol consumption based on beer, spirit, and wine servings</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/beer.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Go to",
            ["📊 Data Explorer", "🤖 Train Model", "🔮 Make Predictions", "📈 Model Performance"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This app uses machine learning to predict total alcohol consumption "
            "based on beer, spirit, and wine servings per country."
        )
    
    # Load data
    if st.session_state.df is None:
        with st.spinner("Loading data..."):
            st.session_state.df = load_or_create_data()
    
    df = st.session_state.df
    
    # Page routing
    if page == "📊 Data Explorer":
        st.header("📊 Data Exploration")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Countries", len(df['country'].unique()))
        with col2:
            st.metric("Continents", len(df['continent'].unique()))
        with col3:
            st.metric("Avg Alcohol (L)", f"{df['total_litres_of_pure_alcohol'].mean():.2f}")
        with col4:
            st.metric("Max Alcohol (L)", f"{df['total_litres_of_pure_alcohol'].max():.2f}")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["Distribution", "By Continent", "Correlations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='total_litres_of_pure_alcohol', nbins=30,
                                  title='Distribution of Alcohol Consumption',
                                  labels={'total_litres_of_pure_alcohol': 'Total Alcohol (Liters)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot for different alcohol types
                alcohol_melted = df[['beer_servings', 'spirit_servings', 'wine_servings']].melt()
                alcohol_melted.columns = ['Type', 'Servings']
                fig = px.box(alcohol_melted, x='Type', y='Servings', color='Type',
                            title='Distribution of Alcohol Types')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                continent_avg = df.groupby('continent')['total_litres_of_pure_alcohol'].mean().reset_index()
                fig = px.bar(continent_avg, x='continent', y='total_litres_of_pure_alcohol',
                            color='continent', title='Average Consumption by Continent')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top_countries = df.nlargest(15, 'total_litres_of_pure_alcohol')
                fig = px.bar(top_countries, x='total_litres_of_pure_alcohol', y='country',
                            orientation='h', title='Top 15 Countries by Alcohol Consumption',
                            color='total_litres_of_pure_alcohol')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            corr_df = df[['beer_servings', 'spirit_servings', 'wine_servings', 'total_litres_of_pure_alcohol']].corr()
            fig = px.imshow(corr_df, text_auto=True, aspect="auto",
                          title="Correlation Matrix", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw data
        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True)
    
    elif page == "🤖 Train Model":
        st.header("🤖 Model Training")
        
        st.markdown("""
        This section will train multiple regression models and select the best one based on R² score.
        The models being trained are:
        - Linear Regression
        - Ridge Regression (with hyperparameter tuning)
        - Lasso Regression (with hyperparameter tuning)
        - Random Forest (with hyperparameter tuning)
        - Gradient Boosting (with hyperparameter tuning)
        """)
        
        # Data split configuration
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test set size (%)", 10, 30, 20) / 100
        with col2:
            random_state = st.number_input("Random seed", 1, 100, 42)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            # Prepare data
            X = df.drop('total_litres_of_pure_alcohol', axis=1)
            y = df['total_litres_of_pure_alcohol']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train models
            with st.spinner("Training models... This may take a moment."):
                results, best_model, best_model_name = train_models(X_train, X_test, y_train, y_test)
            
            # Store in session state
            st.session_state.model_results = results
            st.session_state.best_model = best_model
            st.session_state.model_trained = True
            st.session_state.best_model_name = best_model_name
            
            # Display results
            st.success(f"✅ Training completed! Best model: {best_model_name}")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Train R²': [f"{results[m]['train_r2']:.4f}" for m in results],
                'Test R²': [f"{results[m]['test_r2']:.4f}" for m in results],
                'CV R²': [f"{results[m]['cv_mean']:.4f} ± {results[m]['cv_std']:.4f}" for m in results],
                'RMSE': [f"{results[m]['rmse']:.4f}" for m in results]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Save best model
            joblib.dump(best_model, 'models/best_model.pkl')
            st.info("💾 Best model saved to 'models/best_model.pkl'")
    
    elif page == "🔮 Make Predictions":
        st.header("🔮 Make Predictions")
        
        if not st.session_state.model_trained and not Path("models/best_model.pkl").exists():
            st.warning("⚠️ No trained model found. Please go to the 'Train Model' page first.")
        else:
            # Load model if not in session
            if not st.session_state.model_trained and Path("models/best_model.pkl").exists():
                st.session_state.best_model = joblib.load('models/best_model.pkl')
                st.session_state.model_trained = True
            
            # Input form
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Numerical Inputs")
                    beer = st.number_input("Beer Servings", 0, 500, 200)
                    spirit = st.number_input("Spirit Servings", 0, 500, 100)
                    wine = st.number_input("Wine Servings", 0, 500, 100)
                
                with col2:
                    st.subheader("📍 Categorical Inputs")
                    countries = sorted(df['country'].unique())
                    continents = sorted(df['continent'].unique())
                    
                    country = st.selectbox("Country", countries)
                    continent = st.selectbox("Continent", continents)
                
                submitted = st.form_submit_button("🎯 Predict", type="primary", use_container_width=True)
            
            if submitted:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'country': [country],
                    'beer_servings': [beer],
                    'spirit_servings': [spirit],
                    'wine_servings': [wine],
                    'continent': [continent]
                })
                
                # Make prediction
                try:
                    prediction = st.session_state.best_model.predict(input_data)[0]
                    
                    # Display prediction
                    st.markdown("### Prediction Result")
                    
                    # Metric display
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        st.metric(
                            "Predicted Total Alcohol",
                            f"{prediction:.2f} Liters",
                            delta=None
                        )
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Predicted Alcohol Consumption (Liters)"},
                        gauge={
                            'axis': {'range': [0, df['total_litres_of_pure_alcohol'].max()]},
                            'bar': {'color': "#ff4b4b"},
                            'steps': [
                                {'range': [0, 5], 'color': "#98fb98"},
                                {'range': [5, 10], 'color': "#ffffe0"},
                                {'range': [10, 15], 'color': "#ffb6c1"}
                            ]
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show similar countries
                    st.markdown("### Similar Countries")
                    similar = df[
                        (df['beer_servings'].between(beer-50, beer+50)) &
                        (df['spirit_servings'].between(spirit-50, spirit+50)) &
                        (df['wine_servings'].between(wine-50, wine+50))
                    ]
                    
                    if len(similar) > 0:
                        st.dataframe(
                            similar[['country', 'beer_servings', 'spirit_servings', 
                                    'wine_servings', 'total_litres_of_pure_alcohol']],
                            use_container_width=True
                        )
                    else:
                        st.info("No directly comparable countries found.")
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    elif page == "📈 Model Performance":
        st.header("📈 Model Performance")
        
        if not st.session_state.model_results and not st.session_state.model_trained:
            st.warning("No model has been trained yet. Please go to the 'Train Model' page.")
        else:
            results = st.session_state.model_results
            
            # Model comparison chart
            fig = go.Figure()
            for name in results:
                fig.add_trace(go.Bar(
                    name=name,
                    x=['Train R²', 'Test R²', 'CV R²'],
                    y=[results[name]['train_r2'], results[name]['test_r2'], results[name]['cv_mean']],
                    error_y=dict(type='data', array=[0, 0, results[name]['cv_std']]) if name == list(results.keys())[0] else None
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                barmode='group',
                yaxis_title="R² Score",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            metrics_data = []
            for name in results:
                metrics_data.append({  
                    'Model': name,
                    'Train R²': f"{results[name]['train_r2']:.4f}",
                    'Test R²': f"{results[name]['test_r2']:.4f}",
                    'CV R² (mean)': f"{results[name]['cv_mean']:.4f}",
                    'CV R² (std)': f"{results[name]['cv_std']:.4f}",

                })
