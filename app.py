# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Alcohol Consumption Predictor",
    page_icon="🍺",
    layout="wide"
)

# Title and description
st.title("🍺 Alcohol Consumption Prediction App")
st.markdown("""
This app predicts **total alcohol consumption** based on different types of alcohol servings.
Enter the values below to get a prediction!
""")

# Check if model exists
model_path = 'models/best_model.pkl'
model_loaded = False

if os.path.exists(model_path):
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data.get('scaler')
        features = model_data.get('features', [])
        model_name = model_data.get('name', 'Unknown')
        model_score = model_data.get('score', 0)
        model_loaded = True
        
        st.sidebar.success(f"✅ Model loaded: {model_name}")
        st.sidebar.info(f"Model R² Score: {model_score:.4f}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
else:
    st.sidebar.warning("⚠️ No trained model found. Please run model.py first.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Predictor", "📈 Data Explorer", "ℹ️ About"])

with tab1:
    if model_loaded and features:
        st.header("Enter Input Values")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        input_values = {}
        
        # Distribute features across columns
        for i, feature in enumerate(features):
            with col1 if i % 2 == 0 else col2:
                # Clean up feature name for display
                display_name = feature.replace('_', ' ').title()
                
                # Get min/max from training data if available
                # Default to reasonable ranges
                input_values[feature] = st.number_input(
                    f"{display_name}",
                    min_value=0.0,
                    max_value=1000.0,
                    value=100.0,
                    step=10.0,
                    help=f"Enter the value for {display_name}"
                )
        
        # Prediction button
        if st.button("🔮 Predict Alcohol Consumption", type="primary", use_container_width=True):
            try:
                # Create dataframe with input values
                input_df = pd.DataFrame([input_values])
                
                # Apply scaling if needed
                if scaler:
                    input_scaled = scaler.transform(input_df[features])
                    prediction = model.predict(input_scaled)[0]
                else:
                    prediction = model.predict(input_df[features])[0]
                
                # Display prediction
                st.balloons()
                st.success("### Prediction Complete!")
                
                # Create metric display
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.metric(
                        label="Predicted Total Alcohol Consumption",
                        value=f"{prediction:.2f} litres",
                        delta=f"R²: {model_score:.3f}"
                    )
                
                # Visualization of prediction
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    title = {'text': "Litres of Pure Alcohol"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, max(20, prediction*1.5)]},
                        'bar': {'color': "#4CAF50"},
                        'steps': [
                            {'range': [0, 5], 'color': "#e8f5e8"},
                            {'range': [5, 10], 'color': "#c8e6c9"},
                            {'range': [10, 15], 'color': "#a5d6a7"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show input summary
                with st.expander("📋 View Input Values"):
                    st.json(input_values)
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.warning("Please train the model first by running model.py")
        if st.button("Run Model Training Now"):
            with st.spinner("Training models..."):
                import subprocess
                result = subprocess.run(["python", "model.py"], capture_output=True, text=True)
                st.code(result.stdout)
                st.success("Model training complete! Refresh the page to use the predictor.")

with tab2:
    st.header("Data Explorer")
    
    # Look for data file
    data_file = None
    possible_paths = ['data/beer-servings.csv', 'data/drinks.csv', 'beer-servings.csv', 'drinks.csv']
    
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if data_file:
        df = pd.read_csv(data_file)
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10))
        
        # Basic statistics
        st.subheader("Statistics")
        st.dataframe(df.describe())
        
        # Visualizations
        st.subheader("Visualizations")
        
        # Select columns for plotting
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu")
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No data file found. Please upload your dataset.")

with tab3:
    st.header("About This App")
    st.markdown("""
    ### Alcohol Consumption Prediction Model
    
    This application uses machine learning to predict total alcohol consumption based on:
    - Beer servings
    - Wine servings
    - Spirit servings
    - Other related factors
    
    ### How it works:
    1. The model is trained on historical alcohol consumption data
    2. Multiple regression algorithms are compared
    3. The best performing model is selected
    4. You can input values to get real-time predictions
    
    ### Model Performance
    The current best model achieves an R² score shown in the sidebar.
    
    ### Technologies Used
    - **Streamlit** for the web interface
    - **Scikit-learn** for machine learning
    - **Pandas/NumPy** for data processing
    - **Plotly** for visualizations
    """)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Built with Streamlit | 📊 Machine Learning Model | 🍺 Alcohol Consumption Predictor")
