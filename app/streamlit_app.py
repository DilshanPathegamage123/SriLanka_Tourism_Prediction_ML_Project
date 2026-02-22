"""
streamlit_app_final_fixed.py
Fixed Streamlit app with working year selection and searchable dropdown
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Sri Lanka Tourism Predictor",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS (FIXED TITLE COLORS)
# ============================================
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e0e0e0;
        padding: 20px 10px;
    }
    
    [data-testid="stSidebar"] * {
        color: #2c3e50 !important;
        font-family: 'Arial', sans-serif;
    }
    
    /* Sidebar select boxes */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 5px 10px !important;
        min-height: 50px !important;
    }
    
    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #1e3c72 !important;
        font-size: 1.8rem !important;
        font-weight: 600;
    }
    
    /* ===== MAIN CONTENT STYLING ===== */
    /* Fix title colors */
    h1, h2, h3, h4, h5, h6 {
        color: #1e3c72 !important;
        font-weight: 600;
    }
    
    .stMarkdown h1 {
        color: #1e3c72 !important;
        font-size: 2.5rem !important;
    }
    
    .stMarkdown h2 {
        color: #1e3c72 !important;
        font-size: 1.8rem !important;
    }
    
    .stMarkdown h3 {
        color: #1e3c72 !important;
        font-size: 1.3rem !important;
    }
    
    .info-header {
        background-color: #e3f2fd;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .info-header p {
        color: #1e3c72 !important;
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
    }
    
    .confidence-box {
        background-color: #fff3cd;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #ffc107;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .confidence-box p {
        color: #856404 !important;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0;
    }
    
    .quick-info {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .quick-info p {
        color: #155724 !important;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .prediction-card h3 {
        color: #ffffff !important;
        font-size: 1.5rem;
        margin-bottom: 10px;
        opacity: 0.95;
    }
    
    .prediction-number {
        color: #ffffff !important;
        font-size: 4rem !important;
        font-weight: 700;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .prediction-date {
        color: #ffffff !important;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .footer-box {
        background-color: #1e3c72;
        padding: 20px;
        border-radius: 15px;
        margin-top: 30px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .footer-box p {
        color: #ffffff !important;
        margin: 5px 0;
        font-size: 1rem;
    }
    
    .footer-box p:first-child {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white !important;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(46, 204, 113, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(46, 204, 113, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURATION
# ============================================
MODELS_PATH = "../models"
MASTER_PATH = "../data/master"

# ============================================
# LOAD MODELS AND DATA
# ============================================
@st.cache_resource
def load_artifacts():
    """Load all trained models and data"""
    try:
        # Load Random Forest model (final chosen model)
        model = joblib.load(f"{MODELS_PATH}/random_forest_final.pkl")
        scaler = joblib.load(f"{MODELS_PATH}/scaler_final.pkl")
        features = joblib.load(f"{MODELS_PATH}/feature_list_final.pkl")
        label_encoder = joblib.load(f"{MODELS_PATH}/label_encoder_final.pkl")
        metrics = joblib.load(f"{MODELS_PATH}/performance_metrics_final.pkl")
        
        # Load historical data
        df = pd.read_csv(f"{MASTER_PATH}/tourism_master_final.csv")
        
        return model, scaler, features, label_encoder, metrics, df
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Load artifacts
model, scaler, features, label_encoder, metrics, df = load_artifacts()

if model is None:
    st.error("Failed to load models. Please check the paths.")
    st.stop()

# ============================================
# IDENTIFY FEATURE TYPES
# ============================================
binary_cols = ['Is_Top10', 'Is_Consistent_Top10', 'Is_Peak_Month', 
               'Is_COVID_Year', 'Is_Post_COVID']
categorical_cols = [col for col in features if col.startswith('Quarter_')]

cols_to_scale = [col for col in features 
                 if col not in binary_cols + categorical_cols 
                 and col not in ['Year', 'Month_Num', 'Country_Code']
                 and not col.endswith('_log')]

# Months
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

# ============================================
# HELPER FUNCTIONS (FIXED FOR YEAR SELECTION)
# ============================================
def get_quarter(month):
    """Get quarter from month name"""
    month_num = months.index(month) + 1
    if month_num <= 3:
        return 'Q1'
    elif month_num <= 6:
        return 'Q2'
    elif month_num <= 9:
        return 'Q3'
    else:
        return 'Q4'

def estimate_lag_features(country_data, feature_name, target_year, current_year=2024):
    """
    Estimate lag features for future years based on historical patterns
    """
    if feature_name in country_data.columns:
        hist_vals = country_data[feature_name].replace(0, np.nan).dropna()
        if len(hist_vals) > 0:
            # Get the last available value
            last_val = hist_vals.iloc[-1]
            
            # For future years, we need to adjust based on typical growth patterns
            if target_year > current_year:
                years_ahead = target_year - current_year
                
                # Apply different growth rates based on feature type
                if 'Lag_1_Month' in feature_name:
                    # Lag features should be roughly similar to current values
                    # We'll use the rolling mean as a proxy
                    return last_val * (1 + 0.03 * years_ahead)  # 3% annual growth
                elif 'Rolling_Mean' in feature_name:
                    # Rolling averages also grow slowly
                    return last_val * (1 + 0.03 * years_ahead)
                else:
                    return last_val * (1 + 0.05 * years_ahead)  # 5% growth for others
            else:
                return last_val
    return 0

def create_input_features(country_name, month, year):
    """
    Create feature vector for prediction (FIXED for year selection)
    """
    # Handle "All Countries" special case
    if country_name == "All Countries (Total)":
        return None, None
    
    # Get country code
    try:
        country_code = label_encoder.transform([country_name])[0]
    except:
        st.error(f"Country '{country_name}' not found in training data")
        return None, None
    
    # Get month number
    month_num = months.index(month) + 1
    
    # Get quarter
    quarter = get_quarter(month)
    
    # Create input dictionary with ALL features initialized to 0
    input_dict = {feature: 0 for feature in features}
    
    # Fill in basic features (these change with year/month)
    input_dict['Year'] = year
    input_dict['Month_Num'] = month_num
    input_dict['Month_Sin'] = np.sin(2 * np.pi * month_num / 12)
    input_dict['Month_Cos'] = np.cos(2 * np.pi * month_num / 12)
    input_dict[f'Quarter_{quarter}'] = 1
    input_dict['Country_Code'] = country_code
    
    # COVID indicators (these change with year)
    input_dict['Is_COVID_Year'] = 1 if year in [2020, 2021] else 0
    input_dict['Is_Post_COVID'] = 1 if year >= 2022 else 0
    
    if year == 2020:
        input_dict['COVID_Severity'] = 3
    elif year == 2021:
        input_dict['COVID_Severity'] = 2
    elif year == 2022:
        input_dict['COVID_Severity'] = 1
    else:
        input_dict['COVID_Severity'] = 0
    
    # Get historical data for this country
    if 'Country' in df.columns:
        country_data = df[df['Country'] == country_name]
    elif 'Country_Code' in df.columns:
        country_data = df[df['Country_Code'] == country_code]
    else:
        country_data = df
    
    # Calculate base values from historical data (2019-2024)
    if len(country_data) > 0:
        # Get the most recent complete year (2024)
        recent_data = country_data[country_data['Year'] == 2024]
        
        # Calculate typical monthly pattern from 2023-2024
        recent_years = country_data[country_data['Year'].isin([2023, 2024])]
        
        for feature in features:
            if feature in country_data.columns:
                # For features that should change with year
                if feature in ['Total_Rooms', 'Avg_Rooms_per_Province', 'Std_Rooms_per_Province']:
                    base_val = country_data[feature].median()
                    if pd.notna(base_val) and base_val > 0:
                        years_from_2024 = max(0, year - 2024)
                        input_dict[feature] = base_val * (1 + 0.02 * years_from_2024)
                
                # For visitor numbers, add growth for future years
                elif feature.startswith('Foreign_Visitors_') or feature.startswith('Total_Foreign_'):
                    base_val = country_data[feature].median()
                    if pd.notna(base_val) and base_val > 0:
                        years_from_2024 = max(0, year - 2024)
                        input_dict[feature] = base_val * (1 + 0.05 * years_from_2024)
                
                # For lag features - THESE MUST CHANGE WITH YEAR
                elif feature.startswith('Lag_') or feature.startswith('Rolling_') or feature == 'Prev_Year_Same_Month':
                    # For future years, we need to estimate based on recent patterns
                    if year > 2024:
                        # Get monthly pattern from 2023-2024
                        monthly_pattern = recent_years.groupby('Month_Num')[feature].mean().to_dict()
                        
                        if month_num in monthly_pattern and monthly_pattern[month_num] > 0:
                            base_lag = monthly_pattern[month_num]
                        else:
                            base_lag = estimate_lag_features(country_data, feature, year)
                        
                        # Apply growth factor for future years
                        years_ahead = year - 2024
                        input_dict[feature] = base_lag * (1 + 0.05 * years_ahead)
                    else:
                        # For historical years, use actual data if available
                        hist_vals = country_data[country_data['Year'] == year-1][feature].values
                        if len(hist_vals) > 0:
                            input_dict[feature] = hist_vals[0]
                        else:
                            input_dict[feature] = estimate_lag_features(country_data, feature, year)
                
                # For other numeric features
                elif feature not in binary_cols + categorical_cols and feature not in ['Year', 'Month_Num', 'Country_Code']:
                    # Use most recent value for historical, trend for future
                    if year <= 2024:
                        recent_val = recent_data[feature].values
                        if len(recent_val) > 0:
                            input_dict[feature] = recent_val[0]
                        else:
                            input_dict[feature] = country_data[feature].median()
                    else:
                        base_val = country_data[feature].median()
                        if pd.notna(base_val) and base_val > 0:
                            years_ahead = year - 2024
                            input_dict[feature] = base_val * (1 + 0.03 * years_ahead)
    
    # Add interaction features (these will change with year because COVID_Severity changes)
    input_dict['COVID_Season_Interaction'] = input_dict['COVID_Severity'] * input_dict['Month_Sin']
    input_dict['Top10_COVID_Effect'] = input_dict['Is_Top10'] * input_dict['COVID_Severity']
    
    # Create dataframe with correct feature order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[features]  # Ensure correct order
    
    return input_df, country_data

def predict_all_countries(month, year):
    """Predict total for all countries combined"""
    total_prediction = 0
    countries = sorted(label_encoder.classes_)
    
    with st.spinner(f"Calculating predictions for all {len(countries)} countries..."):
        for i, country in enumerate(countries):
            input_df, _ = create_input_features(country, month, year)
            if input_df is not None:
                # Scale
                input_scaled = input_df.copy()
                input_scaled[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
                # Predict
                pred = model.predict(input_scaled)[0]
                total_prediction += max(0, pred)
            
            # Update progress every 10 countries
            if (i + 1) % 10 == 0:
                st.write(f"Processed {i + 1}/{len(countries)} countries...")
    
    return total_prediction

def prepare_for_prediction(input_df, scaler):
    """Prepare features for prediction"""
    try:
        input_prepared = input_df.copy()
        input_prepared[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
        return input_prepared
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        return input_df

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("📊 Input Parameters")

# Year selection (updated range)
col1, col2 = st.sidebar.columns(2)
with col1:
    year = st.selectbox("Year", [2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035], index=0)  # Default to 2027
with col2:
    month = st.selectbox("Month", months, index=0)  # Default to January

# Country selection (with "All Countries" option) - this is searchable by default
if hasattr(label_encoder, 'classes_'):
    countries_list = sorted(label_encoder.classes_)
    countries = ["All Countries (Total)"] + countries_list
else:
    countries_list = sorted(df['Country'].unique()) if 'Country' in df.columns else []
    countries = ["All Countries (Total)"] + countries_list

selected_country = st.sidebar.selectbox(
    "Source Country", 
    countries,
    help="Type to search for a country"
)

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    show_explanations = st.checkbox("Show SHAP Explanations", True)
    confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)

# Model performance
if metrics:
    st.sidebar.header("📈 Model Performance")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Test MAE", f"{metrics.get('test_mae', 0):,.0f}")
    with col2:
        st.metric("Test R²", f"{metrics.get('test_r2', 0):.3f}")

# ============================================
# MAIN CONTENT
# ============================================
st.title("🏝️ Sri Lanka Foreign Tourist Arrivals Predictor")

st.markdown("""
<div class="info-header">
    <p>This application predicts foreign tourist arrivals to Sri Lanka using a Random Forest model.
    The model considers multiple factors including seasonality, source country, accommodation capacity,
    and historical patterns. You can now search for countries by typing in the dropdown.</p>
</div>
""", unsafe_allow_html=True)

# Create two columns
left_col, right_col = st.columns([2, 1])

with left_col:
    if selected_country == "All Countries (Total)":
        st.subheader(f"📈 Total Predicted Tourist Arrivals for {month} {year}")
    else:
        st.subheader(f"📈 Prediction for {selected_country} in {month} {year}")
    
    # Prediction button
    if st.button("🔮 Predict Tourist Arrivals", type="primary", use_container_width=True):
        
        with st.spinner("Calculating prediction..."):
            
            if selected_country == "All Countries (Total)":
                # Handle all countries prediction
                total_prediction = predict_all_countries(month, year)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Total Predicted Tourist Arrivals</h3>
                    <div class="prediction-number">{int(total_prediction):,}</div>
                    <div class="prediction-date">across all countries in {month} {year}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence interval
                if metrics and 'test_rmse' in metrics:
                    z_scores = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
                    z_score = z_scores.get(confidence_level, 1.96)
                    
                    rmse = metrics['test_rmse']
                    # For total, approximate error as sqrt(number of countries) * RMSE
                    num_countries = len(countries) - 1  # exclude "All"
                    approx_rmse = rmse * np.sqrt(num_countries)
                    
                    lower = max(0, int(total_prediction - z_score * approx_rmse))
                    upper = int(total_prediction + z_score * approx_rmse)
                    
                    st.markdown(f"""
                    <div class="confidence-box">
                        <p>📊 <strong>{int(confidence_level*100)}% Confidence Interval:</strong> [{lower:,}, {upper:,}]</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                # Handle single country prediction
                input_df, country_data = create_input_features(selected_country, month, year)
                
                if input_df is not None:
                    # Prepare for prediction
                    input_prepared = prepare_for_prediction(input_df, scaler)
                    
                    # Predict
                    prediction = model.predict(input_prepared)[0]
                    prediction = max(0, prediction)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Predicted Tourist Arrivals</h3>
                        <div class="prediction-number">{int(prediction):,}</div>
                        <div class="prediction-date">from {selected_country} in {month} {year}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence interval
                    if metrics and 'test_rmse' in metrics:
                        z_scores = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
                        z_score = z_scores.get(confidence_level, 1.96)
                        
                        rmse = metrics['test_rmse']
                        lower = max(0, int(prediction - z_score * rmse))
                        upper = int(prediction + z_score * rmse)
                        
                        st.markdown(f"""
                        <div class="confidence-box">
                            <p>📊 <strong>{int(confidence_level*100)}% Confidence Interval:</strong> [{lower:,}, {upper:,}]</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # SHAP Explanations
                    if show_explanations:
                        st.subheader("🔍 What Influenced This Prediction?")
                        
                        try:
                            import shap
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(input_prepared)
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            shap.waterfall_plot(
                                shap.Explanation(
                                    values=shap_values[0],
                                    base_values=explainer.expected_value,
                                    data=input_prepared.values[0],
                                    feature_names=list(input_prepared.columns)
                                ),
                                show=False,
                                max_display=10
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                        except Exception as e:
                            st.warning(f"SHAP explanation failed: {e}")
                    
                    # Historical Context
                    if len(country_data) > 0 and 'Year' in country_data.columns:
                        yearly = country_data.groupby('Year')['Tourist_Arrivals'].mean().reset_index()
                        
                        fig = px.line(yearly, x='Year', y='Tourist_Arrivals', markers=True,
                                    title=f"Historical Arrivals from {selected_country}",
                                    color_discrete_sequence=['#2ecc71'])
                        
                        fig.add_scatter(x=[year], y=[prediction], mode='markers+text',
                                      marker=dict(size=15, color='red'), name='Prediction',
                                      text=['Prediction'], textposition='top center')
                        
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='#1e3c72'),
                            xaxis=dict(
                                title_font=dict(color='#1e3c72', size=14),
                                tickfont=dict(color='#2c3e50', size=12),
                                showgrid=True,
                                gridcolor='#e0e0e0'
                            ),
                            yaxis=dict(
                                title_font=dict(color='#1e3c72', size=14),
                                tickfont=dict(color='#2c3e50', size=12),
                                showgrid=True,
                                gridcolor='#e0e0e0'
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("🌍 Quick Info")
    
    st.markdown(f"""
    <div class="quick-info">
        <p><strong>Selection:</strong> {selected_country}</p>
    </div>
    <div class="quick-info">
        <p><strong>Month:</strong> {month}</p>
    </div>
    <div class="quick-info">
        <p><strong>Year:</strong> {year}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if selected_country != "All Countries (Total)":
        # Show top features
        st.markdown("---")
        st.subheader("⚡ Key Factors")
        
        try:
            # Get feature importances
            importances = model.feature_importances_
            
            # Create dataframe with top 10
            importance_df = pd.DataFrame({
                'feature': features[:10],
                'importance': importances[:10]
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        title="Top 10 Feature Importances",
                        color='importance', color_continuous_scale='viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.write("Feature importance not available")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer-box">
    <p>Developed for Machine Learning Assignment | Data Source: Sri Lanka Tourism Dataset</p>
    <p>Model: Random Forest Regressor | Explainability: SHAP</p>
</div>
""", unsafe_allow_html=True)