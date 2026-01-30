"""
Water Consumption Forecasting Dashboard - Streamlit App
Multi-Municipality Water Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Water Consumption Forecasting",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and encoders"""
    try:
        with open('water_consumption_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('municipality_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        with open('feature_columns.pkl', 'rb') as f:
            features = pickle.load(f)
        
        return model, encoder, features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('water_consumption_100000_rows_improved.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load model and data
model, municipality_encoder, feature_columns = load_model()
df = load_data()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def prepare_features(municipality, temperature, humidity, rainfall, day_type, df, encoder, feature_columns):
    """Prepare features for prediction"""
    # Encode municipality
    municipality_code = encoder.transform([municipality])[0]
    
    # Get municipality data
    muni_data = df[df['region_name'] == municipality]
    
    # Set day type flags
    is_weekend = 1 if day_type == 'Weekend' else 0
    is_holiday = 1 if day_type == 'Holiday' else 0
    
    # Get current month and season
    current_month = datetime.now().month
    season_map = {1: 0, 2: 0, 3: 3, 4: 3, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 3, 11: 0, 12: 0}
    current_season = season_map.get(current_month, 0)
    
    # Create feature dictionary
    features_dict = {
        'temperature_celsius': temperature,
        'humidity_percent': humidity,
        'rainfall_mm': rainfall,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'municipality_encoded': municipality_code,
        'population_scaled': muni_data['population'].mean() / 1_000_000,
        'industrial_scaled': muni_data['industrial_activity_index'].mean(),
        'prev_day_consumption_normalized': muni_data['water_consumption_liters'].mean() / 100_000_000,
        'prev_7day_avg_normalized': muni_data['water_consumption_liters'].mean() / 100_000_000,
        'consumption_variance': 1.0,
        'month': current_month,
        'season': current_season
    }
    
    # Create DataFrame with correct feature order
    X = pd.DataFrame([features_dict])[feature_columns]
    
    return X, muni_data

def make_prediction(municipality, temperature, humidity, rainfall, day_type):
    """Make water consumption prediction"""
    if model is None or df is None:
        return None, None, None
    
    X, muni_data = prepare_features(
        municipality, temperature, humidity, rainfall, day_type,
        df, municipality_encoder, feature_columns
    )
    
    # Predict
    prediction = model.predict(X)[0] / 1_000_000  # Convert to Million Liters
    average = muni_data['water_consumption_liters'].mean() / 1_000_000
    change_percent = ((prediction - average) / average) * 100
    
    return prediction, average, change_percent

def create_comparison_chart(temperature, humidity, rainfall, day_type):
    """Create comparison chart for all municipalities"""
    municipalities = sorted(df['region_name'].unique())
    predictions = []
    
    for municipality in municipalities:
        pred, avg, change = make_prediction(municipality, temperature, humidity, rainfall, day_type)
        muni_data = df[df['region_name'] == municipality]
        predictions.append({
            'Municipality': municipality,
            'Predicted (ML)': pred,
            'Average (ML)': avg,
            'Population': int(muni_data['population'].mean())
        })
    
    comp_df = pd.DataFrame(predictions).sort_values('Predicted (ML)', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comp_df['Municipality'],
        y=comp_df['Predicted (ML)'],
        name='Predicted',
        marker_color='#1E88E5',
        text=comp_df['Predicted (ML)'].round(1),
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=comp_df['Municipality'],
        y=comp_df['Average (ML)'],
        name='Average',
        marker_color='#FFA726',
        text=comp_df['Average (ML)'].round(1),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Municipality Comparison - {temperature}°C, {humidity}% Humidity",
        xaxis_title="Municipality",
        yaxis_title="Water Consumption (Million Liters)",
        barmode='group',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig, comp_df

def create_forecast(municipality, days=7):
    """Create 7-day forecast"""
    dates = pd.date_range(datetime.now(), periods=days, freq='D')
    
    # Simulated weather forecast (in production, use real weather API)
    np.random.seed(42)
    weather_forecast = [
        {
            'temp': np.random.randint(28, 38),
            'humidity': np.random.randint(55, 85),
            'rainfall': np.random.choice([0, 0, 0, 0, 2, 5, 10], p=[0.5, 0.2, 0.15, 0.05, 0.05, 0.03, 0.02])
        }
        for _ in range(days)
    ]
    
    forecast_data = []
    
    for i, date in enumerate(dates):
        weather = weather_forecast[i]
        day_type = 'Weekend' if date.dayofweek >= 5 else 'Weekday'
        
        pred, avg, change = make_prediction(
            municipality,
            weather['temp'],
            weather['humidity'],
            weather['rainfall'],
            day_type
        )
        
        forecast_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Day': date.strftime('%A'),
            'Temperature': weather['temp'],
            'Humidity': weather['humidity'],
            'Rainfall': weather['rainfall'],
            'Day Type': day_type,
            'Predicted (ML)': round(pred, 2)
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Create forecast chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted (ML)'],
        mode='lines+markers',
        name='Predicted Consumption',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=f"7-Day Water Consumption Forecast - {municipality}",
        xaxis_title="Date",
        yaxis_title="Water Consumption (Million Liters)",
        height=400,
        hovermode='x unified'
    )
    
    return fig, forecast_df

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<div class="main-header">💧 Water Consumption Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Multi-Municipality Water Demand Prediction</div>', unsafe_allow_html=True)

# Check if model is loaded
if model is None or df is None:
    st.error("⚠️ Model or data not loaded. Please ensure all required files are present.")
    st.stop()

# Success message
st.success("✅ ML Model Loaded Successfully | 99.4% Accuracy | 10 Municipalities")

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Municipality selection
    municipalities = sorted(df['region_name'].unique())
    selected_municipality = st.selectbox(
        "Select Municipality",
        municipalities,
        index=municipalities.index('Tirupati') if 'Tirupati' in municipalities else 0
    )
    
    st.markdown("---")
    
    # Weather inputs
    st.subheader("🌡️ Weather Conditions")
    temperature = st.slider("Temperature (°C)", 15, 45, 32)
    humidity = st.slider("Humidity (%)", 20, 100, 65)
    rainfall = st.slider("Rainfall (mm)", 0, 50, 0)
    
    st.markdown("---")
    
    # Day type
    st.subheader("📅 Day Type")
    day_type = st.radio("Select Day Type", ["Weekday", "Weekend", "Holiday"])
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 Get Prediction", use_container_width=True)
    
    st.markdown("---")
    
    # Municipality info
    muni_data = df[df['region_name'] == selected_municipality]
    st.subheader(f"ℹ️ {selected_municipality} Info")
    st.metric("Population", f"{int(muni_data['population'].mean()):,}")
    st.metric("Industrial Index", f"{int(muni_data['industrial_activity_index'].mean())}/5")
    st.metric("Avg Consumption", f"{muni_data['water_consumption_liters'].mean()/1_000_000:.1f}M L")

# Main content area
if predict_button or 'first_run' not in st.session_state:
    st.session_state.first_run = False
    
    # Make prediction
    prediction, average, change_percent = make_prediction(
        selected_municipality, temperature, humidity, rainfall, day_type
    )
    
    # Display prediction
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Predicted Consumption",
            f"{prediction:.2f} ML",
            f"{change_percent:+.2f}%"
        )
    
    with col2:
        st.metric(
            "📊 Average Consumption",
            f"{average:.2f} ML"
        )
    
    with col3:
        st.metric(
            "🌡️ Temperature",
            f"{temperature}°C"
        )
    
    with col4:
        st.metric(
            "💧 Humidity",
            f"{humidity}%"
        )
    
    # Prediction visualization
    st.markdown("---")
    st.subheader("📈 Prediction vs Average")
    
    fig_pred = go.Figure()
    
    fig_pred.add_trace(go.Bar(
        x=['Average', 'Predicted'],
        y=[average, prediction],
        marker_color=['#FFA726', '#1E88E5'],
        text=[f"{average:.1f} ML", f"{prediction:.1f} ML"],
        textposition='outside'
    ))
    
    fig_pred.update_layout(
        title=f"{selected_municipality} - Water Consumption Comparison",
        yaxis_title="Water Consumption (Million Liters)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🏙️ Municipality Comparison", "📅 7-Day Forecast", "📊 Model Info"])
    
    with tab1:
        st.subheader("Compare All Municipalities")
        st.write(f"Under current conditions: {temperature}°C, {humidity}% humidity, {rainfall}mm rainfall, {day_type}")
        
        comparison_fig, comparison_df = create_comparison_chart(temperature, humidity, rainfall, day_type)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        total = comparison_df['Predicted (ML)'].sum()
        st.info(f"💧 **Total water needed for all municipalities:** {total:.2f} Million Liters")
    
    with tab2:
        st.subheader(f"7-Day Forecast for {selected_municipality}")
        
        forecast_fig, forecast_df = create_forecast(selected_municipality)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Download forecast
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{selected_municipality}_forecast.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Model Details:**
            - **Algorithm:** Random Forest Regressor
            - **Accuracy (R²):** 99.4%
            - **Training Samples:** 100,000
            - **Features:** 13
            - **Municipalities:** 10
            """)
        
        with col2:
            st.markdown("""
            **Key Features:**
            - Temperature & Humidity
            - Rainfall
            - Day Type (Weekday/Weekend/Holiday)
            - Municipality
            - Population & Industrial Index
            - Historical Consumption
            - Seasonal Patterns
            """)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Prediction"
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>💧 Water Consumption Forecasting System | Built with Streamlit & Machine Learning</p>
    <p>Powered by Random Forest Algorithm | Real-time Predictions for 10 Municipalities</p>
</div>
""", unsafe_allow_html=True)
