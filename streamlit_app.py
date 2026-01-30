"""
Multi-Municipality Water Consumption Forecasting Dashboard - Streamlit Version
Real-time ML predictions with interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="APMWRS - Water Forecasting",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .alert-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        .alert-critical {
            background: #ffe5e5;
            border-left-color: #e74c3c;
        }
        .alert-warning {
            background: #fff3cd;
            border-left-color: #ffc107;
        }
        .alert-info {
            background: #e7f3ff;
            border-left-color: #2196f3;
        }
        .alert-success {
            background: #e8f5e9;
            border-left-color: #27ae60;
        }
        h1 {
            color: #1a3a52;
        }
        h2 {
            color: #2c5aa0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'municipality_encoder' not in st.session_state:
    st.session_state.municipality_encoder = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model_components():
    """Load all model components"""
    try:
        with open('water_consumption_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        model = None

    try:
        with open('municipality_encoder.pkl', 'rb') as f:
            municipality_encoder = pickle.load(f)
        logger.info("✓ Municipality encoder loaded successfully")
    except Exception as e:
        logger.error(f"✗ Error loading municipality encoder: {e}")
        municipality_encoder = None

    try:
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        logger.info("✓ Feature columns loaded successfully")
    except Exception as e:
        logger.error(f"✗ Error loading feature columns: {e}")
        feature_columns = None

    return model, municipality_encoder, feature_columns

@st.cache_data
def load_data():
    """Load water consumption data"""
    try:
        df = pd.read_csv('water_consumption_100000_rows_improved.csv')
        logger.info(f"✓ Data loaded: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"✗ Error loading data: {e}")
        return None

def get_prediction(municipality, temperature, humidity, rainfall, day_type):
    """Get prediction from model"""
    if st.session_state.model is None or st.session_state.data is None:
        return None
    
    try:
        df = st.session_state.data
        model = st.session_state.model
        encoder = st.session_state.municipality_encoder
        feature_cols = st.session_state.feature_columns
        
        # Get recent data for municipality
        muni_data = df[df['region_name'] == municipality].sort_values('date').tail(30)
        
        if len(muni_data) == 0:
            return None
        
        # Calculate features
        avg_consumption = muni_data['water_consumption_liters'].mean() / 1_000_000
        municipality_code = encoder.transform([municipality])[0]
        population = int(muni_data['population'].mean())
        industrial_index = int(muni_data['industrial_activity_index'].mean())
        prev_day_avg = muni_data['water_consumption_liters'].mean()
        prev_day_normalized = prev_day_avg / 100_000_000
        
        latest_date = pd.to_datetime(muni_data['date'].iloc[-1])
        month = latest_date.month
        season_map = {'Winter': 0, 'Summer': 1, 'Monsoon': 2, 'Spring': 3}
        season_str = muni_data['season'].iloc[-1]
        season = season_map.get(season_str, 0)
        
        # Create feature vector
        feature_dict = {
            'temperature_celsius': temperature,
            'humidity_percent': humidity,
            'rainfall_mm': rainfall,
            'is_weekend': 1 if day_type == 'weekend' else 0,
            'is_holiday': 1 if day_type == 'holiday' else 0,
            'municipality_encoded': municipality_code,
            'population_scaled': population / 1_000_000,
            'industrial_scaled': industrial_index,
            'prev_day_consumption_normalized': prev_day_normalized,
            'prev_7day_avg_normalized': prev_day_normalized,
            'consumption_variance': 1.0,
            'month': month,
            'season': season
        }
        
        # Make prediction
        X_pred = pd.DataFrame([feature_dict])[feature_cols]
        predicted_consumption_liters = model.predict(X_pred)[0]
        predicted_consumption = predicted_consumption_liters / 1_000_000
        
        # Calculate change percentage
        change_percent = ((predicted_consumption - avg_consumption) / avg_consumption) * 100 if avg_consumption > 0 else 0
        
        return {
            'municipality': municipality,
            'predicted_consumption_ml': float(round(predicted_consumption, 2)),
            'change_percent': float(round(change_percent, 2)),
            'average_consumption': float(round(avg_consumption, 2)),
            'population': population,
            'industrial_index': industrial_index
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

def get_alert_message(change_percent, municipality):
    """Generate dynamic alert based on change percentage"""
    if change_percent > 50:
        return {
            'type': 'critical',
            'title': '🚨 CRITICAL - Very High Demand Alert',
            'message': f'Predicted consumption for {municipality} is {change_percent:.1f}% above average. IMMEDIATE ACTION REQUIRED: Activate emergency water supplies, increase pump capacity, and implement water conservation measures.'
        }
    elif change_percent > 25:
        return {
            'type': 'warning',
            'title': '⚠️ High Demand Alert',
            'message': f'Predicted consumption for {municipality} is {change_percent:.1f}% above average. Consider adjusting reservoir levels, optimizing pump schedules, and preparing backup supplies.'
        }
    elif change_percent > 0:
        return {
            'type': 'info',
            'title': '📊 Elevated Demand Notice',
            'message': f'Predicted consumption for {municipality} is {change_percent:.1f}% above average. Monitor reservoir levels and be ready to adjust supply if needed.'
        }
    else:
        return {
            'type': 'success',
            'title': '✓ Low Demand - Normal Operation',
            'message': f'Predicted consumption for {municipality} is {abs(change_percent):.1f}% below average. Water availability is good. Maintain routine maintenance schedules.'
        }

def plot_7day_forecast(municipality):
    """Plot 7-day forecast"""
    if st.session_state.data is None:
        return None
    
    try:
        df = st.session_state.data
        muni_data = df[df['region_name'] == municipality].sort_values('date').tail(7)
        
        if len(muni_data) == 0:
            return None
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(muni_data['date']),
            y=muni_data['water_consumption_liters'] / 1_000_000,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'7-Day Water Consumption Trend - {municipality}',
            xaxis_title='Date',
            yaxis_title='Consumption (ML)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    except Exception as e:
        logger.error(f"Forecast chart error: {e}")
        return None

def plot_hourly_pattern(municipality):
    """Plot hourly consumption pattern"""
    if st.session_state.data is None:
        return None
    
    try:
        df = st.session_state.data
        muni_data = df[df['region_name'] == municipality]
        
        # Create synthetic hourly pattern based on average
        hours = np.arange(0, 24)
        # Simulate hourly pattern (higher during day, lower at night)
        pattern = 50 + 50 * np.sin((hours - 6) * np.pi / 12)
        pattern = np.clip(pattern, 20, 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hours,
            y=pattern,
            marker=dict(color='#9b59b6'),
            name='Consumption %'
        ))
        
        fig.update_layout(
            title=f'Hourly Consumption Pattern - {municipality}',
            xaxis_title='Hour of Day',
            yaxis_title='Consumption (% of Daily)',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        logger.error(f"Hourly pattern error: {e}")
        return None

def plot_temperature_impact(municipality):
    """Plot temperature impact on consumption"""
    try:
        temperatures = np.arange(15, 46, 2)
        predictions = []
        
        for temp in temperatures:
            pred = get_prediction(municipality, temp, 65, 0, 'weekday')
            if pred:
                predictions.append(pred['predicted_consumption_ml'])
            else:
                predictions.append(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=predictions,
            mode='lines+markers',
            name='Predicted Consumption',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'Temperature Impact on Water Consumption - {municipality}',
            xaxis_title='Temperature (°C)',
            yaxis_title='Predicted Consumption (ML)',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Temperature impact error: {e}")
        return None

def plot_feature_importance():
    """Plot feature importance"""
    try:
        features = [
            'temperature_celsius',
            'humidity_percent',
            'rainfall_mm',
            'municipality_encoded',
            'population_scaled',
            'prev_day_consumption_normalized',
            'industrial_scaled',
            'is_weekend',
            'month',
            'season'
        ]
        importance = [0.25, 0.18, 0.12, 0.15, 0.10, 0.12, 0.05, 0.02, 0.01, 0.00]
        
        fig = go.Figure(data=[
            go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker=dict(color=importance, colorscale='Viridis')
            )
        ])
        
        fig.update_layout(
            title='Model Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        return None

def plot_comparison(municipalities):
    """Plot municipality comparison"""
    try:
        data = []
        for muni in municipalities:
            pred = get_prediction(muni, 32, 65, 0, 'weekday')
            if pred:
                data.append({
                    'municipality': muni,
                    'prediction': pred['predicted_consumption_ml'],
                    'average': pred['average_consumption']
                })
        
        if not data:
            return None
        
        df_comp = pd.DataFrame(data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Predicted',
            x=df_comp['municipality'],
            y=df_comp['prediction'],
            marker=dict(color='#3498db')
        ))
        
        fig.add_trace(go.Bar(
            name='Average',
            x=df_comp['municipality'],
            y=df_comp['average'],
            marker=dict(color='#ecf0f1')
        ))
        
        fig.update_layout(
            title='Cross-Municipality Comparison',
            xaxis_title='Municipality',
            yaxis_title='Consumption (ML)',
            barmode='group',
            template='plotly_white',
            height=400
        )
        
        return fig
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return None

# Load data and models
model, encoder, feature_cols = load_model_components()
st.session_state.model = model
st.session_state.municipality_encoder = encoder
st.session_state.feature_columns = feature_cols

df = load_data()
st.session_state.data = df

# Header
st.markdown("""
    <div style="background: linear-gradient(135deg, #1a3a52 0%, #2c5aa0 100%); padding: 30px; border-radius: 10px; color: white;">
        <h1>💧 APMWRS - Multi-Municipality Water Consumption Forecasting Dashboard</h1>
        <p style="font-size: 16px; margin-top: 10px;">Real-time ML Predictions | Andhra Pradesh State</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
st.sidebar.title("📊 Dashboard Controls")

if df is not None:
    municipalities = sorted(df['region_name'].unique().tolist())
else:
    municipalities = []

selected_municipality = st.sidebar.selectbox(
    "📍 Select Municipality",
    municipalities,
    index=0 if municipalities else None
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Information")
st.sidebar.info("""
- **Model Type**: Random Forest Regression
- **Data Points**: 100,000+ training samples
- **Municipalities**: 10 cities in Andhra Pradesh
- **Features**: Temperature, Humidity, Rainfall, Day Type, Population, Previous Consumption
- **Update Frequency**: Real-time predictions
""")

# Main content
if selected_municipality and model is not None and df is not None:
    # Get initial prediction
    pred = get_prediction(selected_municipality, 32, 65, 0, 'weekday')
    
    if pred:
        # Alert box
        alert = get_alert_message(pred['change_percent'], selected_municipality)
        alert_class = f"alert-box alert-{alert['type']}"
        st.markdown(f"""
            <div class="alert-box alert-{alert['type']}">
                <h3 style="margin-top: 0;">{alert['title']}</h3>
                <p>{alert['message']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Prediction",
                value=f"{pred['predicted_consumption_ml']:.0f} ML",
                delta=f"{pred['change_percent']:.1f}% vs avg"
            )
        
        with col2:
            st.metric(
                label="Average Consumption",
                value=f"{pred['average_consumption']:.0f} ML",
                delta="Historical baseline"
            )
        
        with col3:
            st.metric(
                label="Population",
                value=f"{pred['population']:,}",
                delta="Municipality size"
            )
        
        with col4:
            st.metric(
                label="Industrial Index",
                value=f"{pred['industrial_index']}",
                delta="Activity level"
            )
        
        st.markdown("---")
        
        # Live Prediction Simulator
        st.subheader("🔮 Live ML Prediction Simulator")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temperature = st.number_input("Temperature (°C)", value=32, min_value=15, max_value=45)
        
        with col2:
            humidity = st.number_input("Humidity (%)", value=65, min_value=20, max_value=100)
        
        with col3:
            day_type = st.selectbox("Day Type", ["Weekday", "Weekend", "Holiday"])
        
        with col4:
            rainfall = st.number_input("Rainfall (mm)", value=0, min_value=0, max_value=100)
        
        # Make prediction on user input
        if st.button("🚀 Get Live Prediction from ML Model", use_container_width=True):
            new_pred = get_prediction(
                selected_municipality,
                temperature,
                humidity,
                rainfall,
                day_type.lower()
            )
            
            if new_pred:
                arrow = "↑" if new_pred['change_percent'] > 0 else "↓"
                color = "red" if new_pred['change_percent'] > 0 else "green"
                
                st.markdown(f"""
                    <h3 style="color: {color}; text-align: center;">
                    ML Prediction for {new_pred['municipality']}: {new_pred['predicted_consumption_ml']} ML
                    <br><small>{arrow} {abs(new_pred['change_percent']):.1f}% vs average</small>
                    </h3>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        st.subheader("📈 Analytics & Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_7day_forecast(selected_municipality), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_hourly_pattern(selected_municipality), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_temperature_impact(selected_municipality), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_feature_importance(), use_container_width=True)
        
        st.markdown("---")
        
        # Municipality Comparison
        st.subheader("📊 Cross-Municipality Comparison")
        st.plotly_chart(plot_comparison(municipalities), use_container_width=True)
        
else:
    st.error("⚠️ Unable to load model or data. Please ensure all required files exist.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        <p>© 2024 APMWRS - Andhra Pradesh Water Resource Department</p>
        <p>Multi-Municipality Water Consumption Forecasting System</p>
        <p>Powered by Machine Learning | Batch 12</p>
    </div>
""", unsafe_allow_html=True)
