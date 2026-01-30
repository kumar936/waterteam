"""
===============================================================================
WATER CONSUMPTION PREDICTION SCRIPT
Use the trained model to make predictions for any municipality
===============================================================================
"""

import pandas as pd
import pickle
import numpy as np

# ============================================================================
# STEP 1: LOAD THE TRAINED MODEL
# ============================================================================
print("=" * 80)
print("Loading trained model and encoders...")
print("=" * 80)

# Load the model
with open('water_consumption_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✓ Model loaded successfully")

# Load the municipality encoder
with open('municipality_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
print("✓ Municipality encoder loaded")

# Load feature columns
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
print("✓ Feature columns loaded")
print()

print("Available municipalities:")
for i, municipality in enumerate(le.classes_):
    print(f"  {i}. {municipality}")
print()

# ============================================================================
# STEP 2: MAKE A PREDICTION FOR A SPECIFIC MUNICIPALITY
# ============================================================================
print("=" * 80)
print("EXAMPLE 1: Predict for Tirupati on a hot weekday")
print("=" * 80)

# Define input features
municipality_name = 'Tirupati'
temperature = 35          # Celsius
humidity = 60             # Percent
rainfall = 0              # mm
is_weekend = 0            # 0 = No, 1 = Yes
is_holiday = 0            # 0 = No, 1 = Yes
population = 675000       # Tirupati population
industrial_index = 3      # Scale 1-5
prev_day_consumption = 142000000      # 142 Million liters
prev_7day_avg = 140000000             # 140 Million liters

# Encode municipality
municipality_code = le.transform([municipality_name])[0]

# Create input DataFrame
input_data = pd.DataFrame({
    'temperature_celsius': [temperature],
    'humidity_percent': [humidity],
    'rainfall_mm': [rainfall],
    'is_weekend': [is_weekend],
    'is_holiday': [is_holiday],
    'municipality_encoded': [municipality_code],
    'population': [population],
    'industrial_activity_index': [industrial_index],
    'prev_day_consumption': [prev_day_consumption],
    'prev_7day_avg_consumption': [prev_7day_avg]
})

# Make prediction
prediction = model.predict(input_data)[0]

print(f"INPUT:")
print(f"  Municipality: {municipality_name}")
print(f"  Temperature: {temperature}°C")
print(f"  Humidity: {humidity}%")
print(f"  Rainfall: {rainfall}mm")
print(f"  Day Type: {'Weekend' if is_weekend else 'Holiday' if is_holiday else 'Weekday'}")
print()
print(f"PREDICTION:")
print(f"  Expected Water Consumption: {prediction/1_000_000:.1f} Million Liters")
print()

# ============================================================================
# STEP 3: COMPARE PREDICTIONS ACROSS MUNICIPALITIES
# ============================================================================
print("=" * 80)
print("EXAMPLE 2: Compare all municipalities under same conditions")
print("=" * 80)

# Same weather conditions for all cities
conditions = {
    'temperature': 32,
    'humidity': 65,
    'rainfall': 0,
    'is_weekend': 0,
    'is_holiday': 0
}

print(f"Conditions: {conditions['temperature']}°C, {conditions['humidity']}% humidity, Weekday")
print()

# Load actual dataset to get average values per municipality
df = pd.read_csv('water_consumption_100000_rows.csv')

predictions_comparison = []

for municipality in le.classes_:
    # Get municipality-specific data
    muni_data = df[df['region_name'] == municipality]
    municipality_code = le.transform([municipality])[0]
    
    # Create input
    input_data = pd.DataFrame({
        'temperature_celsius': [conditions['temperature']],
        'humidity_percent': [conditions['humidity']],
        'rainfall_mm': [conditions['rainfall']],
        'is_weekend': [conditions['is_weekend']],
        'is_holiday': [conditions['is_holiday']],
        'municipality_encoded': [municipality_code],
        'population': [int(muni_data['population'].mean())],
        'industrial_activity_index': [int(muni_data['industrial_activity_index'].mean())],
        'prev_day_consumption': [int(muni_data['water_consumption_liters'].mean())],
        'prev_7day_avg_consumption': [int(muni_data['water_consumption_liters'].mean())]
    })
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    predictions_comparison.append({
        'Municipality': municipality,
        'Predicted (M L)': prediction / 1_000_000,
        'Population': int(muni_data['population'].mean())
    })

# Display as table
comparison_df = pd.DataFrame(predictions_comparison)
comparison_df = comparison_df.sort_values('Predicted (M L)', ascending=False)

print(comparison_df.to_string(index=False))
print()
print(f"Total water needed for all municipalities: {comparison_df['Predicted (M L)'].sum():.1f} Million Liters")
print()

# ============================================================================
# STEP 4: WHAT-IF SCENARIO ANALYSIS
# ============================================================================
print("=" * 80)
print("EXAMPLE 3: What-if scenario - Temperature impact on Guntur")
print("=" * 80)

municipality_name = 'Guntur'
municipality_code = le.transform([municipality_name])[0]
muni_data = df[df['region_name'] == municipality_name]

# Test different temperatures
temperatures = [25, 30, 35, 40]
predictions_temp = []

print(f"Analyzing temperature impact on {municipality_name}:")
print()

for temp in temperatures:
    input_data = pd.DataFrame({
        'temperature_celsius': [temp],
        'humidity_percent': [65],
        'rainfall_mm': [0],
        'is_weekend': [0],
        'is_holiday': [0],
        'municipality_encoded': [municipality_code],
        'population': [int(muni_data['population'].mean())],
        'industrial_activity_index': [int(muni_data['industrial_activity_index'].mean())],
        'prev_day_consumption': [int(muni_data['water_consumption_liters'].mean())],
        'prev_7day_avg_consumption': [int(muni_data['water_consumption_liters'].mean())]
    })
    
    prediction = model.predict(input_data)[0] / 1_000_000
    predictions_temp.append(prediction)
    
    print(f"  {temp}°C → {prediction:.1f} Million Liters")

# Calculate increase
baseline = predictions_temp[0]
peak = predictions_temp[-1]
increase_percent = ((peak - baseline) / baseline) * 100

print()
print(f"INSIGHT: Temperature increase from {temperatures[0]}°C to {temperatures[-1]}°C")
print(f"         causes {increase_percent:.1f}% increase in water consumption")
print()

# ============================================================================
# STEP 5: INTERACTIVE PREDICTION FUNCTION
# ============================================================================
print("=" * 80)
print("EXAMPLE 4: Interactive prediction function")
print("=" * 80)

def predict_water_consumption(municipality, temperature, humidity, rainfall, 
                             day_type='weekday'):
    """
    Predict water consumption for a given municipality and conditions.
    
    Parameters:
    -----------
    municipality : str
        Name of municipality (e.g., 'Tirupati', 'Guntur')
    temperature : float
        Temperature in Celsius
    humidity : float
        Humidity percentage
    rainfall : float
        Rainfall in mm
    day_type : str
        'weekday', 'weekend', or 'holiday'
    
    Returns:
    --------
    float : Predicted water consumption in Million Liters
    """
    
    # Encode municipality
    try:
        municipality_code = le.transform([municipality])[0]
    except ValueError:
        print(f"Error: '{municipality}' not found in trained municipalities")
        print(f"Available: {list(le.classes_)}")
        return None
    
    # Get municipality data
    muni_data = df[df['region_name'] == municipality]
    
    # Set day type flags
    is_weekend = 1 if day_type == 'weekend' else 0
    is_holiday = 1 if day_type == 'holiday' else 0
    
    # Create input
    input_data = pd.DataFrame({
        'temperature_celsius': [temperature],
        'humidity_percent': [humidity],
        'rainfall_mm': [rainfall],
        'is_weekend': [is_weekend],
        'is_holiday': [is_holiday],
        'municipality_encoded': [municipality_code],
        'population': [int(muni_data['population'].mean())],
        'industrial_activity_index': [int(muni_data['industrial_activity_index'].mean())],
        'prev_day_consumption': [int(muni_data['water_consumption_liters'].mean())],
        'prev_7day_avg_consumption': [int(muni_data['water_consumption_liters'].mean())]
    })
    
    # Predict
    prediction = model.predict(input_data)[0] / 1_000_000
    
    return prediction

# Test the function
print("Testing prediction function:")
print()

test_cases = [
    ('Tirupati', 35, 70, 0, 'weekday'),
    ('Guntur', 30, 60, 5, 'weekend'),
    ('Nellore', 38, 65, 0, 'holiday')
]

for municipality, temp, humidity, rain, day_type in test_cases:
    result = predict_water_consumption(municipality, temp, humidity, rain, day_type)
    print(f"{municipality:12s} | {temp}°C | {humidity}% | {rain}mm | {day_type:8s} → {result:.1f} M L")

print()

# ============================================================================
# STEP 6: SAVE PREDICTIONS TO CSV
# ============================================================================
print("=" * 80)
print("EXAMPLE 5: Generate predictions for next 7 days")
print("=" * 80)

# Simulate weather forecast for next 7 days
forecast_data = []
dates = pd.date_range('2026-01-12', periods=7, freq='D')
municipalities = ['Tirupati', 'Guntur', 'Nellore', 'Vijayawada']

# Simulated weather (in real scenario, this comes from weather API)
weather_forecast = [
    {'temp': 32, 'humidity': 65, 'rainfall': 0},
    {'temp': 33, 'humidity': 68, 'rainfall': 0},
    {'temp': 35, 'humidity': 70, 'rainfall': 0},
    {'temp': 34, 'humidity': 67, 'rainfall': 2},
    {'temp': 30, 'humidity': 75, 'rainfall': 5},
    {'temp': 28, 'humidity': 80, 'rainfall': 10},
    {'temp': 29, 'humidity': 78, 'rainfall': 3}
]

for municipality in municipalities:
    for i, date in enumerate(dates):
        weather = weather_forecast[i]
        is_weekend = 1 if date.dayofweek >= 5 else 0
        
        prediction = predict_water_consumption(
            municipality,
            weather['temp'],
            weather['humidity'],
            weather['rainfall'],
            'weekend' if is_weekend else 'weekday'
        )
        
        forecast_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Municipality': municipality,
            'Temperature': weather['temp'],
            'Humidity': weather['humidity'],
            'Rainfall': weather['rainfall'],
            'Day_Type': 'Weekend' if is_weekend else 'Weekday',
            'Predicted_Consumption_ML': round(prediction, 2)
        })

forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_csv('7day_forecast.csv', index=False)

print("7-Day Forecast Preview:")
print(forecast_df.head(10).to_string(index=False))
print()
print(f"✓ Full forecast saved to: 7day_forecast.csv")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("PREDICTION EXAMPLES COMPLETED")
print("=" * 80)
print()
print("WHAT YOU LEARNED:")
print("  1. How to load a trained model")
print("  2. How to make predictions for specific municipalities")
print("  3. How to compare predictions across cities")
print("  4. How to analyze what-if scenarios")
print("  5. How to create a reusable prediction function")
print("  6. How to generate forecasts and save to CSV")
print()
print("YOU CAN NOW:")
print("  - Integrate this with your dashboard")
print("  - Create an API for real-time predictions")
print("  - Generate daily forecasts automatically")
print()
print("=" * 80)
