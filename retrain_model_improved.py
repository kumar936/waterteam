"""
IMPROVED MODEL TRAINING - Reduce prev_day_consumption dominance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAINING MODEL WITH IMPROVED FEATURE ENGINEERING")
print("="*80)

# Load data
df = pd.read_csv('water_consumption_100000_rows_improved.csv')

# FEATURE ENGINEERING - Normalize historical features to reduce dominance
print("\nSTEP 1: Feature Engineering")
print("-" * 80)

# Create normalized versions of consumption features
df['prev_day_consumption_normalized'] = df['prev_day_consumption'] / 100_000_000
df['prev_7day_avg_normalized'] = df['prev_7day_avg_consumption'] / 100_000_000

# Add new meaningful features
df['consumption_variance'] = df['water_consumption_liters'] / df['prev_7day_avg_consumption']
df['population_scaled'] = df['population'] / 1_000_000
df['industrial_scaled'] = df['industrial_activity_index']

# Encode municipality
le = LabelEncoder()
df['municipality_encoded'] = le.fit_transform(df['region_name'])

print("[OK] Feature engineering completed")
print(f"  Municipalities: {list(le.classes_)}")

# NEW FEATURE SET - Better balanced
feature_columns = [
    'temperature_celsius',           # Weather: Strong predictor
    'humidity_percent',              # Weather
    'rainfall_mm',                   # Weather
    'is_weekend',                    # Pattern
    'is_holiday',                    # Pattern
    'municipality_encoded',          # Location
    'population_scaled',             # City size (normalized)
    'industrial_scaled',             # Economic activity
    'prev_day_consumption_normalized',  # Recent history (normalized)
    'prev_7day_avg_normalized',      # Weekly pattern (normalized)
    'consumption_variance',          # Relative change
    'month',                         # Seasonal
    'season'                         # Seasonal encoded
]

# Encode season
season_map = {'Winter': 0, 'Summer': 1, 'Monsoon': 2, 'Spring': 3}
df['season'] = df['season'].map(season_map)

X = df[feature_columns]
y = df['water_consumption_liters']

print(f"[OK] Features selected: {len(feature_columns)}")
print(f"  Training samples: {len(X)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"[OK] Train: {len(X_train):,} | Test: {len(X_test):,}")

# Train improved model
print("\nSTEP 2: Training Improved Random Forest")
print("-" * 80)

model = RandomForestRegressor(
    n_estimators=15,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)
print("[OK] Model training completed")

# Evaluate
print("\nSTEP 3: Model Evaluation")
print("-" * 80)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred) / 1_000_000
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) / 1_000_000

print(f"Training R2: {train_r2:.4f}")
print(f"Testing R2: {test_r2:.4f}")
print(f"MAE: {test_mae:.2f}M L")
print(f"RMSE: {test_rmse:.2f}M L")

# Feature importance analysis
print("\nSTEP 4: Feature Importance")
print("-" * 80)

importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

# Save improved model
print("\nSTEP 5: Saving Model")
print("-" * 80)

with open('water_consumption_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("[OK] Model saved: water_consumption_model.pkl")

with open('municipality_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("[OK] Encoder saved: municipality_encoder.pkl")

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("[OK] Features saved: feature_columns.pkl")

print("\n" + "="*80)
print("TEST PREDICTIONS - Different Municipalities")
print("="*80)

for municipality in ['Anantapur', 'Kurnool', 'Vijayawada', 'Tirupati']:
    muni_data = df[df['region_name'] == municipality]
    muni_code = le.transform([municipality])[0]
    
    test_features = {
        'temperature_celsius': 32,
        'humidity_percent': 65,
        'rainfall_mm': 0,
        'is_weekend': 0,
        'is_holiday': 0,
        'municipality_encoded': muni_code,
        'population_scaled': muni_data['population'].mean() / 1_000_000,
        'industrial_scaled': muni_data['industrial_activity_index'].mean(),
        'prev_day_consumption_normalized': muni_data['water_consumption_liters'].mean() / 100_000_000,
        'prev_7day_avg_normalized': muni_data['water_consumption_liters'].mean() / 100_000_000,
        'consumption_variance': 1.0,
        'month': 1,
        'season': 0
    }
    
    X_test_single = pd.DataFrame([test_features])[feature_columns]
    pred = model.predict(X_test_single)[0]
    
    print(f"\n{municipality}:")
    print(f"  Predicted: {pred/1_000_000:.2f}M L")
    print(f"  Average:   {muni_data['water_consumption_liters'].mean()/1_000_000:.2f}M L")
    print(f"  Change:    {((pred - muni_data['water_consumption_liters'].mean()) / muni_data['water_consumption_liters'].mean() * 100):+.2f}%")

print("\n" + "="*80)
print("RETRAINED MODEL READY! Restart API server to use new model.")
print("="*80)
