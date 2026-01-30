"""
===============================================================================
MULTI-MUNICIPALITY WATER CONSUMPTION FORECASTING
Complete Model Training, Testing, and Evaluation Code
===============================================================================

This script trains a Random Forest model on 10 Andhra Pradesh municipalities
to predict daily water consumption.

Dataset: water_consumption_100000_rows.csv
Municipalities: Tirupati, Guntur, Nellore, Vijayawada, Kakinada, 
                Vizianagaram, Kadapa, Kurnool, Rajahmundry, Anantapur
"""

# ============================================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================================
print("=" * 80)
print("STEP 1: Importing necessary libraries...")
print("=" * 80)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully!\n")

# ============================================================================
# STEP 2: LOAD DATASET
# ============================================================================
print("=" * 80)
print("STEP 2: Loading dataset...")
print("=" * 80)

# Load the CSV file
df = pd.read_csv('water_consumption_100000_rows.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  - Total rows: {len(df):,}")
print(f"  - Total columns: {len(df.columns)}")
print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
print()

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())
print()

# ============================================================================
# STEP 3: EXPLORE THE DATA
# ============================================================================
print("=" * 80)
print("STEP 3: Exploring the data...")
print("=" * 80)

# Check municipalities
print(f"Municipalities in dataset:")
print(df['region_name'].value_counts())
print(f"\nTotal unique municipalities: {df['region_name'].nunique()}")
print()

# Check for missing values
print("Missing values in dataset:")
print(df.isnull().sum()[df.isnull().sum() > 0])
if df.isnull().sum().sum() == 0:
    print("✓ No missing values found!")
print()

# Basic statistics
print("Water Consumption Statistics by Municipality:")
stats = df.groupby('region_name')['water_consumption_liters'].agg([
    'count', 'mean', 'min', 'max'
])
stats['mean'] = stats['mean'] / 1_000_000  # Convert to millions
stats['min'] = stats['min'] / 1_000_000
stats['max'] = stats['max'] / 1_000_000
stats.columns = ['Count', 'Avg (M L)', 'Min (M L)', 'Max (M L)']
print(stats.round(2))
print()

# ============================================================================
# STEP 4: PREPARE FEATURES
# ============================================================================
print("=" * 80)
print("STEP 4: Preparing features for the model...")
print("=" * 80)

# Encode municipality names to numbers (0-9)
print("Encoding municipalities...")
le = LabelEncoder()
df['municipality_encoded'] = le.fit_transform(df['region_name'])

# Show encoding mapping
print("\nMunicipality Encoding:")
for i, municipality in enumerate(le.classes_):
    print(f"  {municipality:15s} → {i}")
print()

# Select features for the model
feature_columns = [
    'temperature_celsius',           # Weather feature
    'humidity_percent',              # Weather feature
    'rainfall_mm',                   # Weather feature
    'is_weekend',                    # Calendar feature
    'is_holiday',                    # Calendar feature
    'municipality_encoded',          # KEY: Which city (0-9)
    'population',                    # City characteristic
    'industrial_activity_index',     # City characteristic
    'prev_day_consumption',          # Historical pattern
    'prev_7day_avg_consumption'      # Historical pattern
]

# Target variable (what we want to predict)
target_column = 'water_consumption_liters'

print(f"Selected {len(feature_columns)} features:")
for i, feature in enumerate(feature_columns, 1):
    print(f"  {i}. {feature}")
print(f"\nTarget variable: {target_column}")
print()

# Create feature matrix (X) and target vector (y)
X = df[feature_columns]
y = df[target_column]

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target vector shape: {y.shape}")
print()

# ============================================================================
# STEP 5: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("=" * 80)
print("STEP 5: Splitting data into training and testing sets...")
print("=" * 80)

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)

print(f"✓ Data split completed!")
print(f"  - Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  - Testing set:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
print()

# ============================================================================
# STEP 6: BUILD AND TRAIN THE MODEL
# ============================================================================
print("=" * 80)
print("STEP 6: Building and training Random Forest model...")
print("=" * 80)

# Create Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=10,         # Number of trees in the forest
    max_depth=10,            # Maximum depth of each tree
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=2,      # Minimum samples in a leaf node
    random_state=42,         # For reproducibility
    n_jobs=-1,               # Use all CPU cores
    verbose=1                # Show progress
)

print("Model parameters:")
print(f"  - Number of trees: {model.n_estimators}")
print(f"  - Max depth: {model.max_depth}")
print(f"  - Random state: {model.random_state}")
print()

print("Training the model... (this may take a few minutes)")
model.fit(X_train, y_train)

print("✓ Model training completed!\n")

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================
print("=" * 80)
print("STEP 7: Making predictions on test set...")
print("=" * 80)

# Predict on training set
y_train_pred = model.predict(X_train)

# Predict on test set
y_test_pred = model.predict(X_test)

print(f"✓ Predictions completed!")
print(f"  - Training predictions: {len(y_train_pred):,}")
print(f"  - Testing predictions: {len(y_test_pred):,}")
print()

# ============================================================================
# STEP 8: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("=" * 80)
print("STEP 8: Evaluating model performance...")
print("=" * 80)

# Calculate metrics for TRAINING set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for TESTING set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("TRAINING SET PERFORMANCE:")
print(f"  - R² Score:  {train_r2:.4f} (closer to 1.0 is better)")
print(f"  - MAE:       {train_mae/1_000_000:.2f} Million Liters")
print(f"  - RMSE:      {train_rmse/1_000_000:.2f} Million Liters")
print()

print("TESTING SET PERFORMANCE:")
print(f"  - R² Score:  {test_r2:.4f} (closer to 1.0 is better)")
print(f"  - MAE:       {test_mae/1_000_000:.2f} Million Liters")
print(f"  - RMSE:      {test_rmse/1_000_000:.2f} Million Liters")
print()

# Interpretation
print("INTERPRETATION:")
if test_r2 > 0.90:
    print(f"  ✓ EXCELLENT! Model explains {test_r2*100:.1f}% of variance in water consumption")
elif test_r2 > 0.80:
    print(f"  ✓ GOOD! Model explains {test_r2*100:.1f}% of variance in water consumption")
else:
    print(f"  ⚠ FAIR. Model explains {test_r2*100:.1f}% of variance - consider more features")

print(f"  ✓ On average, predictions are off by {test_mae/1_000_000:.2f}M liters")
print()

# ============================================================================
# STEP 9: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("=" * 80)
print("STEP 9: Analyzing feature importance...")
print("=" * 80)

# Get feature importance from the model
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance Ranking:")
print(feature_importance.to_string(index=False))
print()

print("KEY INSIGHTS:")
top_3 = feature_importance.head(3)
for idx, row in top_3.iterrows():
    print(f"  - {row['Feature']:30s} contributes {row['Importance']*100:.1f}% to predictions")
print()

# ============================================================================
# STEP 10: TEST PREDICTIONS FOR SPECIFIC MUNICIPALITIES
# ============================================================================
print("=" * 80)
print("STEP 10: Testing predictions for specific municipalities...")
print("=" * 80)

# Create sample data for prediction
test_scenarios = []

municipalities = ['Tirupati', 'Guntur', 'Nellore']

for municipality in municipalities:
    # Get municipality code
    municipality_code = le.transform([municipality])[0]
    
    # Get average values for this municipality
    muni_data = df[df['region_name'] == municipality]
    
    scenario = {
        'temperature_celsius': 32,
        'humidity_percent': 65,
        'rainfall_mm': 0,
        'is_weekend': 0,
        'is_holiday': 0,
        'municipality_encoded': municipality_code,
        'population': int(muni_data['population'].mean()),
        'industrial_activity_index': int(muni_data['industrial_activity_index'].mean()),
        'prev_day_consumption': int(muni_data['water_consumption_liters'].mean()),
        'prev_7day_avg_consumption': int(muni_data['water_consumption_liters'].mean())
    }
    test_scenarios.append(scenario)

# Convert to DataFrame
test_df = pd.DataFrame(test_scenarios)

# Make predictions
predictions = model.predict(test_df)

# Display results
print("SAMPLE PREDICTIONS (32°C, 65% humidity, weekday):")
print()
for municipality, prediction in zip(municipalities, predictions):
    print(f"  {municipality:15s} → Predicted: {prediction/1_000_000:.1f} Million Liters")
print()

# ============================================================================
# STEP 11: SAVE THE MODEL
# ============================================================================
print("=" * 80)
print("STEP 11: Saving the model...")
print("=" * 80)

import pickle

# Save the trained model
with open('water_consumption_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as: water_consumption_model.pkl")

# Save the label encoder
with open('municipality_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✓ Label encoder saved as: municipality_encoder.pkl")

# Save feature names
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✓ Feature columns saved as: feature_columns.pkl")
print()

# ============================================================================
# STEP 12: VISUALIZATIONS
# ============================================================================
print("=" * 80)
print("STEP 12: Creating visualizations...")
print("=" * 80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Water Consumption Forecasting - Model Performance', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted (Test Set)
ax1 = axes[0, 0]
sample_size = min(1000, len(y_test))  # Plot only 1000 points for clarity
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

ax1.scatter(y_test.iloc[sample_indices]/1_000_000, 
           y_test_pred[sample_indices]/1_000_000, 
           alpha=0.5, s=20)
ax1.plot([y_test.min()/1_000_000, y_test.max()/1_000_000], 
         [y_test.min()/1_000_000, y_test.max()/1_000_000], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Consumption (Million Liters)', fontsize=12)
ax1.set_ylabel('Predicted Consumption (Million Liters)', fontsize=12)
ax1.set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Feature Importance
ax2 = axes[0, 1]
top_features = feature_importance.head(8)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax2.barh(top_features['Feature'], top_features['Importance'], color=colors)
ax2.set_xlabel('Importance Score', fontsize=12)
ax2.set_title('Top 8 Feature Importance', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Prediction Error Distribution
ax3 = axes[1, 0]
errors = (y_test - y_test_pred) / 1_000_000
ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax3.set_xlabel('Prediction Error (Million Liters)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title(f'Prediction Error Distribution\nMAE = {test_mae/1_000_000:.2f}M L', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Municipality-wise Average Consumption
ax4 = axes[1, 1]
muni_avg = df.groupby('region_name')['water_consumption_liters'].mean() / 1_000_000
muni_avg = muni_avg.sort_values(ascending=True)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(muni_avg)))
ax4.barh(muni_avg.index, muni_avg.values, color=colors)
ax4.set_xlabel('Average Consumption (Million Liters)', fontsize=12)
ax4.set_title('Average Daily Consumption by Municipality', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as: model_performance.png")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print()
print("✓ Model Training: COMPLETED")
print(f"✓ Test Accuracy (R²): {test_r2:.4f}")
print(f"✓ Average Error (MAE): {test_mae/1_000_000:.2f} Million Liters")
print()
print("FILES CREATED:")
print("  1. water_consumption_model.pkl      - Trained model")
print("  2. municipality_encoder.pkl         - Municipality encoder")
print("  3. feature_columns.pkl              - Feature names")
print("  4. model_performance.png            - Performance visualizations")
print()
print("NEXT STEPS:")
print("  1. Use the saved model to make predictions")
print("  2. Integrate with the dashboard")
print("  3. Deploy for real-time forecasting")
print()
print("=" * 80)
print("ALL DONE! 🎉")
print("=" * 80)
