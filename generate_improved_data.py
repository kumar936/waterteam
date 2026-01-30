"""
IMPROVED SYNTHETIC DATA GENERATOR
Creates realistic water consumption data with strong municipality differentiators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING IMPROVED SYNTHETIC DATA WITH MUNICIPALITY DIFFERENTIATORS")
print("="*80)
print()

# Set random seed for reproducibility
np.random.seed(42)

# Define municipalities with characteristics
municipalities_config = {
    'Tirupati': {
        'population': 271958,
        'industrial_activity': 3,
        'region_type': 'Urban',
        'base_consumption_per_capita': 0.45,  # ML per person per day
        'climate_factor': 1.2,  # Hot climate = higher consumption
        'seasonal_variation': [1.3, 0.9, 0.8, 1.1]  # Winter, Summer, Monsoon, Spring
    },
    'Guntur': {
        'population': 741723,
        'industrial_activity': 1,
        'region_type': 'Urban',
        'base_consumption_per_capita': 0.42,
        'climate_factor': 1.1,
        'seasonal_variation': [1.2, 0.95, 0.85, 1.1]
    },
    'Nellore': {
        'population': 421836,
        'industrial_activity': 2,
        'region_type': 'Urban',
        'base_consumption_per_capita': 0.44,
        'climate_factor': 1.15,
        'seasonal_variation': [1.25, 0.92, 0.82, 1.08]
    },
    'Vijayawada': {
        'population': 268451,
        'industrial_activity': 1,
        'region_type': 'Urban',
        'base_consumption_per_capita': 0.43,
        'climate_factor': 1.12,
        'seasonal_variation': [1.22, 0.93, 0.83, 1.09]
    },
    'Kakinada': {
        'population': 1189577,
        'industrial_activity': 5,
        'region_type': 'Urban',
        'base_consumption_per_capita': 0.50,  # High industrial activity
        'climate_factor': 1.18,
        'seasonal_variation': [1.28, 0.88, 0.78, 1.12]
    },
    'Vizianagaram': {
        'population': 981691,
        'industrial_activity': 4,
        'region_type': 'Semi-Urban',
        'base_consumption_per_capita': 0.48,
        'climate_factor': 1.10,
        'seasonal_variation': [1.2, 0.95, 0.85, 1.1]
    },
    'Kadapa': {
        'population': 675581,
        'industrial_activity': 4,
        'region_type': 'Semi-Urban',
        'base_consumption_per_capita': 0.46,
        'climate_factor': 1.22,  # Very hot climate
        'seasonal_variation': [1.35, 0.85, 0.75, 1.15]
    },
    'Kurnool': {
        'population': 617923,
        'industrial_activity': 3,
        'region_type': 'Semi-Urban',
        'base_consumption_per_capita': 0.44,
        'climate_factor': 1.25,  # Very hot
        'seasonal_variation': [1.38, 0.82, 0.72, 1.18]
    },
    'Rajahmundry': {
        'population': 618714,
        'industrial_activity': 1,
        'region_type': 'Rural',
        'base_consumption_per_capita': 0.40,
        'climate_factor': 1.08,
        'seasonal_variation': [1.15, 0.98, 0.88, 1.08]
    },
    'Anantapur': {
        'population': 378623,
        'industrial_activity': 2,
        'region_type': 'Rural',
        'base_consumption_per_capita': 0.38,
        'climate_factor': 1.28,  # Hottest region
        'seasonal_variation': [1.40, 0.80, 0.70, 1.20]
    }
}

# Generate dates (2019-2026, ~7 years)
start_date = datetime(2019, 1, 1)
end_date = datetime(2026, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

print(f"Generating data from {start_date.date()} to {end_date.date()}")
print(f"Total days: {len(date_range)}")
print()

# Generate data
data = []
seasons = ['Winter', 'Summer', 'Monsoon', 'Spring']
season_months = {
    'Winter': [12, 1, 2],
    'Summer': [3, 4, 5],
    'Monsoon': [6, 7, 8, 9],
    'Spring': [10, 11]
}

print("Generating data for municipalities:")
print("-"*80)

for municipality_name, config in municipalities_config.items():
    print(f"  {municipality_name:15s} - Pop: {config['population']:>10,} | Industry: {config['industrial_activity']}/5 | Climate: {config['climate_factor']:.2f}x")
    
    for date in date_range:
        # Determine season
        month = date.month
        season = None
        for s, months in season_months.items():
            if month in months:
                season = s
                break
        
        # Get season index
        season_idx = seasons.index(season)
        seasonal_factor = config['seasonal_variation'][season_idx]
        
        # Base consumption (scaled by population)
        base_daily = config['population'] * config['base_consumption_per_capita']
        
        # Apply factors
        climate_adjusted = base_daily * config['climate_factor']
        seasonal_adjusted = climate_adjusted * seasonal_factor
        
        # Add industrial activity factor (industries use water continuously)
        industrial_factor = 1 + (config['industrial_activity'] * 0.08)
        industrial_adjusted = seasonal_adjusted * industrial_factor
        
        # Add weather-based variation
        temp_variation = np.random.normal(0, 1)  # Weather randomness
        day_of_week = date.weekday()
        weekly_factor = 1.05 if day_of_week >= 5 else 1.0  # Weekend slightly higher
        
        # Random weather
        temperature = 25 + np.random.normal(0, 5)  # Base 25°C, std 5
        humidity = 60 + np.random.normal(0, 15)
        rainfall = max(0, np.random.normal(2, 10))  # Usually 0-3mm, sometimes more
        
        # Rainfall reduces consumption (people use less when raining)
        rainfall_factor = max(0.85, 1 - (rainfall * 0.05))
        
        # Final consumption
        consumption = industrial_adjusted * weekly_factor * rainfall_factor * (1 + temp_variation * 0.05)
        consumption = max(10, consumption)  # Minimum 10M L per day
        
        # Heat index
        heat_index = temperature + (0.5555 * humidity - 61.8)
        
        # Previous day consumption (for features)
        prev_day = consumption * np.random.normal(1, 0.1)
        prev_7day = consumption * np.random.normal(1, 0.05)
        
        # Construct row
        row = {
            'date': date.strftime('%Y-%m-%d'),
            'day_of_week': date.strftime('%A'),
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_holiday': 1 if (date.month == 1 and date.day == 1) or (date.month == 12 and date.day == 25) else 0,
            'month': date.month,
            'season': season,
            'temperature_celsius': temperature,
            'max_temperature_celsius': temperature + abs(np.random.normal(0, 3)),
            'min_temperature_celsius': temperature - abs(np.random.normal(0, 3)),
            'humidity_percent': max(20, min(95, humidity)),
            'rainfall_mm': rainfall,
            'wind_speed_kmph': abs(np.random.normal(8, 3)),
            'heat_index': heat_index,
            'region_name': municipality_name,
            'state': 'Andhra Pradesh',
            'area_type': config['region_type'],
            'population': config['population'],
            'number_of_households': config['population'] // 4,
            'industrial_activity_index': config['industrial_activity'],
            'water_consumption_liters': consumption * 1_000_000,  # Convert to liters
            'prev_day_consumption': prev_day * 1_000_000,
            'prev_7day_avg_consumption': prev_7day * 1_000_000,
            'prev_30day_avg_consumption': consumption * 0.95 * 1_000_000,
            'same_day_last_year_consumption': consumption * np.random.normal(1, 0.15) * 1_000_000,
            'water_supplied_liters': consumption * 1_000_000 * 1.05,
            'water_loss_percentage': np.random.normal(10, 2),
            'reservoir_level_percent': np.random.normal(70, 15),
            'pipe_maintenance_flag': np.random.choice([0, 1], p=[0.95, 0.05]),
            'supply_restriction_flag': 1 if rainfall > 20 else np.random.choice([0, 1], p=[0.98, 0.02]),
            'demand_supply_gap': np.random.normal(-5, 10),
            'consumption_per_capita': (consumption * 1_000_000) / config['population'],
            'temperature_change_from_yesterday': np.random.normal(0, 2),
            'rainfall_last_3_days': rainfall * np.random.normal(1, 0.5),
            'input_temperature': temperature,
            'input_humidity': max(20, min(95, humidity)),
            'input_rainfall': rainfall,
            'input_day_type': 'Weekend' if day_of_week >= 5 else ('Holiday' if (date.month == 1 and date.day == 1) or (date.month == 12 and date.day == 25) else 'Weekday'),
            'scenario_temperature_change': np.random.normal(0, 2)
        }
        
        data.append(row)

print()
print(f"Generated {len(data)} rows")
print()

# Convert to DataFrame
df = pd.DataFrame(data)

# Reorder columns to match original structure
column_order = [
    'date', 'day_of_week', 'is_weekend', 'is_holiday', 'month', 'season',
    'temperature_celsius', 'max_temperature_celsius', 'min_temperature_celsius',
    'humidity_percent', 'rainfall_mm', 'wind_speed_kmph', 'heat_index',
    'region_name', 'state', 'area_type', 'population', 'number_of_households',
    'industrial_activity_index', 'water_consumption_liters', 'prev_day_consumption',
    'prev_7day_avg_consumption', 'prev_30day_avg_consumption', 'same_day_last_year_consumption',
    'water_supplied_liters', 'water_loss_percentage', 'reservoir_level_percent',
    'pipe_maintenance_flag', 'supply_restriction_flag', 'demand_supply_gap',
    'consumption_per_capita', 'temperature_change_from_yesterday', 'rainfall_last_3_days',
    'input_temperature', 'input_humidity', 'input_rainfall', 'input_day_type',
    'scenario_temperature_change'
]

df = df[column_order]

# Save to CSV
output_file = 'water_consumption_100000_rows_improved.csv'
df.to_csv(output_file, index=False)

print("="*80)
print("DATA GENERATION STATISTICS")
print("="*80)
print()

# Analyze the generated data
print("CONSUMPTION BY MUNICIPALITY (Improved Data):")
print("-"*80)
muni_stats = df.groupby('region_name')['water_consumption_liters'].agg([
    'count', 'mean', 'min', 'max', 'std'
]).sort_values('mean', ascending=False)

muni_stats['mean_ml'] = muni_stats['mean'] / 1_000_000
muni_stats['min_ml'] = muni_stats['min'] / 1_000_000
muni_stats['max_ml'] = muni_stats['max'] / 1_000_000

print(f"{'Municipality':<15} | {'Count':<6} | {'Avg ML':<10} | {'Min ML':<10} | {'Max ML':<10}")
print("-"*80)
for muni, row in muni_stats.iterrows():
    config = municipalities_config[muni]
    print(f"{muni:<15} | {int(row['count']):<6} | {row['mean_ml']:<10.2f} | {row['min_ml']:<10.2f} | {row['max_ml']:<10.2f} | Pop: {config['population']:>10,}")

print()
print("VARIANCE ANALYSIS:")
print("-"*80)
variance_between = muni_stats['mean_ml'].var()
variance_within = (muni_stats['std'] / 1_000_000).mean() ** 2

print(f"Highest avg: {muni_stats['mean_ml'].max():.2f}M L ({muni_stats['mean_ml'].idxmax()})")
print(f"Lowest avg: {muni_stats['mean_ml'].min():.2f}M L ({muni_stats['mean_ml'].idxmin()})")
print(f"Difference: {muni_stats['mean_ml'].max() - muni_stats['mean_ml'].min():.2f}M L ({((muni_stats['mean_ml'].max() - muni_stats['mean_ml'].min()) / muni_stats['mean_ml'].mean() * 100):.1f}%)")
print()
print(f"Variance between municipalities: {variance_between:.4f}")
print(f"Variance within municipalities: {variance_within:.4f}")
print(f"Ratio: {variance_between / variance_within:.4f}x")
print()

# Population vs Consumption correlation
pop_data = df.groupby('region_name').agg({
    'population': 'first',
    'water_consumption_liters': 'mean'
})
correlation = pop_data['population'].corr(pop_data['water_consumption_liters'])
print(f"Correlation (Population vs Consumption): {correlation:.4f}")
print(f"(Closer to 1.0 = Strong relationship)")
print()

print("="*80)
print(f"SAVED: {output_file}")
print("="*80)
print()
print("NEXT STEPS:")
print("1. Update the model training to use the new file")
print("2. Run retrain_model_improved.py with the new data")
print("3. Restart API server")
print()
