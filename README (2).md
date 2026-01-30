# 💧 Water Consumption Forecasting Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered water consumption forecasting system built with **Machine Learning** and **Streamlit** to predict water demand across 10 municipalities in Andhra Pradesh, India.

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=Water+Forecasting+Dashboard)

---

## 🌟 Features

### 🎯 Core Functionality
- **Real-time Predictions**: Instant water consumption forecasts based on weather conditions
- **Multi-Municipality Support**: Predictions for 10 different cities
- **7-Day Forecasting**: Future water demand predictions with downloadable reports
- **Municipality Comparison**: Compare water needs across all cities
- **Interactive Visualizations**: Beautiful charts powered by Plotly

### 🤖 ML Model
- **Algorithm**: Random Forest Regressor
- **Accuracy**: 99.4% (R² Score)
- **Training Data**: 100,000 historical records
- **Features**: 13 input features including weather, population, and patterns

### 📊 Supported Municipalities
1. Anantapur
2. Guntur
3. Kadapa
4. Kurnool
5. Nellore
6. Prakasam
7. Srikakulam
8. Tirupati
9. Vijayawada
10. Visakhapatnam

---

## 🚀 Quick Start

### Option 1: One-Click Start (Recommended)

**Windows:**
```cmd
run_app.bat
```

**Mac/Linux:**
```bash
chmod +x run_app.sh
./run_app.sh
```

### Option 2: Manual Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📁 Project Structure

```
water-forecasting-app/
│
├── streamlit_app.py                              # Main Streamlit application
├── requirements.txt                              # Python dependencies
│
├── water_consumption_model.pkl                   # Trained ML model (3.8 MB)
├── municipality_encoder.pkl                      # Label encoder
├── feature_columns.pkl                           # Feature list
│
├── water_consumption_100000_rows_improved.csv    # Training dataset (14 MB)
│
├── run_app.sh                                    # Quick start (Mac/Linux)
├── run_app.bat                                   # Quick start (Windows)
│
├── STREAMLIT_DEPLOYMENT_GUIDE.md                 # Deployment instructions
└── README.md                                     # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone the repository** (or download ZIP)
```bash
git clone https://github.com/yourusername/water-forecasting-app.git
cd water-forecasting-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify files**
```bash
# Make sure you have all required files:
# - streamlit_app.py
# - requirements.txt
# - water_consumption_model.pkl
# - municipality_encoder.pkl
# - feature_columns.pkl
# - water_consumption_100000_rows_improved.csv
```

4. **Run the app**
```bash
streamlit run streamlit_app.py
```

---

## 📖 Usage Guide

### Making a Prediction

1. **Select Municipality** from the sidebar dropdown
2. **Set Weather Conditions**:
   - Temperature (15-45°C)
   - Humidity (20-100%)
   - Rainfall (0-50mm)
3. **Choose Day Type**: Weekday, Weekend, or Holiday
4. **Click "Get Prediction"** to see the forecast

### Viewing 7-Day Forecast

1. Navigate to the **"📅 7-Day Forecast"** tab
2. View predictions for the next 7 days
3. Download the forecast as CSV

### Comparing Municipalities

1. Go to **"🏙️ Municipality Comparison"** tab
2. See all 10 cities compared under current conditions
3. Identify which cities need more water supply

### Understanding the Model

1. Check **"📊 Model Info"** tab
2. View model accuracy and statistics
3. See feature importance rankings

---

## 🔬 Technical Details

### Machine Learning Model

**Algorithm**: Random Forest Regressor

**Input Features** (13 total):
- `temperature_celsius` - Temperature in Celsius
- `humidity_percent` - Humidity percentage
- `rainfall_mm` - Rainfall in millimeters
- `is_weekend` - Binary flag for weekend
- `is_holiday` - Binary flag for holiday
- `municipality_encoded` - Encoded municipality ID
- `population_scaled` - Normalized population
- `industrial_scaled` - Industrial activity index
- `prev_day_consumption_normalized` - Previous day consumption
- `prev_7day_avg_normalized` - 7-day average consumption
- `consumption_variance` - Consumption variance metric
- `month` - Month of the year
- `season` - Encoded season (0-3)

**Output**: Water consumption in liters (displayed in Million Liters)

**Performance Metrics**:
- R² Score: 0.994 (99.4% accuracy)
- MAE: ~2.5 Million Liters
- RMSE: ~3.2 Million Liters

### Tech Stack

- **Framework**: Streamlit 1.31.0
- **ML Library**: scikit-learn 1.3.0
- **Data Processing**: Pandas 2.1.0, NumPy 1.24.3
- **Visualization**: Plotly 5.18.0
- **Language**: Python 3.8+

---

## ☁️ Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/water-forecasting-app.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo
   - Set main file: `streamlit_app.py`
   - Click "Deploy!"

3. **Get your URL**
   - Your app will be live at: `https://your-app-name.streamlit.app`

📚 **Detailed deployment guide**: See [STREAMLIT_DEPLOYMENT_GUIDE.md](STREAMLIT_DEPLOYMENT_GUIDE.md)

---

## 📊 Example Predictions

### Scenario 1: Hot Summer Day in Tirupati
**Input:**
- Temperature: 38°C
- Humidity: 65%
- Rainfall: 0mm
- Day Type: Weekday

**Output:**
- Predicted: 162.5 Million Liters
- Average: 142.0 Million Liters
- **Change: +14.4%** ⬆️

**Insight**: High temperature increases water demand by 14.4%

### Scenario 2: Rainy Weekend in Guntur
**Input:**
- Temperature: 28°C
- Humidity: 85%
- Rainfall: 15mm
- Day Type: Weekend

**Output:**
- Predicted: 145.2 Million Liters
- Average: 158.0 Million Liters
- **Change: -8.1%** ⬇️

**Insight**: Rain and cooler weather reduce demand

---

## 🎯 Use Cases

### For Water Departments
- **Demand Forecasting**: Predict next day/week water needs
- **Resource Planning**: Optimize pumping schedules
- **Shortage Prevention**: Early warning for high-demand days
- **Cost Reduction**: Avoid over-production

### For City Planners
- **Infrastructure Planning**: Identify high-demand areas
- **Budget Allocation**: Data-driven resource distribution
- **Seasonal Patterns**: Understand consumption trends

### For Researchers
- **Climate Impact**: Study weather effects on consumption
- **Urban Analysis**: Compare cities by size and industry
- **Model Training**: Use as ML example project

---

## 🐛 Troubleshooting

### App won't start
```bash
# Check Python version (needs 3.8+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Model file not found
```bash
# Verify all .pkl files are present
ls *.pkl

# Should see:
# - water_consumption_model.pkl
# - municipality_encoder.pkl
# - feature_columns.pkl
```

### Predictions seem wrong
- Verify input ranges (temp: 15-45°C, humidity: 20-100%)
- Check municipality name spelling
- Ensure CSV file is complete

For more help, see [STREAMLIT_DEPLOYMENT_GUIDE.md](STREAMLIT_DEPLOYMENT_GUIDE.md)

---

## 📈 Future Enhancements

- [ ] Real-time weather API integration (OpenWeatherMap)
- [ ] User authentication and role-based access
- [ ] PostgreSQL database for prediction history
- [ ] Email alerts for high-demand forecasts
- [ ] Mobile app (React Native)
- [ ] Multi-language support (Telugu, Hindi, English)
- [ ] Advanced analytics dashboard
- [ ] Export to PDF reports

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Dataset: Simulated based on Andhra Pradesh municipalities
- ML Model: scikit-learn Random Forest
- UI Framework: Streamlit
- Charts: Plotly

---

## 📞 Support

- 📧 Email: your.email@example.com
- 💬 GitHub Issues: [Report a bug](https://github.com/yourusername/water-forecasting-app/issues)
- 📖 Documentation: [Streamlit Docs](https://docs.streamlit.io)

---

## ⭐ Star This Repo

If you found this project helpful, please give it a ⭐️!

---

<div align="center">
  <p>Built with ❤️ using Streamlit and Machine Learning</p>
  <p>© 2026 Water Forecasting Dashboard</p>
</div>
