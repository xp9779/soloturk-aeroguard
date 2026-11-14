import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Page Configuration
st.set_page_config(page_title="AeroGuard AI - Climate & Air Quality Forecasting", page_icon="🌍", layout="wide")

st.title("🌍 AeroGuard: Smart Air Quality & School Decision Support System")
st.markdown("This tool **automatically detects your IP location** to forecast next-day $PM_{2.5}$ air pollution levels and generates actionable safety warnings for schools and families.")

@st.cache_data
def get_user_location():
    """Automatic city and coordinate detection via IP address"""
    try:
        res = requests.get("http://ip-api.com/json/", timeout=5).json()
        if res["status"] == "success":
            return res["city"], res["lat"], res["lon"], res["country"]
    except:
        pass
    return "Antalya", 36.8969, 30.7133, "Turkey"

city, lat, lon, country = get_user_location()

st.sidebar.header("📍 Location & Settings")
st.sidebar.success(f"Detected Location:\n**{city}, {country}**\n\nCoordinates: `{lat}, {lon}`")

@st.cache_data
def load_data(lat, lon):
    start_date = "2025-05-01"
    end_date = "2026-11-13"
    
    aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5,pm10&start_date={start_date}&end_date={end_date}"
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&start_date={start_date}&end_date={end_date}"
    
    aq_data = requests.get(aq_url).json()
    weather_data = requests.get(weather_url).json()
    
    df_aq = pd.DataFrame({
        "time": pd.to_datetime(aq_data["hourly"]["time"]),
        "pm2_5": aq_data["hourly"]["pm2_5"],
        "pm10": aq_data["hourly"]["pm10"]
    })
    
    df_weather = pd.DataFrame({
        "time": pd.to_datetime(weather_data["hourly"]["time"]),
        "temp": weather_data["hourly"]["temperature_2m"],
        "humidity": weather_data["hourly"]["relative_humidity_2m"],
        "rain": weather_data["hourly"]["precipitation"],
        "wind_speed": weather_data["hourly"]["wind_speed_10m"]
    })
    
    df = pd.merge(df_aq, df_weather, on="time").dropna()
    df["date"] = df["time"].dt.date
    
    daily_df = df.groupby("date").agg({
        "pm2_5": "mean",
        "pm10": "mean",
        "temp": "mean",
        "humidity": "mean",
        "rain": "sum",
        "wind_speed": "mean"
    }).reset_index()
    
    daily_df["target_tomorrow_pm2_5"] = daily_df["pm2_5"].shift(-1)
    daily_df["yesterday_pm2_5"] = daily_df["pm2_5"].shift(1)
    return daily_df.dropna().reset_index(drop=True)

with st.spinner(f"Loading weather and pollution history for {city}..."):
    daily_df = load_data(lat, lon)

split_idx = int(len(daily_df) * 0.8)
train_df = daily_df.iloc[:split_idx]
test_df = daily_df.iloc[split_idx:].copy()

features = ["pm2_5", "pm10", "temp", "humidity", "rain", "wind_speed", "yesterday_pm2_5"]
X_train, y_train = train_df[features], train_df["target_tomorrow_pm2_5"]
X_test, y_test = test_df[features], test_df["target_tomorrow_pm2_5"]

baseline_preds = test_df["pm2_5"]
baseline_mae = mean_absolute_error(y_test, baseline_preds)

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
model.fit(X_train, y_train)
model_preds = model.predict(X_test)
model_mae = mean_absolute_error(y_test, model_preds)

tab1, tab2, tab3 = st.tabs(["🚀 Live Forecast & Decision", "📊 Model Performance & Baseline", "⚠️ Error Analysis"])

with tab1:
    st.subheader(f"Next-Day Air Quality Forecast for {city}")
    latest_day = daily_df.iloc[-1:]
    tomorrow_forecast = model.predict(latest_day[features])[0]
    
    col1, col2 = st.columns(2)
    col1.metric("Today's Average PM2.5", f"{latest_day['pm2_5'].values[0]:.2f} µg/m³")
    col2.metric("Tomorrow's Forecasted PM2.5", f"{tomorrow_forecast:.2f} µg/m³", delta=f"{tomorrow_forecast - latest_day['pm2_5'].values[0]:.2f}")
    
    EPA_THRESHOLD = 35.0
    if tomorrow_forecast > EPA_THRESHOLD:
        st.error(f"🚨 **STATUS: UNHEALTHY / AT RISK (Above US EPA 35 µg/m³ Threshold!)**\n\n**Recommendation:** Schools should move physical education classes indoors. Sensitive individuals should limit outdoor exposure.")
    else:
        st.success(f"✅ **STATUS: SAFE (Below US EPA 35 µg/m³ Threshold)**\n\n**Recommendation:** Air quality is suitable for outdoor activities and school sports.")
        
    st.markdown("---")
    st.subheader("Historical PM2.5 Trend Chart")
    st.line_chart(daily_df.set_index("date")["pm2_5"].tail(60))

with tab2:
    st.subheader("AI vs. Baseline Performance Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Error (MAE)", f"{baseline_mae:.2f} µg/m³")
    col2.metric("Random Forest Model Error (MAE)", f"{model_mae:.2f} µg/m³")
    improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
    col3.metric("Improvement Rate", f"%{improvement:.1f}")

with tab3:
    st.subheader("Where Does It Fail? (Error Analysis)")
    test_df["predicted"] = model_preds
    test_df["abs_error"] = np.abs(test_df["target_tomorrow_pm2_5"] - test_df["predicted"])
    worst_case = test_df.loc[test_df["abs_error"].idxmax()]
    
    st.warning(f"**Date with Largest Prediction Error:** {worst_case['date']}")
    st.write(f"- **Actual Value:** {worst_case['target_tomorrow_pm2_5']:.2f} µg/m³")
    st.write(f"- **Model Prediction:** {worst_case['predicted']:.2f} µg/m³")
    st.write(f"- **Absolute Error / Deviation:** {worst_case['abs_error']:.2f} µg/m³")
    st.markdown("**Why did it fail?** Because the model relies strictly on local weather variables and historical trends, it cannot anticipate sudden external anomalies such as transboundary dust storms (e.g., Saharan dust) or local wildfires.")
