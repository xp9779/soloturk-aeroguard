AeroGuard Antalya: Next-Day PM2.5 Air Quality Forecasting & School Decision Support

Track: AI + Climate
Team: SOLOTURK (Solo Participant: Mustafa Kemal YILDIZLI)

Project Overview

AeroGuard is an AI-powered climate and air quality forecasting system localized for Antalya, Turkey. Designed for school administrators and parents, the application automatically detects the user's IP location, fetches meteorological and particulate data via Open-Meteo APIs, and predicts next-day PM_{2.5} levels using a RandomForestRegressor. Forecasts are evaluated against the US EPA 24-hour health threshold (35\ \mu g/m^3) to deliver actionable outdoor safety recommendations and decision support.

Key Features & Customization

⚬ Automatic IP Geolocation: Dynamically detects local coordinates to tailor environmental forecasts and UI insights to the user's region.
⚬ Advanced Feature Engineering: Incorporates lagged pollution memory (yesterday_pm2_5), precipitation, and wind speed to capture multi-day particulate accumulation and dispersion.
⚬ Actionable Decision Support: Translates regression outputs into explicit safety statuses and school activity recommendations.

Validation & Results

⚬ Validation Strategy: Chronological split (80% training, 20% testing) to prevent data leakage.
⚬ Persistence Baseline MAE: 1.94\ \mu g/m^3
⚬ Random Forest Model MAE: 1.83\ \mu g/m^3 (5.8% improvement over baseline).
⚬ Error Analysis: On September 9, 2026, the model predicted 16.41\ \mu g/m^3 against an actual spike of 44.02\ \mu g/m^3, highlighting limitations in anticipating sudden transboundary dust transport or local anomalies like wildfires without external aerosol optical depth inputs.

Installation & Running Locally

1. Install required dependencies:
   pip install -r requirements.txt
   
2. Launch the Streamlit application:
   streamlit run app.py
   
3. Open http://localhost:8501 in your web browser.

Tech Stack & Sources

⚬ Languages & Frameworks: Python, Streamlit, Pandas, NumPy, Scikit-Learn, Requests
⚬ APIs: Open-Meteo Air Quality & Historical Weather APIs, IP-API Geolocation

(Tested on Local Antalya Data)
