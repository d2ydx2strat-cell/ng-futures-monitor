# data_loader.py

import os
import requests
import datetime
import pandas as pd
import yfinance as yf
import requests_cache
from retry_requests import retry
import openmeteo_requests
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import streamlit as st

# Import safe constants only
from constants import EIA_API_KEY, SHAPEFILE_PATH, get_storage_centroids

# --- API Endpoints ---
HOURLY_FORECAST_URL = "https://weather.googleapis.com/v1/forecast/hours:lookup"

# --- 1. Price Data ---
@st.cache_data(ttl=3600*24)
def get_price_data():
    tickers = ['NG=F', 'TTF=F']
    data = yf.download(tickers, period="5y", interval="1d")['Close']

    data.rename(columns={'NG=F': 'HenryHub_USD', 'TTF=F': 'TTF_EUR'}, inplace=True)

    fx = yf.download("EURUSD=X", period="5y", interval="1d")['Close']
    fx = fx.reindex(data.index).ffill()
    data['FX_EURUSD'] = fx

    data['TTF_USD_MMBtu'] = (data['TTF_EUR'] * data['FX_EURUSD']) / 3.412
    data['Spread_TTF_HH'] = data['TTF_USD_MMBtu'] - data['HenryHub_USD']

    return data.sort_index()

# --- 2. US Storage (EIA) ---
@st.cache_data(ttl=3600*24)
def get_eia_series(series_id: str, length_weeks: int = 52 * 20) -> pd.DataFrame | None:
    # Uses EIA_API_KEY imported from constants
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    params = {
        "api_key": EIA_API_KEY, # Use the key imported from constants
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": series_id,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": length_weeks,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        # ... (rest of parsing logic) ...
        if 'response' in data and 'data' in data['response'] and data['response']['data']:
            df = pd.DataFrame(data['response']['data'])
            df['period'] = pd.to_datetime(df['period'])
            df['value'] = pd.to_numeric(df['value'])
            df = df.sort_values('period').reset_index(drop=True)
            return df
        return None
    except Exception:
        return None

# --- 3. Weather Demand (OpenMeteo HDD) ---
@st.cache_data(ttl=3600*12)
def get_weather_demand():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Coordinates for key consumption hubs
    params = {
        "latitude": [41.85, 40.71, 29.76],
        "longitude": [-87.62, -74.00, -95.36],
        "hourly": "temperature_2m",
        "timezone": "auto",
        "forecast_days": 10,
    }

    url = "https://api.open-meteo.com/v1/forecast"
    responses = openmeteo.weather_api(url, params=params)

    cities = ["Chicago", "New York", "Houston"]
    results = []

    # ... (rest of OpenMeteo parsing logic) ...
    for i, response in enumerate(responses):
        hourly = response.Hourly()
        temp = hourly.Variables(0).ValuesAsNumpy()

        dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        df = pd.DataFrame({"date": dates, "temp_c": temp})
        df['temp_f'] = (df['temp_c'] * 9 / 5) + 32
        df['HDD'] = df['temp_f'].apply(lambda x: max(0, 65 - x) / 24)
        daily = df.groupby(df['date'].dt.date)['HDD'].sum().reset_index()
        daily['City'] = cities[i]
        results.append(daily)

    final_df = pd.concat(results)
    final_df.rename(columns={'date': 'date'}, inplace=True)
    return final_df


# --- 4. NOAA 7-Day Temp Anomaly ---
@st.cache_data(ttl=3600*3)
def get_noaa_temp_anomaly_by_region(centroids: dict) -> pd.DataFrame:
    # Implementation depends on standard library imports (math, requests)
    import math

    def rough_normal_temp(lat, month):
        base = 65 - (abs(lat) - 30) * 0.8
        seasonal = 15 * math.cos((month - 7) / 6.0 * math.pi)
        return base + seasonal

    rows = []

    for region, c in centroids.items():
        lat = c["Lat"]
        lon = c["Lon"]
        # ... (rest of NOAA API logic using requests) ...
        try:
            meta_url = f"https://api.weather.gov/points/{lat},{lon}"
            m = requests.get(meta_url, timeout=15)
            m.raise_for_status()
            meta = m.json()
            forecast_url = meta["properties"]["forecast"]

            f = requests.get(forecast_url, timeout=15)
            f.raise_for_status()
            js = f.json()
            periods = js["properties"]["periods"]

            temps = [p["temperature"] for p in periods[:14] if "temperature" in p]
            if not temps:
                continue

            forecast_mean = float(np.mean(temps))

            now = datetime.datetime.utcnow()
            normal = rough_normal_temp(lat, now.month)
            temp_bias = forecast_mean - normal

            rows.append(
                {
                    "region": region,
                    "lat": lat,
                    "lon": lon,
                    "forecast_mean_temp": forecast_mean,
                    "normal_temp_est": normal,
                    "temp_bias": temp_bias,
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)

# --- 5. OpenWeather 14-Day Forecast ---
@st.cache_data(ttl=3600*3)
def get_openweather_forecast_for_storage_regions(
    centroids: dict,
    days: int = 14
) -> pd.DataFrame:
    # Requires OPENWEATHER_API_KEY to be passed or read from os.environ by app.py
    openweather_api_key = st.secrets.get("OPENWEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY"))
    if not openweather_api_key:
        return pd.DataFrame()

    base_url = "https://api.openweathermap.org/data/2.5/forecast/daily"

    rows = []
    for region, c in centroids.items():
        lat = c["Lat"]
        lon = c["Lon"]

        params = {
            "lat": lat,
            "lon": lon,
            "cnt": days,
            "units": "metric",
            "appid": openweather_api_key,
        }
        
        # ... (rest of OpenWeather logic using requests) ...
        try:
            r = requests.get(base_url, params=params, timeout=20)
            r.raise_for_status()
            js = r.json()
            # ... (rest of parsing logic) ...
            if "list" not in js:
                continue

            for d in js["list"]:
                ts = d.get("dt")
                if ts is None:
                    continue

                date = datetime.datetime.utcfromtimestamp(ts).date()
                temp_day_c = d.get("temp", {}).get("day")
                
                # Simplified C-to-F conversion logic for brevity
                def c_to_f(x): return None if x is None else (x * 9 / 5) + 32
                temp_day_f = c_to_f(temp_day_c)
                
                rows.append({
                    "region": region,
                    "lat": lat,
                    "lon": lon,
                    "date": date,
                    "temp_f": temp_day_f,
                    "HDD": max(0, 65 - temp_day_f) if temp_day_f is not None else None,
                    # ... other weather metrics ...
                })

        except Exception:
            continue

    return pd.DataFrame(rows)


# --- 6. Google Weather Hourly Forecast (The Function that needs the key passed) ---

HOURLY_FORECAST_URL = "https://weather.googleapis.com/v1/forecast/hours:lookup" # Corrected endpoint

@st.cache_data(ttl=3600 * 3)
def get_google_weather_forecast(locations_dict: dict, api_key: str) -> pd.DataFrame:
    """
    Pulls 10-day hourly forecast from Google Weather API for specific locations.
    """
    if not api_key:
        st.error("Could not retrieve hourly forecast data. Google Weather API Key is missing.")
        return pd.DataFrame()

    rows = []
    
    # Use a generic name for regions since Google API doesn't use the EIA ones
    region_names = {
        "East": "East",
        "Midwest": "Midwest",
        "Mountain": "Mountain",
        "Pacific": "Pacific",
        "South Central Salt": "SC-Salt",
        "South Central Non-Salt": "SC-NonSalt",
    }
    
    # Iterate through all location centroids
    for region, c in locations_dict.items():
        if region not in region_names: continue
            
        lat = c["Lat"]
        lon = c["Lon"]
        
        params = {
            "key": api_key,
            "location.latitude": lat,
            "location.longitude": lon
        }

        try:
            r = requests.get(HOURLY_FORECAST_URL, params=params, timeout=20)
            r.raise_for_status() 
            js = r.json()

            if "hours" in js:
                for h in js["hours"]:
                    
                    # Convert 'hours.forecastTime' (e.g., '2025-12-14T01:00:00Z') to datetime
                    time_str = h.get("forecastTime")
                    if time_str:
                        dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    else:
                        continue
                        
                    temp_f = h.get("temp", {}).get("value")
                    
                    # Google reports in Celsius, you need to convert to Fahrenheit
                    if temp_f is not None:
                         temp_f = (temp_f * 9/5) + 32
                    
                    rows.append({
                        "Region": region_names[region],
                        "Date_Time": dt,
                        "Latitude": lat,
                        "Longitude": lon,
                        "Temperature_F": temp_f,
                        # Add other fields as needed (e.g., condition, wind)
                    })

        except requests.exceptions.HTTPError as e:
            # Catch specific HTTP errors (like 403 Forbidden or 400 Bad Request)
            st.warning(f"Google API Error for {region}: {e}")
            continue
        except requests.exceptions.RequestException:
            # Catch connection/timeout errors
            continue

    if not rows:
        st.error("Could not retrieve hourly forecast data. Check your Google Weather API Key and ensure the Weather API is enabled in your Google Cloud Project.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["Date_Time", "Region"], inplace=True)
    return df


# --- 7. Load Pipeline Data (Geo) ---

@st.cache_data
def load_pipeline_data():
    try:
        # Load pipelines from SHAPEFILE_PATH imported from constants
        pipelines_gdf = gpd.read_file(SHAPEFILE_PATH)
        
        # LNG terminals data (kept local or imported from constants if defined there)
        terminals = [
            {"Name": "Sabine Pass (Cheniere)", "Lat": 29.742, "Lon": -93.872, "Capacity_Bcfd": 4.6, "Status": "Operating"},
            # ... other terminals
        ]
        lng_df = pd.DataFrame(terminals)
        
        # Calculate bounding box (simplified)
        minx, miny, maxx, maxy = pipelines_gdf.total_bounds
        bbox_polygon = box(minx, miny, maxx, maxy)
        boundary_gdf = gpd.GeoDataFrame(
            {"name": ["Pipeline Extent"]},
            geometry=[bbox_polygon],
            crs=pipelines_gdf.crs,
        )

        return pipelines_gdf, boundary_gdf, lng_df

    except Exception:
        # Ensure proper error handling if shapefile is missing
        return None, None, None
