# data_loader.py

import os
import datetime
import math
import requests
import requests_cache
from retry_requests import retry
import pandas as pd
import yfinance as yf
import openmeteo_requests
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import streamlit as st
from constants import GOOGLE_WEATHER_API_KEY # Ensure this is imported

from constants import (
    EIA_API_KEY,
    OPENWEATHER_API_KEY,
    SHAPEFILE_PATH,
    HDD_CITIES,
    get_storage_centroids,
    get_lng_terminals
)

# --- PRICE DATA ---
@st.cache_data(ttl=3600*24)
def get_price_data():
    """Fetches Henry Hub, TTF prices, and EUR/USD FX from Yahoo Finance."""
    tickers = ['NG=F', 'TTF=F']
    data = yf.download(tickers, period="5y", interval="1d")['Close']
    data.rename(columns={'NG=F': 'HenryHub_USD', 'TTF=F': 'TTF_EUR'}, inplace=True)

    fx = yf.download("EURUSD=X", period="5y", interval="1d")['Close']
    fx = fx.reindex(data.index).ffill()
    data['FX_EURUSD'] = fx

    # Convert TTF EUR/MWh to USD/MMBtu
    data['TTF_USD_MMBtu'] = (data['TTF_EUR'] * data['FX_EURUSD']) / 3.412
    data['Spread_TTF_HH'] = data['TTF_USD_MMBtu'] - data['HenryHub_USD']

    return data.sort_index()

# --- EIA STORAGE DATA ---
@st.cache_data(ttl=3600*24)
def get_eia_series(series_id: str, length_weeks: int = 52 * 20) -> pd.DataFrame | None:
    """Fetches a single EIA weekly storage series."""
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    params = {
        "api_key": EIA_API_KEY,
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
        r.raise_for_status()
        data = r.json()

        if 'response' in data and 'data' in data['response'] and data['response']['data']:
            df = pd.DataFrame(data['response']['data'])
            df['period'] = pd.to_datetime(df['period'])
            df['value'] = pd.to_numeric(df['value'])
            df = df.sort_values('period').reset_index(drop=True)
            return df
        else:
            return None

    except Exception:
        return None

# --- WEATHER DATA (HDD FORECAST) ---
@st.cache_data(ttl=3600*12)
def get_weather_demand():
    """Fetches 10-day hourly temperature forecast and computes daily HDD for key cities."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    cities = list(HDD_CITIES.keys())
    latitudes = [c["lat"] for c in HDD_CITIES.values()]
    longitudes = [c["lon"] for c in HDD_CITIES.values()]

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": "temperature_2m",
        "timezone": "auto",
        "forecast_days": 10,
    }

    url = "https://api.open-meteo.com/v1/forecast"
    responses = openmeteo.weather_api(url, params=params)

    results = []
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
        # HDD calculation per hour: max(0, 65 - T_hourly) / 24
        df['HDD'] = df['temp_f'].apply(lambda x: max(0, 65 - x) / 24)
        daily = df.groupby(df['date'].dt.date)['HDD'].sum().reset_index()
        daily['City'] = cities[i]
        results.append(daily)

    final_df = pd.concat(results)
    final_df.rename(columns={'date': 'date'}, inplace=True)
    return final_df

# --- NOAA 7-DAY TEMP ANOMALY ---
def rough_normal_temp(lat, month):
    """Estimate a rough normal temperature based on latitude and month."""
    base = 65 - (abs(lat) - 30) * 0.8
    seasonal = 15 * math.cos((month - 7) / 6.0 * math.pi)
    return base + seasonal

@st.cache_data(ttl=3600*3)
def get_noaa_temp_anomaly_by_region() -> pd.DataFrame:
    """Fetches NOAA forecast for storage region centroids and computes 7-day temp anomaly."""
    centroids = get_storage_centroids()
    rows = []

    for region, c in centroids.items():
        lat = c["Lat"]
        lon = c["Lon"]

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

            # NOAA forecast is often 7 days (14 periods: day/night)
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

# --- OPENWEATHER 14-DAY FORECAST ---
@st.cache_data(ttl=3600*3)
def get_openweather_forecast_for_storage_regions(
    days: int = 14
) -> pd.DataFrame:
    """Pull 14-day daily forecast from OpenWeather for each storage centroid."""
    if not OPENWEATHER_API_KEY:
        return pd.DataFrame()

    centroids = get_storage_centroids()
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
            "appid": OPENWEATHER_API_KEY,
        }

        try:
            r = requests.get(base_url, params=params, timeout=20)
            r.raise_for_status()
            js = r.json()

            if "list" not in js:
                continue

            for d in js["list"]:
                ts = d.get("dt")
                if ts is None:
                    continue

                date = datetime.datetime.utcfromtimestamp(ts).date()
                temp_day_c = d.get("temp", {}).get("day")
                temp_min_c = d.get("temp", {}).get("min")
                temp_max_c = d.get("temp", {}).get("max")
                wind_speed = d.get("speed")
                clouds = d.get("clouds")
                pop = d.get("pop")

                def c_to_f(x):
                    return None if x is None else (x * 9 / 5) + 32

                temp_day_f = c_to_f(temp_day_c)
                temp_min_f = c_to_f(temp_min_c)
                temp_max_f = c_to_f(temp_max_c)

                hdd = None
                if temp_day_f is not None:
                    hdd = max(0, 65 - temp_day_f)

                rows.append(
                    {
                        "region": region,
                        "lat": lat,
                        "lon": lon,
                        "date": date,
                        "temp_c": temp_day_c,
                        "temp_f": temp_day_f,
                        "temp_min_f": temp_min_f,
                        "temp_max_f": temp_max_f,
                        "wind_speed": wind_speed,
                        "clouds_pct": clouds,
                        "pop": pop,
                        "HDD": hdd,
                    }
                )

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["date", "region"], inplace=True)
    return df

# --- GEO DATA ---
@st.cache_data
def load_pipeline_data():
    """Loads pipeline shapefile data."""
    if not os.path.exists(SHAPEFILE_PATH):
        st.error(f"Shapefile not found: {SHAPEFILE_PATH}. Ensure .shp, .shx, .dbf, .prj are present.")
        return None, None

    try:
        pipelines_gdf = gpd.read_file(SHAPEFILE_PATH)
        minx, miny, maxx, maxy = pipelines_gdf.total_bounds
        bbox_polygon = box(minx, miny, maxx, maxy)

        boundary_gdf = gpd.GeoDataFrame(
            {"name": ["Pipeline Extent"]},
            geometry=[bbox_polygon],
            crs=pipelines_gdf.crs,
        )

        lng_df = pd.DataFrame(get_lng_terminals())

        return pipelines_gdf, boundary_gdf, lng_df

    except Exception as e:
        st.error(f"Error loading pipeline shapefile: {e}")
        st.warning("Ensure .shp, .shx, .dbf, .prj (and .cpg) are present in the app directory.")
        return None, None, pd.DataFrame()



# The base URL for the daily forecast endpoint
DAILY_FORECAST_URL = "https://weather.googleapis.com/v1/forecast/days:lookup"


def get_google_weather_forecast(lat: float, lon: float, api_key: str) -> pd.DataFrame:
    """
    Fetches the 10-day daily forecast from the Google Maps Platform Weather API
    for a given latitude and longitude.
    """
    params = {
        "location.latitude": lat,
        "location.longitude": lon,
        "key": api_key,
    }

    try:
        # Use your existing session setup for caching if available, otherwise use requests.get
        # Assuming you use a requests session setup similar to your other functions:
        session = requests.Session() # Replace with your retry-enabled session if you have one
        response = session.get(DAILY_FORECAST_URL, params=params)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Google Weather data: {e}")
        return pd.DataFrame()

    if not data or 'dailyForecasts' not in data:
        return pd.DataFrame()

    forecasts = data['dailyForecasts']
    
    # Process the response into a standardized DataFrame
    parsed_data = []
    for day in forecasts:
        # Google uses Celsius by default; you may need to convert to Fahrenheit
        # if your app expects it, or use units=imperial in the API call if supported.
        # Assuming you want the maximum temperature.
        date_str = day.get('date')
        max_temp_c = day.get('day', {}).get('maxTemperature', {}).get('value')
        
        if date_str and max_temp_c is not None:
            # Conversion from Celsius to Fahrenheit (F = C * 9/5 + 32)
            max_temp_f = (max_temp_c * 9/5) + 32
            
            parsed_data.append({
                "Date": pd.to_datetime(date_str),
                "Max_Temp_F": max_temp_f,
                # Add other useful fields like min temp, precipitation, etc.
            })

    return pd.DataFrame(parsed_data).set_index("Date").sort_index()
