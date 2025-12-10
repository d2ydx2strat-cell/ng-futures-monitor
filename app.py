import os
import datetime

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import statsmodels.api as sm

# Geo imports for Section 4
import geopandas as gpd
from shapely.geometry import box

# --- CONFIGURATION ---
st.set_page_config(page_title="NG Trading Monitor", layout="wide")
st.title("🔥 Global NG Spreads, Storage & Weather Monitor")

# --- GEO ASSETS: LNG & STORAGE LOCATIONS ---
def get_lng_terminals():
    terminals = [
        {"Name": "Sabine Pass (Cheniere)", "Lat": 29.742, "Lon": -93.872, "Capacity_Bcfd": 4.6, "Status": "Operating"},
        {"Name": "Corpus Christi (Cheniere)", "Lat": 27.876, "Lon": -97.280, "Capacity_Bcfd": 2.4, "Status": "Operating"},
        {"Name": "Cameron LNG", "Lat": 29.803, "Lon": -93.303, "Capacity_Bcfd": 2.0, "Status": "Operating"},
        {"Name": "Freeport LNG", "Lat": 28.943, "Lon": -95.308, "Capacity_Bcfd": 2.4, "Status": "Operating"},
        {"Name": "Cove Point (Dominion)", "Lat": 38.386, "Lon": -76.410, "Capacity_Bcfd": 0.8, "Status": "Operating"},
        {"Name": "Elba Island", "Lat": 32.083, "Lon": -80.996, "Capacity_Bcfd": 0.35, "Status": "Operating"},
        {"Name": "Calcasieu Pass (Venture Global)", "Lat": 29.771, "Lon": -93.332, "Capacity_Bcfd": 1.5, "Status": "Operating"},
        {"Name": "Plaquemines (Venture Global)", "Lat": 29.620, "Lon": -89.920, "Capacity_Bcfd": 2.6, "Status": "Under Const."},
        {"Name": "Golden Pass", "Lat": 29.760, "Lon": -93.920, "Capacity_Bcfd": 2.4, "Status": "Under Const."},
    ]
    return pd.DataFrame(terminals)

def get_storage_centroids(storage_df_latest=None):
    centroids = {
        "East": {"Lat": 40.5, "Lon": -78.0},
        "Midwest": {"Lat": 41.0, "Lon": -88.0},
        "Mountain": {"Lat": 42.0, "Lon": -108.0},
        "Pacific": {"Lat": 38.0, "Lon": -121.0},
        "South Central Salt": {"Lat": 30.0, "Lon": -92.0},
        "South Central Non-Salt": {"Lat": 34.0, "Lon": -99.0},
    }
    return centroids

# --- CONSTANTS / KEYS ---

EIA_API_KEY = "KzzwPVmMSTVCI3pQbpL9calvF4CqGgEbwWy0qqXV"

EIA_SERIES = {
    "Lower 48 Total": "NW2_EPG0_SWO_R48_BCF",
    "East": "NW2_EPG0_SWO_R31_BCF",
    "Midwest": "NW2_EPG0_SWO_R32_BCF",
    "Mountain": "NW2_EPG0_SWO_R33_BCF",
    "Pacific": "NW2_EPG0_SWO_R34_BCF",
    "South Central Total": "NW2_EPG0_SWO_R35_BCF",
    "South Central Salt": "NW2_EPG0_SSO_R33_BCF",
    "South Central Non-Salt": "NW2_EPG0_SNO_R33_BCF",
}

REGION_CAPACITY_BCF = {k: None for k in EIA_SERIES.keys()}

SHAPEFILE_PATH = "Natural_Gas_Interstate_and_Intrastate_Pipelines.shp"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    if st.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    show_noaa = st.checkbox("Show NOAA 7‑Day Temp Anomaly on Map", value=True)

# --- 1. DATA SOURCE: PRICES (International Spreads) ---
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

# --- 2. DATA SOURCE: US STORAGE (EIA) ---
@st.cache_data(ttl=3600*24)
def get_eia_series(api_key: str, series_id: str, length_weeks: int = 52 * 20) -> pd.DataFrame | None:
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    params = {
        "api_key": api_key,
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

# --- 3. DATA SOURCE: WEATHER (HDD Forecast) ---
@st.cache_data(ttl=3600*12)
def get_weather_demand():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

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

# --- 3B. NOAA 7‑DAY TEMP ANOMALY BY REGION (api.weather.gov) ---
@st.cache_data(ttl=3600*3)
def get_noaa_temp_anomaly_by_region(centroids: dict) -> pd.DataFrame:
    import math

    def rough_normal_temp(lat, month):
        base = 65 - (abs(lat) - 30) * 0.8
        seasonal = 15 * math.cos((month - 7) / 6.0 * math.pi)
        return base + seasonal

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

# --- 3C. OPENWEATHER 14-DAY FORECAST FOR STORAGE REGIONS (for Map Overlay) ---

OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY", None))

@st.cache_data(ttl=3600*3)
def get_openweather_forecast_for_storage_regions(
    centroids: dict,
    days: int = 14
) -> pd.DataFrame:
    """
    Pull 14-day daily forecast from OpenWeather for each storage centroid.
    Returns a long DataFrame with one row per region per day.
    Uses the legacy /forecast/daily endpoint; adjust if you are on One Call 3.0.
    """
    if not OPENWEATHER_API_KEY:
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
            "units": "metric",  # convert to F later
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
                pop = d.get("pop")  # precip prob (0–1)

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

# --- STORAGE ANALYTICS TRANSFORMS ---
def compute_storage_analytics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('period').reset_index(drop=True)

    df['week_of_year'] = df['period'].dt.isocalendar().week.astype(int)
    df['year'] = df['period'].dt.year
    df['delta'] = df['value'].diff()

    start_year = df['year'].min() + 5
    df = df[df['year'] >= start_year].copy()
    
    grouped = df.groupby('week_of_year')

    delta_mean = grouped['delta'].mean()
    delta_std = grouped['delta'].std(ddof=0)
    level_mean = grouped['value'].mean()
    level_std = grouped['value'].std(ddof=0)

    df['level_5y_avg'] = df['week_of_year'].map(level_mean)
    df['delta_5y_avg'] = df['week_of_year'].map(delta_mean)
    df['delta_dev_vs_5y'] = df['delta'] - df['delta_5y_avg']

    def _safe_z(row, mean_series, std_series, col_name):
        w = row['week_of_year']
        if w not in std_series.index:
            return np.nan
        std = std_series.loc[w]
        if std is None or np.isnan(std) or std == 0:
            return np.nan
        val = row[col_name]
        mean = mean_series.loc[w]
        return (val - mean) / std

    df['delta_zscore'] = df.apply(
        lambda r: _safe_z(r, delta_mean, delta_std, 'delta'),
        axis=1
    )

    df['level_zscore'] = df.apply(
        lambda r: _safe_z(r, level_mean, level_std, 'value'),
        axis=1
    )

    df['gas_year'] = np.where(df['period'].dt.month >= 4, df['period'].dt.year, df['period'].dt.year - 1)
    df['cum_dev_vs_5y'] = df.groupby('gas_year')['delta_dev_vs_5y'].cumsum()

    percentiles = grouped['value'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack(level=1)
    percentiles.columns = ['p10', 'p25', 'p50', 'p75', 'p90']
    df =
