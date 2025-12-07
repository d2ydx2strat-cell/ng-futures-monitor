import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import datetime
import openmeteo_requests
import requests_cache
from retry_requests import retry
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
st.set_page_config(page_title="NG Trading Monitor", layout="wide")
st.title("🔥 Global NG Spreads, Storage & Weather Monitor")

# Sidebar for API Keys
with st.sidebar:
    st.header("Settings")
    EIA_API_KEY = st.text_input("Enter EIA API Key", type="password")
    st.info("Get key at: https://www.eia.gov/opendata/")
    
    if st.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# --- 1. DATA SOURCE: PRICES (International Spreads) ---
@st.cache_data(ttl=3600*24) # Cache for 24 hours
def get_price_data():
    """
    Fetches Henry Hub (US), and proxies for TTF (EU) and JKM (Asia).
    Note: True TTF/JKM futures require paid APIs (e.g., Databento/Barchart).
    We use tickers that are commonly available or proxies.
    """
    # NG=F is Henry Hub Futures
    # TTF=F is often the ticker for Dutch TTF on Yahoo (data may be delayed/limited)
    # JKM is harder to get free; we often monitor the spread via ETF proxies or scrape if needed.
    # For this demo, we fetch Henry Hub and TTF.
    
    tickers = ['NG=F', 'TTF=F'] 
    data = yf.download(tickers, period="1y", interval="1d")['Close']
    
    # Clean up column names
    data.rename(columns={'NG=F': 'HenryHub_USD', 'TTF=F': 'TTF_EUR'}, inplace=True)
    
    # Currency Conversion (Approximate EUR to USD for Spread)
    # Getting live EURUSD rate
    fx = yf.download("EURUSD=X", period="1d", interval="1d")['Close'].iloc[-1].item()
    
    # Convert TTF (usually in EUR/MWh) to USD/MMBtu
    # Conversion factor: 1 MWh = 3.412 MMBtu
    # Formula: (Price EUR/MWh * FX Rate) / 3.412
    data['TTF_USD_MMBtu'] = (data['TTF_EUR'] * fx) / 3.412
    
    # Calculate Spread (Arb Window)
    data['Spread_TTF_HH'] = data['TTF_USD_MMBtu'] - data['HenryHub_USD']
    
    return data.sort_index(ascending=False)

# --- 2. DATA SOURCE: US STORAGE (EIA) ---
@st.cache_data(ttl=3600*24)
def get_eia_storage(api_key):
    if not api_key:
        return None
    
    # EIA API v2 Endpoint for Natural Gas Storage
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "NG.NW2_EPG0_SWO_R48_BCF.W", # Lower 48 Total Storage
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 52 * 5 # Last 5 years
    }
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            df['period'] = pd.to_datetime(df['period'])
            df['value'] = pd.to_numeric(df['value'])
            df = df.sort_values('period')
            return df
        else:
            st.error("EIA Data structure changed or Key Invalid")
            return None
    except Exception as e:
        st.error(f"EIA Fetch Error: {e}")
        return None

# --- 3. DATA SOURCE: WEATHER (HDD Forecast) ---
@st.cache_data(ttl=3600*12) # Update weather every 12 hours
def get_weather_demand():
    """
    Uses Open-Meteo to fetch 7-day forecast for key gas-consuming hubs:
    Chicago (Midwest), New York (East), Houston (South - CDD).
    Calculates proxy HDD (Heating Degree Days).
    """
    # Setup Open-Meteo Client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Locations: Chicago, NYC, Houston
    params = {
        "latitude": [41.85, 40.71, 29.76],
        "longitude": [-87.62, -74.00, -95.36],
        "hourly": "temperature_2m",
        "timezone": "auto",
        "forecast_days": 10
    }
    
    url = "https://api.open-meteo.com/v1/forecast"
    responses = openmeteo.weather_api(url, params=params)
    
    cities = ["Chicago", "New York", "Houston"]
    results = []

    for i, response in enumerate(responses):
        hourly = response.Hourly()
        temp = hourly.Variables(0).ValuesAsNumpy()
        
        # Create timestamps
        dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        
        df = pd.DataFrame({"date": dates, "temp_c": temp})
        
        # Convert C to F
        df['temp_f'] = (df['temp_c'] * 9/5) + 32
        
        # Calculate HDD (Base 65F)
        df['HDD'] = df['temp_f'].apply(lambda x: max(0, 65 - x) / 24) # Divide by 24 for hourly -> daily contrib
        
        # Aggregate by Day
        daily = df.groupby(df['date'].dt.date)['HDD'].sum().reset_index()
        daily['City'] = cities[i]
        results.append(daily)
        
    final_df = pd.concat(results)
    return final_df

# --- MAIN DASHBOARD LOGIC ---

# 1. Prices & Spreads
st.subheader("1. International Future Spreads (Arbitrage Window)")
try:
    price_df = get_price_data()
    latest = price_df.iloc[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Henry Hub (US)", f"${latest['HenryHub_USD']:.2f}", f"{price_df['HenryHub_USD'].diff().iloc[0]:.2f}")
    col2.metric("TTF Proxy (EU)", f"${latest['TTF_USD_MMBtu']:.2f}", f"{price_df['TTF_USD_MMBtu'].diff().iloc[0]:.2f}")
    col3.metric("Spread (Export Arb)", f"${latest['Spread_TTF_HH']:.2f}", "High spread = Bullish US LNG")
    
    # Chart
    fig_price = make_subplots(specs=[[{"secondary_y": True}]])
    fig_price.add_trace(go.Scatter(x=price_df.index, y=price_df['HenryHub_USD'], name="Henry Hub ($)"), secondary_y=False)
    fig_price.add_trace(go.Scatter(x=price_df.index, y=price_df['TTF_USD_MMBtu'], name="TTF EU ($/MMBtu)"), secondary_y=True)
    st.plotly_chart(fig_price, use_container_width=True)
    
except Exception as e:
    st.warning(f"Could not load price data (Yahoo Finance might be throttling): {e}")

st.markdown("---")

# 2. Storage
st.subheader("2. US Storage Levels (EIA Weekly)")
if EIA_API_KEY:
    storage_df = get_eia_storage(EIA_API_KEY)
    if storage_df is not None:
        # Calculate 5-Year Average for the current week of year
        storage_df['week_of_year'] = storage_df['period'].dt.isocalendar().week
        five_yr_avg = storage_df.groupby('week_of_year')['value'].mean()
        
        latest_storage = storage_df.iloc[-1]
        current_week = latest_storage['week_of_year']
        avg_for_week = five_yr_avg.loc[current_week]
        deficit = latest_storage['value'] - avg_for_week
        
        s_col1, s_col2 = st.columns(2)
        s_col1.metric("Current Storage (Bcf)", f"{latest_storage['value']:,}", delta=f"{deficit:,.0f} vs 5yr Avg")
        
        # Plot
        fig_store = go.Figure()
        fig_store.add_trace(go.Scatter(x=storage_df['period'], y=storage_df['value'], name="Storage Level"))
        st.plotly_chart(fig_store, use_container_width=True)
else:
    st.warning("⚠️ Enter EIA API Key in sidebar to see Storage Data")

st.markdown("---")

# 3. Weather
st.subheader("3. 10-Day HDD Forecast (Gas Demand Proxy)")
st.write("Projected Heating Degree Days (HDD) for key consumption hubs.")
try:
    weather_df = get_weather_demand()
    
    # Pivot for chart
    chart_data = weather_df.pivot(index='date', columns='City', values='HDD')
    
    st.line_chart(chart_data)
    
    # Total System HDD
    total_hdd = chart_data.sum(axis=1)
    st.metric("Total System Forecast HDD (Next 10 Days)", f"{total_hdd.sum():.0f}")
    
except Exception as e:
    st.error(f"Weather data error: {e}")