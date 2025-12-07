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
import numpy as np

# --- CONFIGURATION ---
import streamlit as st
import pandas as pd
st.set_page_config(page_title="NG Trading Monitor", layout="wide")
st.title("🔥 Global NG Spreads, Storage & Weather Monitor")
EIA_API_KEY = "KzzwPVmMSTVCI3pQbpL9calvF4CqGgEbwWy0qqXV"
WORKING_SERIES_ID = "NW2_EPG0_SWO_R48_BCF"  # Lower 48 working gas

# Sidebar for API Keys
with st.sidebar:
    st.header("Settings")
    if st.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# --- 1. DATA SOURCE: PRICES (International Spreads) ---
@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def get_price_data():
    """
    Fetches Henry Hub (US), and proxies for TTF (EU).
    """
    tickers = ['NG=F', 'TTF=F']
    data = yf.download(tickers, period="1y", interval="1d")['Close']

    # Clean up column names
    data.rename(columns={'NG=F': 'HenryHub_USD', 'TTF=F': 'TTF_EUR'}, inplace=True)

    # Currency Conversion (Approximate EUR to USD for Spread)
    fx = yf.download("EURUSD=X", period="1d", interval="1d")['Close'].iloc[-1].item()

    # Convert TTF (usually in EUR/MWh) to USD/MMBtu
    # Conversion factor: 1 MWh = 3.412 MMBtu
    data['TTF_USD_MMBtu'] = (data['TTF_EUR'] * fx) / 3.412

    # Calculate Spread (Arb Window)
    data['Spread_TTF_HH'] = data['TTF_USD_MMBtu'] - data['HenryHub_USD']

    return data.sort_index(ascending=False)

# --- 2. DATA SOURCE: US STORAGE (EIA) ---
@st.cache_data(ttl=3600*24)
def get_eia_storage(api_key):
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": WORKING_SERIES_ID,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 52 * 15  # pull more history for better stats
    }

    try:
        r = requests.get(url, params=params)
        data = r.json()

        if 'error' in data:
            st.error(f"EIA API Error: {data['error']}")
            return None

        if 'response' in data and 'data' in data['response'] and data['response']['data']:
            df = pd.DataFrame(data['response']['data'])

            df['period'] = pd.to_datetime(df['period'])
            df['value'] = pd.to_numeric(df['value'])

            df = df.sort_values('period').reset_index(drop=True)
            return df
        else:
            st.error("EIA Structure Error: API returned a valid structure but the data array is empty [].")
            return None

    except Exception as e:
        st.error(f"EIA Fetch Error: {e}")
        return None

# --- 3. DATA SOURCE: WEATHER (HDD Forecast) ---
@st.cache_data(ttl=3600*12)  # Update weather every 12 hours
def get_weather_demand():
    """
    Uses Open-Meteo to fetch 10-day forecast for key gas-consuming hubs:
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
        df['HDD'] = df['temp_f'].apply(lambda x: max(0, 65 - x) / 24)  # hourly -> daily contrib

        # Aggregate by Day
        daily = df.groupby(df['date'].dt.date)['HDD'].sum().reset_index()
        daily['City'] = cities[i]
        results.append(daily)

    final_df = pd.concat(results)
    return final_df

# --- HELPER: STORAGE ANALYTICS TRANSFORMS ---
def compute_storage_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a storage_df with columns ['period', 'value'],
    compute weekly deltas, 5y avg, deviation, z-score, cumulative deviation,
    and percentile bands for fan chart.
    """
    df = df.copy()
    df = df.sort_values('period').reset_index(drop=True)

    # Week-of-year (ISO)
    df['week_of_year'] = df['period'].dt.isocalendar().week.astype(int)
    df['year'] = df['period'].dt.year

    # Weekly injection/withdrawal (delta)
    df['delta'] = df['value'].diff()

    # 5-year window for stats (rolling by calendar week)
    # Filter to last N years for "current" stats; use more history for fan chart
    # Here we use all history for stats by week_of_year
    grouped = df.groupby('week_of_year')

    # 5-year average level by week_of_year (using all history as proxy)
    df['level_5y_avg'] = df['week_of_year'].map(grouped['value'].mean())

    # Weekly delta stats by week_of_year
    delta_mean = grouped['delta'].mean()
    delta_std = grouped['delta'].std(ddof=0)

    df['delta_5y_avg'] = df['week_of_year'].map(delta_mean)
    df['delta_dev_vs_5y'] = df['delta'] - df['delta_5y_avg']
    df['delta_zscore'] = df.apply(
        lambda row: (row['delta'] - delta_mean.loc[row['week_of_year']]) / delta_std.loc[row['week_of_year']]
        if (row['week_of_year'] in delta_std.index and delta_std.loc[row['week_of_year']] not in [0, np.nan])
        else np.nan,
        axis=1
    )

    # Cumulative deviation vs 5y avg delta, by "gas year" (start April)
    # Define gas year starting April 1
    df['gas_year'] = np.where(df['period'].dt.month >= 4, df['period'].dt.year, df['period'].dt.year - 1)
    df['cum_dev_vs_5y'] = df.groupby('gas_year')['delta_dev_vs_5y'].cumsum()

    # Level z-score by week_of_year
    level_mean = grouped['value'].mean()
    level_std = grouped['value'].std(ddof=0)
    df['level_zscore'] = df.apply(
        lambda row: (row['value'] - level_mean.loc[row['week_of_year']]) / level_std.loc[row['week_of_year']]
        if (row['week_of_year'] in level_std.index and level_std.loc[row['week_of_year']] not in [0, np.nan])
        else np.nan,
        axis=1
    )

    # Percentile bands for fan chart: by week_of_year across all years
    percentiles = grouped['value'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack(level=1)
    percentiles.columns = ['p10', 'p25', 'p50', 'p75', 'p90']
    df = df.merge(percentiles, on='week_of_year', how='left')

    return df

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
    fig_price.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_price, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load price data (Yahoo Finance might be throttling): {e}")

st.markdown("---")

# 2. Storage
st.subheader("2. US Storage Levels (EIA Weekly)")

storage_df = get_eia_storage(EIA_API_KEY)
if storage_df is not None:
    # Compute analytics
    storage_df = compute_storage_analytics(storage_df)

    latest_storage = storage_df.iloc[-1]
    current_week = latest_storage['week_of_year']
    avg_for_week = latest_storage['level_5y_avg']
    deficit = latest_storage['value'] - avg_for_week

    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("Lower 48 Working Gas (Bcf)",
                  f"{latest_storage['value']:,}",
                  delta=f"{deficit:,.0f} vs 5yr Avg")
    s_col2.metric("Weekly Change (Bcf)",
                  f"{latest_storage['delta']:,.0f}",
                  delta=f"{latest_storage['delta_dev_vs_5y']:,.0f} vs 5yr Avg")
    s_col3.metric("Storage Level Z-Score",
                  f"{latest_storage['level_zscore']:.2f}",
                  delta="vs historical week-of-year")

    # --- 2A. Storage Level + Fan Chart ---
    fig_store = go.Figure()

    # Fan chart bands (10-90 and 25-75)
    fig_store.add_trace(go.Scatter(
        x=storage_df['period'],
        y=storage_df['p90'],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_store.add_trace(go.Scatter(
        x=storage_df['period'],
        y=storage_df['p10'],
        fill='tonexty',
        fillcolor='rgba(0, 123, 255, 0.1)',
        line=dict(width=0),
        name='10–90% band',
        hoverinfo='skip'
    ))

    fig_store.add_trace(go.Scatter(
        x=storage_df['period'],
        y=storage_df['p75'],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_store.add_trace(go.Scatter(
        x=storage_df['period'],
        y=storage_df['p25'],
        fill='tonexty',
        fillcolor='rgba(0, 123, 255, 0.2)',
        line=dict(width=0),
        name='25–75% band',
        hoverinfo='skip'
    ))

    # Median
    fig_store.add_trace(go.Scatter(
        x=storage_df['period'],
        y=storage_df['p50'],
        line=dict(color='rgba(0,0,0,0.4)', dash='dash'),
        name='Median (hist.)'
    ))

    # Actual storage
    fig_store.add_trace(go.Scatter(
        x=storage_df['period'],
        y=storage_df['value'],
        line=dict(color='blue', width=2),
        name='Actual Storage'
    ))

    fig_store.update_layout(
        title="Lower 48 Storage vs Historical Distribution",
        xaxis_title="Date",
        yaxis_title="Bcf",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_store, use_container_width=True)

    # --- 2B. Weekly Injection/Withdrawal vs 5-Year Avg ---
    st.markdown("#### Storage Analytics: Weekly Balances vs History")

    # Focus on last ~5 years for readability
    recent = storage_df.tail(52 * 5)

    fig_delta = go.Figure()

    # Actual weekly delta
    fig_delta.add_trace(go.Bar(
        x=recent['period'],
        y=recent['delta'],
        name='Actual Weekly Δ (Bcf)',
        marker_color=recent['delta'].apply(lambda x: 'red' if x < 0 else 'steelblue')
    ))

    # 5-year avg weekly delta (line)
    fig_delta.add_trace(go.Scatter(
        x=recent['period'],
        y=recent['delta_5y_avg'],
        mode='lines',
        name='5yr Avg Weekly Δ',
        line=dict(color='black', dash='dash')
    ))

    fig_delta.update_layout(
        title="Weekly Injection/Withdrawal vs 5-Year Average",
        xaxis_title="Date",
        yaxis_title="Bcf",
        height=400,
        barmode='group',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # --- 2C. Deviation & Z-Score (compact view) ---
    c1, c2 = st.columns(2)

    with c1:
        fig_dev = go.Figure()
        fig_dev.add_trace(go.Bar(
            x=recent['period'],
            y=recent['delta_dev_vs_5y'],
            name='Δ vs 5yr Avg (Bcf)',
            marker_color=recent['delta_dev_vs_5y'].apply(lambda x: 'red' if x < 0 else 'green')
        ))
        fig_dev.update_layout(
            title="Weekly Deviation vs 5-Year Avg (Bcf)",
            xaxis_title="Date",
            yaxis_title="Bcf",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_dev, use_container_width=True)

    with c2:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(
            x=recent['period'],
            y=recent['delta_zscore'],
            mode='lines+markers',
            name='Weekly Δ Z-Score'
        ))
        fig_z.add_hline(y=0, line=dict(color='black', width=1))
        fig_z.add_hline(y=1.5, line=dict(color='orange', width=1, dash='dash'))
        fig_z.add_hline(y=-1.5, line=dict(color='orange', width=1, dash='dash'))
        fig_z.update_layout(
            title="Weekly Injection/Withdrawal Z-Score (by Week-of-Year)",
            xaxis_title="Date",
            yaxis_title="Z-Score",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_z, use_container_width=True)

    # --- 2D. Cumulative Deviation vs 5-Year Avg (Gas Year) ---
    fig_cum = go.Figure()
    # Plot last few gas years
    for gy, sub in storage_df.groupby('gas_year'):
        if gy >= storage_df['gas_year'].max() - 4:  # last ~5 gas years
            fig_cum.add_trace(go.Scatter(
                x=sub['period'],
                y=sub['cum_dev_vs_5y'],
                mode='lines',
                name=f"Gas Year {gy}"
            ))

    fig_cum.add_hline(y=0, line=dict(color='black', width=1))
    fig_cum.update_layout(
        title="Cumulative Deviation vs 5-Year Avg (by Gas Year)",
        xaxis_title="Date",
        yaxis_title="Cumulative Δ vs 5yr Avg (Bcf)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_cum, use_container_width=True)

else:
    st.warning("⚠️ Could not load Lower 48 Storage Data.")

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
