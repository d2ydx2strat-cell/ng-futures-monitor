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
st.set_page_config(page_title="NG Trading Monitor", layout="wide")
st.title("🔥 Global NG Spreads, Storage & Weather Monitor")

EIA_API_KEY = "KzzwPVmMSTVCI3pQbpL9calvF4CqGgEbwWy0qqXV"

# EIA weekly working gas series IDs
# These are standard Lower 48 + 5 regions + South Central salt / non-salt
EIA_SERIES = {
    "Lower 48 Total": "NW2_EPG0_SWO_R48_BCF",
    "East": "NW2_EPG0_SWO_R31_BCF",
    "Midwest": "NW2_EPG0_SWO_R32_BCF",
    "Mountain": "NW2_EPG0_SWO_R33_BCF",
    "Pacific": "NW2_EPG0_SWO_R34_BCF",
    "South Central Total": "NW2_EPG0_SWO_R35_BCF",
    "South Central Salt": "NW2_EPG0_SSO_R33_BCF",
    "South Central Non-Salt": "NW2_EPG0_SWO_R35N_BCF",
}

# If you know working gas capacity by region, you can hardcode here (Bcf).
# For now, leave as None; logic will handle missing capacity.
REGION_CAPACITY_BCF = {
    "Lower 48 Total": None,
    "East": None,
    "Midwest": None,
    "Mountain": None,
    "Pacific": None,
    "South Central Total": None,
    "South Central Salt": None,
    "South Central Non-Salt": None,
}

# Sidebar
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
def get_eia_series(api_key: str, series_id: str, length_weeks: int = 52 * 15) -> pd.DataFrame | None:
    """
    Generic fetcher for a single EIA weekly storage series.
    Returns DataFrame with ['period', 'value'] sorted ascending by period.
    """
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
        r = requests.get(url, params=params)
        data = r.json()

        if 'error' in data:
            st.error(f"EIA API Error for {series_id}: {data['error']}")
            return None

        if 'response' in data and 'data' in data['response'] and data['response']['data']:
            df = pd.DataFrame(data['response']['data'])
            df['period'] = pd.to_datetime(df['period'])
            df['value'] = pd.to_numeric(df['value'])
            df = df.sort_values('period').reset_index(drop=True)
            return df
        else:
            st.error(f"EIA Structure Error: API returned empty data for series {series_id}.")
            return None

    except Exception as e:
        st.error(f"EIA Fetch Error for {series_id}: {e}")
        return None

# --- 3. DATA SOURCE: WEATHER (HDD Forecast) ---
@st.cache_data(ttl=3600*12)  # Update weather every 12 hours
def get_weather_demand():
    """
    Uses Open-Meteo to fetch 10-day forecast for key gas-consuming hubs:
    Chicago (Midwest), New York (East), Houston (South - CDD).
    Calculates proxy HDD (Heating Degree Days).
    """
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

    df['week_of_year'] = df['period'].dt.isocalendar().week.astype(int)
    df['year'] = df['period'].dt.year

    # Weekly injection/withdrawal (delta)
    df['delta'] = df['value'].diff()

    grouped = df.groupby('week_of_year')

    # Level stats by week_of_year
    df['level_5y_avg'] = df['week_of_year'].map(grouped['value'].mean())

    # Weekly delta stats by week_of_year
    delta_mean = grouped['delta'].mean()
    delta_std = grouped['delta'].std(ddof=0)

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

    # Level z-score
    level_mean = grouped['value'].mean()
    level_std = grouped['value'].std(ddof=0)
    df['level_zscore'] = df.apply(
        lambda r: _safe_z(r, level_mean, level_std, 'value'),
        axis=1
    )

    # Cumulative deviation vs 5y avg delta, by gas year (start April)
    df['gas_year'] = np.where(df['period'].dt.month >= 4, df['period'].dt.year, df['period'].dt.year - 1)
    df['cum_dev_vs_5y'] = df.groupby('gas_year')['delta_dev_vs_5y'].cumsum()

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

# Region selector
region_names = list(EIA_SERIES.keys())
default_region = "Lower 48 Total"
selected_region = st.selectbox("Select Region / South Central Detail", region_names, index=region_names.index(default_region))

series_id = EIA_SERIES[selected_region]
capacity_bcf = REGION_CAPACITY_BCF.get(selected_region)

storage_df = get_eia_series(EIA_API_KEY, series_id)

if storage_df is not None and not storage_df.empty:
    # Compute analytics on FULL history
    storage_df = compute_storage_analytics(storage_df)
    latest_storage = storage_df.iloc[-1]

    # Metrics (still based on full-history stats)
    current_level = latest_storage['value']
    current_delta = latest_storage['delta']
    level_5y_avg = latest_storage['level_5y_avg']
    delta_5y_avg = latest_storage['delta_5y_avg']
    level_deficit = current_level - level_5y_avg
    delta_deficit = current_delta - delta_5y_avg
    level_z = latest_storage['level_zscore']

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric(f"{selected_region} Working Gas (Bcf)",
                  f"{current_level:,.0f}",
                  delta=f"{level_deficit:,.0f} vs 5yr Avg")

    s_col2.metric("Weekly Change (Bcf)",
                  f"{current_delta:,.0f}",
                  delta=f"{delta_deficit:,.0f} vs 5yr Avg")

    s_col3.metric("Storage Level Z-Score",
                  f"{level_z:.2f}" if pd.notna(level_z) else "N/A",
                  delta="vs hist. week-of-year")

    if capacity_bcf is not None:
        pct_full = current_level / capacity_bcf * 100
        s_col4.metric("Utilization (% of Capacity)",
                      f"{pct_full:.1f}%",
                      delta=None)
    else:
        s_col4.metric("Utilization (% of Capacity)", "N/A", delta=None)

    # ---- LIMIT DISPLAY TO LAST 2 YEARS (104 WEEKS) ----
    display_window_weeks = 52 * 2
    display_df = storage_df.tail(display_window_weeks)
    recent = display_df  # for deltas / z-scores

    # --- 2A. Storage Level + Fan Chart (last 2 years only) ---
    fig_store = go.Figure()

    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p90'],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p10'],
        fill='tonexty',
        fillcolor='rgba(0, 123, 255, 0.1)',
        line=dict(width=0),
        name='10–90% band',
        hoverinfo='skip'
    ))

    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p75'],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p25'],
        fill='tonexty',
        fillcolor='rgba(0, 123, 255, 0.2)',
        line=dict(width=0),
        name='25–75% band',
        hoverinfo='skip'
    ))

    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p50'],
        line=dict(color='rgba(0,0,0,0.4)', dash='dash'),
        name='Median (hist.)'
    ))

    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['value'],
        line=dict(color='blue', width=2),
        name='Actual Storage'
    ))

    fig_store.update_layout(
        title=f"{selected_region} Storage vs Historical Distribution (Last 2 Years)",
        xaxis_title="Date",
        yaxis_title="Bcf",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_store, use_container_width=True)

    # --- 2B. Weekly Injection/Withdrawal vs 5-Year Avg (last 2 years) ---
    st.markdown("#### Storage Analytics: Weekly Balances vs History (Last 2 Years)")

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Bar(
        x=recent['period'],
        y=recent['delta'],
        name='Actual Weekly Δ (Bcf)',
        marker_color=recent['delta'].apply(lambda x: 'red' if x < 0 else 'steelblue')
    ))
    fig_delta.add_trace(go.Scatter(
        x=recent['period'],
        y=recent['delta_5y_avg'],
        mode='lines',
        name='5yr Avg Weekly Δ',
        line=dict(color='black', dash='dash')
    ))
    fig_delta.update_layout(
        title=f"{selected_region}: Weekly Injection/Withdrawal vs 5-Year Average",
        xaxis_title="Date",
        yaxis_title="Bcf",
        height=400,
        barmode='group',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # --- 2C. Deviation & Z-Score (last 2 years) ---
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
            title=f"{selected_region}: Weekly Deviation vs 5-Year Avg (Bcf)",
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
            title=f"{selected_region}: Weekly Injection/Withdrawal Z-Score",
            xaxis_title="Date",
            yaxis_title="Z-Score",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_z, use_container_width=True)

    # --- 2D. Cumulative Deviation vs 5-Year Avg (Gas Year) ---
    # This is already limited to last ~5 gas years; keep as-is
    fig_cum = go.Figure()
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
        title=f"{selected_region}: Cumulative Deviation vs 5-Year Avg (by Gas Year)",
        xaxis_title="Date",
        yaxis_title="Cumulative Δ vs 5yr Avg (Bcf)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_cum, use_container_width=True)

else:
    st.warning(f"⚠️ Could not load storage data for {selected_region}.")


# 3. Weather
st.subheader("3. 10-Day HDD Forecast (Gas Demand Proxy)")
st.write("Projected Heating Degree Days (HDD) for key consumption hubs.")
try:
    weather_df = get_weather_demand()
    chart_data = weather_df.pivot(index='date', columns='City', values='HDD')
    st.line_chart(chart_data)

    total_hdd = chart_data.sum(axis=1)
    st.metric("Total System Forecast HDD (Next 10 Days)", f"{total_hdd.sum():.0f}")

except Exception as e:
    st.error(f"Weather data error: {e}")


