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
# --- NEW GIS IMPORTS ---
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
# -----------------------

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
    "South Central Non-Salt": "NW2_EPG0_SNO_R33_BCF",
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
@st.cache_data(ttl=3600*24) # Cache for 24 hours
def get_price_data():
    """
    Fetches Henry Hub (US), and proxies for TTF (EU).
    """
    tickers = ['NG=F', 'TTF=F']
    # Use a specific start date to ensure Henry Hub has historical data for a good Z-score baseline
    start_date = (datetime.date.today() - datetime.timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, interval="1d")['Close']

    # Clean up column names
    data.rename(columns={'NG=F': 'HenryHub_USD', 'TTF=F': 'TTF_EUR'}, inplace=True)

    # Currency Conversion (Approximate EUR to USD for Spread)
    fx = yf.download("EURUSD=X", period="1d", interval="1d")['Close'].iloc[-1].item()

    # Convert TTF (usually in EUR/MWh) to USD/MMBtu
    # Conversion factor: 1 MWh = 3.412 MMBtu
    data['TTF_USD_MMBtu'] = (data['TTF_EUR'] * fx) / 3.412

    # Calculate Spread (Arb Window)
    data['Spread_TTF_HH'] = data['TTF_USD_MMBtu'] - data['HenryHub_USD']

    # Calculate Z-score for Henry Hub price (vs 10-year history)
    data['HenryHub_Z'] = (data['HenryHub_USD'] - data['HenryHub_USD'].mean()) / data['HenryHub_USD'].std(ddof=0)

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
            # st.error(f"EIA Structure Error: API returned empty data for series {series_id}.")
            return None

    except Exception as e:
        st.error(f"EIA Fetch Error for {series_id}: {e}")
        return None

# --- 3. DATA SOURCE: WEATHER (HDD Forecast) ---
@st.cache_data(ttl=3600*12) # Update weather every 12 hours
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

# --- HELPER: GET SCT Z-SCORE FOR MAP ---
def get_sct_zscore_for_map(api_key):
    """Fetches and computes the latest storage level Z-score for South Central Total."""
    sct_series = EIA_SERIES["South Central Total"]
    sct_df = get_eia_series(api_key, sct_series)
    if sct_df is not None and not sct_df.empty:
        sct_df_analyzed = compute_storage_analytics(sct_df)
        return sct_df_analyzed.iloc[-1]['level_zscore']
    return None

# --- GEOSPATIAL POC FUNCTION ---
def plot_gis_poc(henry_hub_z: float, sct_z: float):
    """
    Generates the proof-of-concept map using the uploaded shapefile
    and calculated anomalies.
    """
    st.markdown("---")
    st.subheader("4. Geospatial Trading View (PoC)")
    st.write("Visualizing key market anomalies on the physical infrastructure map (Gulf Coast focus).")
    
    try:
        # Load User's Pipeline Shapefile (requires all components: shp, shx, dbf, prj)
        pipeline_gdf = gpd.read_file("Natural_Gas_Interstate_and_Intrastate_Pipelines.shp")
        # Assuming the PRJ file handles the CRS correctly, converting to WGS84 for safety
        pipeline_gdf = pipeline_gdf.to_crs(epsg=4326) 
    
        # Bounding box for Gulf Coast (approx. -100 to -80 longitude, 25 to 35 latitude)
        bbox = [-100, 25, -80, 35]
        pipeline_gdf_filtered = pipeline_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    
        # --- Define Key Trading Points & Anomalies ---
    
        # A. Henry Hub (H_HUB) - Price Benchmark
        henry_hub_data = {
            'name': ['Henry Hub'],
            'longitude': [-92.0000],
            'latitude': [30.0716],
        }
        henry_hub_gdf = gpd.GeoDataFrame(
            henry_hub_data,
            geometry=gpd.points_from_xy(henry_hub_data['longitude'], henry_hub_data['latitude']),
            crs="EPSG:4326"
        )
        # Determine Henry Hub marker color based on Z-score
        # Green = High Price (Bullish), Red = Low Price (Bearish)
        hh_color = 'green' if henry_hub_z > 0.5 else ('red' if henry_hub_z < -0.5 else 'yellow')
        hh_size = 1000 + abs(henry_hub_z) * 500
        hh_label = f"HH Price Z-Score: {henry_hub_z:.2f}"
    
        # B. Key LNG Terminals (Simulated Demand Shock Points)
        # This remains simulated as we don't have LNG flow data
        lng_data = {
            'name': ['Sabine Pass LNG', 'Corpus Christi LNG'],
            'longitude': [-93.8841, -97.3941],
            'latitude': [29.7431, 27.8732],
            'simulated_flow_shock': [0.10, 0.40] # Simulated: Sabine 10% outage, Corpus 40% outage
        }
        lng_gdf = gpd.GeoDataFrame(
            lng_data,
            geometry=gpd.points_from_xy(lng_data['longitude'], lng_data['latitude']),
            crs="EPSG:4326"
        )
    
        # C. South Central (SCT) Storage Region (Conceptual Polygon for visual focus)
        # We use a conceptual polygon for visualization, as the actual complex shapefile is not available
        sct_polygon_coords = [
            (-100, 25), (-100, 35), (-80, 35), (-80, 25), (-100, 25)
        ]
        sct_region_poly = Polygon(sct_polygon_coords)
        sct_region_gdf = gpd.GeoDataFrame(
            {'name': ['South Central (SCT)'], 'storage_zscore': [sct_z]},
            geometry=[sct_region_poly],
            crs="EPSG:4326"
        )
        # Determine SCT region color/opacity based on Z-score
        # Red = Low Storage (Bullish), Green = High Storage (Bearish)
        sct_fill_color = 'red' if sct_z < -0.5 else ('green' if sct_z > 0.5 else 'yellow')
        sct_alpha = min(0.6, 0.2 + abs(sct_z) * 0.4)
        sct_label = f"SCT Level Z-Score: {sct_z:.2f}"
    
        # --- Create the Visualization ---
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
        # Plot 1: SCT Storage Region (Dynamic Color/Opacity)
        sct_region_gdf.plot(
            ax=ax, 
            color=sct_fill_color, 
            alpha=sct_alpha, 
            edgecolor='black', 
            linewidth=1, 
            label='SCT Storage Region'
        )
    
        # Plot 2: Pipelines (Gray)
        pipeline_gdf_filtered.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
    
        # Plot 3: LNG Terminals (Blue Circles, scaled by simulated shock)
        lng_gdf.plot(
            ax=ax,
            marker='o',
            color='blue',
            markersize=lng_gdf['simulated_flow_shock'] * 4000,
            alpha=0.6,
            edgecolor='black',
            label='LNG Terminal Demand Shock'
        )
        # Add text labels for LNG
        for x, y, label, shock in zip(lng_gdf.geometry.x, lng_gdf.geometry.y, lng_gdf['name'], lng_gdf['simulated_flow_shock']):
            if shock > 0.0:
                ax.annotate(f"{label} ({shock*100:.0f}% shock)", xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, color='darkblue')
    
        # Plot 4: Henry Hub (Dynamic Star)
        henry_hub_gdf.plot(
            ax=ax,
            marker='*',
            color=hh_color,
            markersize=hh_size,
            edgecolor='black',
            label='Henry Hub Price Benchmark'
        )
        # Add text label for Henry Hub
        x, y, label = henry_hub_gdf.geometry.x[0], henry_hub_gdf.geometry.y[0], henry_hub_gdf['name'][0]
        ax.annotate(label, xy=(x, y), xytext=(5, -15), textcoords="offset points", fontsize=10, color=hh_color, fontweight='bold')
    
        # Add legend/title
        ax.set_title("PoC: Natural Gas Trading Infrastructure & Anomaly Map (Gulf Coast Focus)", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1)
        
        # Display the anomaly data on the map
        ax.text(bbox[0]+1, bbox[3]-1.5, 
                f"Market Anomaly Summary:\n\n"
                f"SCT Storage Status: {sct_label}\n"
                f"Henry Hub Price Status: {hh_label}",
                fontsize=10, 
                color='black', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"⚠️ Could not generate the GIS map. Ensure all shapefile components are uploaded correctly: {e}")
        st.info("The map requires: Natural_Gas_Interstate_and_Intrastate_Pipelines.shp, .shx, .dbf, .prj.")


# --- MAIN DASHBOARD LOGIC ---

# 1. Prices & Spreads
st.subheader("1. International Future Spreads (Arbitrage Window)")
try:
    price_df = get_price_data()
    latest = price_df.iloc[0]

    # Get the Henry Hub Z-score for use in the map
    henry_hub_z_score = latest['HenryHub_Z'] if pd.notna(latest['HenryHub_Z']) else 0.0

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
    henry_hub_z_score = 0.0 # Default to 0 if data fails

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
    recent = display_df # for deltas / z-scores

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
        if gy >= storage_df['gas_year'].max() - 4: # last ~5 gas years
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
    # Default Z-score if data fails
    sct_storage_z_score = 0.0

st.markdown("---")

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
    
# --- 4. GIS MAP POC CALL ---

# Get the South Central Z-score to drive the map's visual anomaly
sct_storage_z_score = get_sct_zscore_for_map(EIA_API_KEY) or 0.0

if 'henry_hub_z_score' in locals():
    plot_gis_poc(henry_hub_z_score, sct_storage_z_score)
else:
    st.warning("Cannot generate GIS map. Price data (Henry Hub Z-Score) is not available.")
