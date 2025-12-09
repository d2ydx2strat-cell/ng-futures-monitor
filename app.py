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
import statsmodels.api as sm
from typing import Dict, Optional

# Geo imports for Section 5
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

EIA_SERIES: Dict[str, str] = {
    "Lower 48 Total": "NW2_EPG0_SWO_R48_BCF",
    "East": "NW2_EPG0_SWO_R31_BCF",
    "Midwest": "NW2_EPG0_SWO_R32_BCF",
    "Mountain": "NW2_EPG0_SWO_R33_BCF",
    "Pacific": "NW2_EPG0_SWO_R34_BCF",
    "South Central Total": "NW2_EPG0_SWO_R35_BCF",
    "South Central Salt": "NW2_EPG0_SSO_R33_BCF",
    "South Central Non-Salt": "NW2_EPG0_SNO_R33_BCF",
}

# NEW: EIA DAILY SERIES IDs
EIA_DAILY_SERIES: Dict[str, str] = {
    # Natural Gas Liquefied (Feed Gas to Export) - Lower 48 States (MMCFD = Million Cubic Feet per Day)
    "LNG_Feed_Gas_MMCFD": "NG.NOA_C_SUM_SL_A_EPM0_VGM_MMCFD",
    # Dry Gas Production - Lower 48 States (MMCFD)
    "Dry_Gas_Production_MMCFD": "NG.RNGR48.D",
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

# --- 4. DATA SOURCE: DAILY FLOWS (EIA) ---
@st.cache_data(ttl=3600 * 3)
def get_eia_daily_flows(api_key: str, series_ids: Dict[str, str], length_days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetches daily EIA Natural Gas flow data (LNG Feed Gas and Production).
    """
    base_url = "https://api.eia.gov/v2/natural-gas/nggas/d/data/"
    all_data = []

    for name, series_id in series_ids.items():
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[seriesId][]": series_id,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": 0,
            "length": length_days,
        }

        try:
            r = requests.get(base_url, params=params, timeout=30)
            data = r.json()

            if 'response' in data and 'data' in data['response'] and data['response']['data']:
                df = pd.DataFrame(data['response']['data'])
                df['period'] = pd.to_datetime(df['period'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df.rename(columns={'value': name, 'period': 'date'}, inplace=True)
                df = df[['date', name]].sort_values('date').set_index('date')
                all_data.append(df)
            else:
                pass 
                
        except Exception:
            continue

    if not all_data:
        return None
        
    final_df = all_data[0].copy()
    for df in all_data[1:]:
        final_df = final_df.join(df, how='outer')
        
    return final_df.reset_index()


# --- 5. NOAA 7‑DAY TEMP ANOMALY BY REGION (api.weather.gov) ---
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
    df = df.merge(percentiles, on='week_of_year', how='left')

    return df

# --- MAP HELPERS (Pipeline Data) ---
def gdf_to_plotly_lines(gdf: gpd.GeoDataFrame):
    if gdf is None:
        return [], []
    if gdf.crs != "EPSG:4326":
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass

    lons = []
    lats = []

    for geom in gdf.geometry:
        if geom is None or getattr(geom, 'is_empty', False):
            continue
        if geom.geom_type in ["LineString", "MultiLineString"]:
            lines = [geom] if geom.geom_type == "LineString" else geom.geoms
            for line in lines:
                try:
                    x, y = line.xy
                except Exception:
                    continue
                lons.extend(list(x))
                lats.extend(list(y))
                lons.append(None)
                lats.append(None)

    return lons, lats

@st.cache_data
def load_pipeline_data():
    try:
        pipelines_gdf = gpd.read_file(SHAPEFILE_PATH)
        minx, miny, maxx, maxy = pipelines_gdf.total_bounds
        bbox_polygon = box(minx, miny, maxx, maxy)

        boundary_gdf = gpd.GeoDataFrame(
            {"name": ["Pipeline Extent"]},
            geometry=[bbox_polygon],
            crs=pipelines_gdf.crs,
        )

        return pipelines_gdf, boundary_gdf

    except Exception as e:
        st.error(f"Error loading pipeline shapefile: {e}")
        st.warning("Ensure .shp, .shx, .dbf, .prj (and .cpg) are present in the app directory.")
        return None, None

# --- WEEKLY MERGED DATA FOR FAIR VALUE MODEL ---

@st.cache_data(ttl=3600*24)
def build_weekly_merged_dataset():
    """
    Build a weekly DataFrame with:
      - NG1 weekly close
      - Lower 48 storage level, z-score, cum deviation
      - TTF-HH spread (weekly avg)
      - HDD (weekly sum) and HDD deviation vs 5y avg (PLACEHOLDER)
    """
    # 1) Storage (Lower 48)
    stor_raw = get_eia_series(EIA_API_KEY, EIA_SERIES["Lower 48 Total"])
    if stor_raw is None or stor_raw.empty:
        return None
    stor = compute_storage_analytics(stor_raw)
    stor = stor[['period', 'value', 'level_zscore', 'cum_dev_vs_5y', 'delta', 'delta_5y_avg']]
    stor.rename(columns={
        'period': 'week_date',
        'value': 'Storage_Bcf',
        'level_zscore': 'Storage_Z',
        'cum_dev_vs_5y': 'CumDev_Bcf',
        'delta': 'Net_Withdrawal',
        'delta_5y_avg': 'Net_Withdrawal_5y'
    }, inplace=True)
    stor['Net_Withdrawal_Dev'] = stor['Net_Withdrawal'] - stor['Net_Withdrawal_5y']

    # 2) Daily prices & spreads
    price_df = get_price_data()
    # NG1 weekly close (Friday or last trading day)
    ng_weekly = price_df['HenryHub_USD'].resample('W-FRI').last().to_frame('NG1')
    spread_weekly = price_df['Spread_TTF_HH'].resample('W-FRI').mean().to_frame('TTF_HH_Spread')

    price_weekly = ng_weekly.join(spread_weekly, how='inner')
    price_weekly.reset_index(inplace=True)
    price_weekly.rename(columns={'Date': 'week_date'}, inplace=True)

    # 3) HDD historical (placeholder - needs full integration)
    price_weekly['HDD'] = np.nan  # placeholder
    price_weekly['HDD_Dev'] = np.nan

    # 4) Merge storage with price/spreads
    weekly = pd.merge_asof(
        stor.sort_values('week_date'),
        price_weekly.sort_values('week_date'),
        on='week_date',
        direction='backward'
    )

    weekly.dropna(subset=['NG1', 'Storage_Z', 'CumDev_Bcf', 'TTF_HH_Spread'], inplace=True)
    weekly.reset_index(drop=True, inplace=True)

    return weekly

# --- FAIR VALUE MODEL (STORAGE-BASED) ---

def fit_fair_value_model(weekly_df: pd.DataFrame):
    """
    Fit OLS: NG1 = α + β1*Storage_Z + β2*CumDev_Bcf + β3*TTF_HH_Spread
    """
    df = weekly_df.dropna(subset=['NG1', 'Storage_Z', 'CumDev_Bcf', 'TTF_HH_Spread']).copy()
    X = df[['Storage_Z', 'CumDev_Bcf', 'TTF_HH_Spread']]
    X = sm.add_constant(X)
    y = df['NG1']
    model = sm.OLS(y, X).fit()
    df['NG1_FV'] = model.predict(X)
    df['Mispricing'] = df['NG1'] - df['NG1_FV']
    df['Mispricing_ZScore'] = (df['Mispricing'] - df['Mispricing'].mean()) / df['Mispricing'].std()
    return model, df

# --- MAIN DASHBOARD LOGIC ---

# NEW: Fetch Daily Flow Data
daily_flows_df = get_eia_daily_flows(EIA_API_KEY, EIA_DAILY_SERIES)

# 1. Prices & Spreads
st.subheader("1. International Future Spreads (Arbitrage Window)")
try:
    price_df = get_price_data()
    latest = price_df.iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Henry Hub (US)", f"${latest['HenryHub_USD']:.2f}", f"{price_df['HenryHub_USD'].diff().iloc[-1]:.2f}")
    col2.metric("TTF Proxy (EU)", f"${latest['TTF_USD_MMBtu']:.2f}", f"{price_df['TTF_USD_MMBtu'].diff().iloc[-1]:.2f}")
    col3.metric("Spread (Export Arb)", f"${latest['Spread_TTF_HH']:.2f}", "High spread = Bullish US LNG")

    fig_price = make_subplots(specs=[[{"secondary_y": True}]])
    fig_price.add_trace(go.Scatter(x=price_df.index, y=price_df['HenryHub_USD'], name="Henry Hub ($)"), secondary_y=False)
    fig_price.add_trace(go.Scatter(x=price_df.index, y=price_df['TTF_USD_MMBtu'], name="TTF EU ($/MMBtu)"), secondary_y=True)
    fig_price.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_price, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load price data (Yahoo Finance might be throttling): {e}")

st.markdown("---")

# 2. Storage (existing detailed section kept as-is, using selected region)
st.subheader("2. US Storage Levels (EIA Weekly)")

region_names = list(EIA_SERIES.keys())
default_region = "Lower 48 Total"
selected_region = st.selectbox("Select Region / South Central Detail", region_names, index=region_names.index(default_region))

series_id = EIA_SERIES[selected_region]
capacity_bcf = REGION_CAPACITY_BCF.get(selected_region)

storage_df = get_eia_series(EIA_API_KEY, series_id)

if storage_df is not None and not storage_df.empty:
    storage_df = compute_storage_analytics(storage_df)
    latest_storage = storage_df.iloc[-1]

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

    display_window_weeks = 52 * 2
    display_df = storage_df.tail(display_window_weeks)
    recent = display_df

    fig_store = go.Figure()
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p90'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p10'], fill='tonexty', fillcolor='rgba(0, 123, 255, 0.1)', line=dict(width=0), name='10–90% band', hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p75'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p25'], fill='tonexty', fillcolor='rgba(0, 123, 255, 0.2)', line=dict(width=0), name='25–75% band', hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p50'], line=dict(color='rgba(0,0,0,0.4)', dash='dash'), name='Median (hist.)'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['value'], line=dict(color='blue', width=2), name='Actual Storage'))
    fig_store.update_layout(title=f"{selected_region} Storage vs Historical Distribution (Last 2 Years)", xaxis_title="Date", yaxis_title="Bcf", height=450, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_store, use_container_width=True)

    st.markdown("#### Storage Analytics: Weekly Balances vs History (Last 2 Years)")

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Bar(x=recent['period'], y=recent['delta'], name='Actual Weekly Δ (Bcf)', marker_color=recent['delta'].apply(lambda x: 'red' if x < 0 else 'steelblue')))
    fig_delta.add_trace(go.Scatter(x=recent['period'], y=recent['delta_5y_avg'], mode='lines', name='5yr Avg Weekly Δ', line=dict(color='black', dash='dash')))
    fig_delta.update_layout(title=f"{selected_region}: Weekly Injection/Withdrawal vs 5-Year Average", xaxis_title="Date", yaxis_title="Bcf", height=400, barmode='group', margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_delta, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_dev = go.Figure()
        fig_dev.add_trace(go.Bar(x=recent['period'], y=recent['delta_dev_vs_5y'], name='Δ vs 5yr Avg (Bcf)', marker_color=recent['delta_dev_vs_5y'].apply(lambda x: 'red' if x < 0 else 'green')))
        fig_dev.update_layout(title=f"{selected_region}: Weekly Deviation vs 5-Year Avg (Bcf)", xaxis_title="Date", yaxis_title="Bcf", height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_dev, use_container_width=True)

    with c2:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=recent['period'], y=recent['delta_zscore'], mode='lines+markers', name='Weekly Δ Z-Score'))
        fig_z.add_hline(y=0, line=dict(color='black', width=1))
        fig_z.add_hline(y=1.5, line=dict(color='orange', width=1, dash='dash'))
        fig_z.add_hline(y=-1.5, line=dict(color='orange', width=1, dash='dash'))
        fig_z.update_layout(title=f"{selected_region}: Weekly Injection/Withdrawal Z-Score", xaxis_title="Date", yaxis_title="Z-Score", height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_z, use_container_width=True)

    fig_cum = go.Figure()
    for gy, sub in storage_df.groupby('gas_year'):
        if gy >= storage_df['gas_year'].max() - 4:
            fig_cum.add_trace(go.Scatter(x=sub['period'], y=sub['cum_dev_vs_5y'], mode='lines', name=f"Gas Year {gy}"))
    fig_cum.add_hline(y=0, line=dict(color='black', width=1))
    fig_cum.update_layout(title=f"{selected_region}: Cumulative Deviation vs 5-Year Avg (by Gas Year)", xaxis_title="Date", yaxis_title="Cumulative Δ vs 5yr Avg (Bcf)", height=400, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_cum, use_container_width=True)

else:
    st.warning(f"⚠️ Could not load storage data for {selected_region}.")

st.markdown("---")

# 3. Weather (10-day HDD)
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

st.markdown("---")

# 4. FAIR VALUE MODEL SECTION
st.subheader("4. Storage-Based Fair Value Model for NG1")

weekly_df = build_weekly_merged_dataset()
if weekly_df is None or weekly_df.empty:
    st.info("Insufficient data to build fair value model.")
else:
    model, fv_df = fit_fair_value_model(weekly_df)

    latest_row = fv_df.iloc[-1]
    mispricing = latest_row['Mispricing']
    mispricing_z = latest_row['Mispricing_ZScore']
    mispricing_pctile = (fv_df['Mispricing'] <= mispricing).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current NG1", f"${latest_row['NG1']:.2f}")
    c2.metric("Model Fair Value", f"${latest_row['NG1_FV']:.2f}")
    c3.metric("Mispricing (Actual - FV)", f"{mispricing:+.2f}", f"Z-Score: {mispricing_z:+.2f}")
    c4.metric("Mispricing Percentile", f"{mispricing_pctile:.0f}th", "vs last 5–10 years")

    fig_fv = go.Figure()
    fig_fv.add_trace(go.Scatter(x=fv_df['week_date'], y=fv_df['NG1'], name="NG1 Actual", line=dict(color='blue')))
    fig_fv.add_trace(go.Scatter(x=fv_df['week_date'], y=fv_df['NG1_FV'], name="Model Fair Value", line=dict(color='orange')))
    fig_fv.update_layout(title="NG1 vs Storage-Based Fair Value", xaxis_title="Week", yaxis_title="$/MMBtu", height=450, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_fv, use_container_width=True)

    fig_mis = go.Figure()
    fig_mis.add_trace(go.Bar(x=fv_df['week_date'], y=fv_df['Mispricing'], name="Mispricing (NG1 - FV)", marker_color=fv_df['Mispricing'].apply(lambda x: 'red' if x < 0 else 'green')))
    fig_mis.add_hline(y=0, line=dict(color='black', width=1))
    fig_mis.update_layout(title="NG1 Mispricing vs Fair Value", xaxis_title="Week", yaxis_title="$/MMBtu", height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_mis, use_container_width=True)

st.markdown("---")

# 5. U.S. Infrastructure Map (Pipelines, LNG, Storage, NOAA)
st.subheader("5. U.S. Infrastructure Map (Pipelines, LNG, Storage, NOAA Outlook)")

REGION_CAPACITY_BCF = {
    "East": 950,
    "Midwest": 1100,
    "Mountain": 270,
    "Pacific": 420,
    "South Central Salt": 490,
    "South Central Non-Salt": 980,
    "South Central Total": 1470  
}

lng_df = get_lng_terminals()

storage_map_data = []
regions_to_map = ["East", "Midwest", "Mountain", "Pacific", "South Central Salt", "South Central Non-Salt"]
centroids = get_storage_centroids()

noaa_regional_df = get_noaa_temp_anomaly_by_region(centroids)

with st.spinner("Loading infrastructure layers..."):
    for reg in regions_to_map:
        if reg in EIA_SERIES:
            sid = EIA_SERIES[reg]
            df_reg = get_eia_series(EIA_API_KEY, sid, length_weeks=5)
            
            if df_reg is not None and not df_reg.empty:
                latest_val = df_reg.iloc[-1]['value']
                
                cap = REGION_CAPACITY_BCF.get(reg)
                pct_str = "N/A"
                if cap:
                    pct = (latest_val / cap) * 100
                    pct_str = f"{pct:.0f}%"

                coords = centroids.get(reg)
                if coords:
                    storage_map_data.append({
                        "region": reg,
                        "value": latest_val,
                        "pct_full": pct_str,
                        "lat": coords["Lat"],
                        "lon": coords["Lon"]
                    })

storage_points_df = pd.DataFrame(storage_map_data)

if not noaa_regional_df.empty and not storage_points_df.empty:
    storage_points_df = storage_points_df.merge(
        noaa_regional_df[["region", "temp_bias", "forecast_mean_temp", "normal_temp_est"]],
        on="region",
        how="left",
    )
else:
    storage_points_df["temp_bias"] = np.nan
    storage_points_df["forecast_mean_temp"] = np.nan
    storage_points_df["normal_temp_est"] = np.nan

pipelines_gdf, boundary_gdf = load_pipeline_data()

def create_satellite_map_v2(gdf_pipelines, gdf_boundary, lng_df, storage_points_df, show_noaa=True):
    import os
    mapbox_token = None
    if "MAPBOX_TOKEN" in st.secrets:
        mapbox_token = st.secrets["MAPBOX_TOKEN"]
    if not mapbox_token:
        mapbox_token = os.getenv("MAPBOX_TOKEN")

    use_satellite = False
    map_style = "carto-darkmatter"
    if mapbox_token and mapbox_token.startswith("pk."):
        use_satellite = True
        map_style = "satellite-streets"

    pipeline_lons, pipeline_lats = gdf_to_plotly_lines(gdf_pipelines)
    
    try:
        gdf_boundary_4326 = gdf_boundary.to_crs(epsg=4326)
        center_point = gdf_boundary_4326.geometry.unary_union.centroid
        center_lat, center_lon = center_point.y, center_point.x
    except Exception:
        center_lat, center_lon = 39.8, -98.6

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=pipeline_lons,
        lat=pipeline_lats,
        name="Pipelines",
        line=dict(width=1, color="rgba(255, 50, 50, 0.6)"),
        hoverinfo="none"
    ))

    if not lng_df.empty:
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=lng_df['Lon'],
            lat=lng_df['Lat'],
            name="LNG Terminals",
            text=lng_df['Name'] + "<br>Cap: " + lng_df['Capacity_Bcfd'].astype(str) + " Bcfd",
            marker=dict(
                size=12,
                color='#39FF14',
                opacity=1.0,
            ),
            hoverinfo='text'
        ))

    if storage_points_df is not None and not storage_points_df.empty:
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=storage_points_df['lon'],
            lat=storage_points_df['lat'],
            name="Regional Storage",
            text=storage_points_df['pct_full'],
            textposition="middle center",
            textfont=dict(size=11, color="white", weight="bold"),
            hovertext=storage_points_df['region'] + 
                      "<br>Vol: " + storage_points_df['value'].astype(int).astype(str) + " Bcf" +
                      "<br>Full: " + storage_points_df['pct_full'],
            marker=dict(
                size=storage_points_df['value'] / 12,  
                sizemin=15,
                color='#003366',
                opacity=0.8,
            ),
            hoverinfo='text'
        ))

    if show_noaa and storage_points_df is not None and not storage_points_df.empty and "temp_bias" in storage_points_df.columns:
        df_noaa = storage_points_df.dropna(subset=["temp_bias"]).copy()
        if not df_noaa.empty:
            df_noaa["temp_bias_clipped"] = df_noaa["temp_bias"].clip(-20, 20)

            fig.add_trace(
                go.Scattermapbox(
                    mode="markers",
                    lon=df_noaa["lon"],
                    lat=df_noaa["lat"],
                    name="NOAA 7d Temp Anomaly",
                    hovertext=(
                        df_noaa["region"]
                        + "<br>Forecast mean: " + df_noaa["forecast_mean_temp"].round(1).astype(str) + "°F"
                        + "<br>'Normal' est: " + df_noaa["normal_temp_est"].round(1).astype(str) + "°F"
                        + "<br>Bias: " + df_noaa["temp_bias"].round(1).astype(str) + "°F"
                    ),
                    marker=dict(
                        size=22,
                        color=df_noaa["temp_bias_clipped"],
                        colorscale=[
                            [0.0, "rgb(0, 70, 200)"],
                            [0.5, "rgb(255, 255, 255)"],
                            [1.0, "rgb(200, 0, 0)"],
                        ],
                        cmin=-20,
                        cmax=20,
                        opacity=0.7,
                    ),
                    hoverinfo="text",
                )
            )

    layout_args = dict(
        title="US Natural Gas Infrastructure, Storage & NOAA 7‑Day Outlook",
        mapbox=dict(
            style=map_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=3.5,
        ),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=750,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white"))
    )
    
    if use_satellite:
        layout_args["mapbox_accesstoken"] = mapbox_token

    fig.update_layout(**layout_args)
    return fig

if pipelines_gdf is not None:
    fig_map = create_satellite_map_v2(pipelines_gdf, boundary_gdf, lng_df, storage_points_df, show_noaa=show_noaa)
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Pipeline map components missing.")

st.markdown("---")

# --- 6. REAL-TIME SUPPLY & EXPORT FLOWS ---
st.subheader("6. Real-Time Supply & LNG Export Flows (Short-Term Balance)")

if daily_flows_df is not None and not daily_flows_df.empty:
    # Use the last 90 days for flow visualization and calculation
    flows = daily_flows_df.set_index('date').tail(90).dropna(how='all')

    if 'LNG_Feed_Gas_MMCFD' in flows.columns and 'Dry_Gas_Production_MMCFD' in flows.columns:
        
        # --- LNG Flow Analysis ---
        flows['LNG_MA30'] = flows['LNG_Feed_Gas_MMCFD'].rolling(window=30).mean()
        latest_lng = flows['LNG_Feed_Gas_MMCFD'].iloc[-1]
        latest_ma = flows['LNG_MA30'].iloc[-1]
        flow_delta = latest_lng - latest_ma

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest LNG Feed Gas (MMCFD)", f"{latest_lng:,.0f}", help="Daily flow to US liquefaction terminals.")
        c2.metric("30-Day Avg LNG Flow (MMCFD)", f"{latest_ma:,.0f}")
        
        signal_text = "Neutral"
        if flow_delta < -1000: # Threshold for a significant short-term outage (e.g., > 1 Bcfd)
            signal_text = "🐻 Bearish: Major Demand Loss"
        elif flow_delta > 500:
            signal_text = "🐂 Bullish: Strong Demand"
            
        c3.metric("Flow Delta vs 30-Day Avg", f"{flow_delta:,.0f} MMCFD", signal_text)
        
        # Plot LNG Flows
        st.markdown("#### LNG Feed Gas Flow and 30-Day Average (90 Days)")
        fig_lng = go.Figure()
        fig_lng.add_trace(go.Scatter(x=flows.index, y=flows['LNG_Feed_Gas_MMCFD'], name="Actual LNG Flow"))
        fig_lng.add_trace(go.Scatter(x=flows.index, y=flows['LNG_MA30'], name="30-Day Average", line=dict(dash='dash')))
        fig_lng.update_layout(title="", height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_lng, use_container_width=True)

        st.markdown("---")

        # --- Production Analysis ---
        st.markdown("#### Lower 48 Dry Gas Production (90 Days)")
        flows['Prod_MA30'] = flows['Dry_Gas_Production_MMCFD'].rolling(window=30).mean()
        latest_prod = flows['Dry_Gas_Production_MMCFD'].iloc[-1]
        latest_prod_ma = flows['Prod_MA30'].iloc[-1]
        prod_delta = latest_prod - latest_prod_ma

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Latest Production (MMCFD)", f"{latest_prod:,.0f}")
        pc2.metric("30-Day Avg Production (MMCFD)", f"{latest_prod_ma:,.0f}")
        
        prod_signal = "Neutral"
        if prod_delta > 2000: # Threshold for a new production surge (e.g., > 2 Bcfd)
            prod_signal = "🐻 Bearish: Strong Supply"
        elif prod_delta < -1000:
            prod_signal = "🐂 Bullish: Supply Drop"
            
        pc3.metric("Production Delta vs 30-Day Avg", f"{prod_delta:,.0f} MMCFD", prod_signal)

        fig_prod = go.Figure()
        fig_prod.add_trace(go.Scatter(x=flows.index, y=flows['Dry_Gas_Production_MMCFD'], name="Production"))
        fig_prod.add_trace(go.Scatter(x=flows.index, y=flows['Prod_MA30'], name="30-Day Average", line=dict(dash='dash')))
        fig_prod.update_layout(title="", height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_prod, use_container_width=True)
        
    else:
        st.info("Daily flow columns not found in the fetched data.")
else:
    st.info("Daily flow data could not be loaded from EIA API.")

st.markdown("---")

# 7. Regional Trade Screen (Original Section 6 now becomes 7)
st.markdown("### 7. Regional Trade Screen: Storage vs NOAA 7‑Day Outlook")

if not storage_points_df.empty:
    trade_rows = []
    for reg in regions_to_map:
        sid = EIA_SERIES.get(reg)
        if not sid:
            continue
        df_reg_full = get_eia_series(EIA_API_KEY, sid, length_weeks=52*15)
        if df_reg_full is None or df_reg_full.empty:
            continue
        df_reg_full = compute_storage_analytics(df_reg_full)
        latest_row = df_reg_full.iloc[-1]

        level_z = latest_row.get("level_zscore", np.nan)
        level = latest_row.get("value", np.nan)

        row_map = storage_points_df[storage_points_df["region"] == reg]
        if row_map.empty:
            continue
        row_map = row_map.iloc[0]

        trade_rows.append(
            {
                "Region": reg,
                "Storage_Bcf": level,
                "Storage_Z": level_z,
                "Pct_Full": row_map["pct_full"],
                "Temp_Bias_F": row_map.get("temp_bias", np.nan),
                "Forecast_Mean_T": row_map.get("forecast_mean_temp", np.nan),
                "Normal_T_Est": row_map.get("normal_temp_est", np.nan),
            }
        )

    if trade_rows:
        trade_df = pd.DataFrame(trade_rows)

        # Actionable Score: Lower Storage Z + Colder Weather (Negative Temp Bias) = More Bullish
        trade_df["Bullish_Score"] = -trade_df["Storage_Z"] + (-trade_df["Temp_Bias_F"] / 5.0)

        trade_df_sorted = trade_df.sort_values("Bullish_Score", ascending=False)

        display_df = trade_df_sorted.copy()
        display_df["Storage_Bcf"] = display_df["Storage_Bcf"].round(0).astype(int)
        display_df["Storage_Z"] = display_df["Storage_Z"].round(2)
        display_df["Temp_Bias_F"] = display_df["Temp_Bias_F"].round(1)
        display_df["Forecast_Mean_T"] = display_df["Forecast_Mean_T"].round(1)
        display_df["Normal_T_Est"] = display_df["Normal_T_Est"].round(1)
        display_df["Bullish_Score"] = display_df["Bullish_Score"].round(2)

        st.dataframe(
            display_df[
                [
                    "Region",
                    "Storage_Bcf",
                    "Storage_Z",
                    "Pct_Full",
                    "Forecast_Mean_T",
                    "Normal_T_Est",
                    "Temp_Bias_F",
                    "Bullish_Score",
                ]
            ],
            hide_index=True
        )

# ... (End of Code) ...
