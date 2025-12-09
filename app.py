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

# Geo imports for Section 4
import geopandas as gpd
from shapely.geometry import box

# --- CONFIGURATION ---
st.set_page_config(page_title="NG Trading Monitor", layout="wide")
st.title("🔥 Global NG Spreads, Storage & Weather Monitor")

# --- 5. NEW GEO ASSETS: LNG & STORAGE LOCATIONS ---
def get_lng_terminals():
    """
    Returns a DataFrame of major US LNG Export Terminals.
    (Hardcoded as these don't change location often)
    """
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

def get_storage_centroids(storage_df_latest):
    """
    Maps the EIA Regions (which we have data for) to approximate geographic centroids 
    so we can plot 'Bubbles' of inventory on the map.
    """
    # Approximate centers of EIA storage regions
    centroids = {
        "East": {"Lat": 40.5, "Lon": -78.0}, # PA/NY area
        "Midwest": {"Lat": 41.0, "Lon": -88.0}, # IL/IN area
        "Mountain": {"Lat": 42.0, "Lon": -108.0}, # WY/CO area
        "Pacific": {"Lat": 38.0, "Lon": -121.0}, # NorCal
        "South Central Salt": {"Lat": 30.0, "Lon": -92.0}, # LA/TX Gulf Coast
        "South Central Non-Salt": {"Lat": 34.0, "Lon": -99.0}, # TX/OK Panhandle
    }
    
    data = []
    # We iterate through the specific regions we know we have keys for in EIA_SERIES
    # Note: We need the *latest* value for each region.
    # This requires fetching data for ALL regions, not just the selected one.
    # For performance, we will only map the regions if we have the data.
    
    return centroids


# IMPORTANT: Replace with your actual EIA key if the default doesn't work.
EIA_API_KEY = "KzzwPVmMSTVCI3pQbpL9calvF4CqGgEbwWy0qqXV" 

# EIA weekly working gas series IDs
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

# Path to your shapefile (ensure components are uploaded to the app directory)
SHAPEFILE_PATH = "Natural_Gas_Interstate_and_Intrastate_Pipelines.shp"

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# --- 1. DATA SOURCE: PRICES (International Spreads) ---
@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def get_price_data():
    tickers = ['NG=F', 'TTF=F']
    data = yf.download(tickers, period="1y", interval="1d")['Close']

    data.rename(columns={'NG=F': 'HenryHub_USD', 'TTF=F': 'TTF_EUR'}, inplace=True)

    # Get recent FX for EUR->USD
    fx = yf.download("EURUSD=X", period="1d", interval="1d")['Close'].iloc[-1].item()

    # Convert TTF (EUR/MWh) to USD/MMBtu approximation
    data['TTF_USD_MMBtu'] = (data['TTF_EUR'] * fx) / 3.412

    data['Spread_TTF_HH'] = data['TTF_USD_MMBtu'] - data['HenryHub_USD']

    return data.sort_index(ascending=False)

# --- 2. DATA SOURCE: US STORAGE (EIA) ---
@st.cache_data(ttl=3600*24)
def get_eia_series(api_key: str, series_id: str, length_weeks: int = 52 * 15) -> pd.DataFrame | None:
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
@st.cache_data(ttl=3600*12)  # Update weather every 12 hours
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
        # daily HDD per day approximated from hourly temps
        df['HDD'] = df['temp_f'].apply(lambda x: max(0, 65 - x) / 24)
        daily = df.groupby(df['date'].dt.date)['HDD'].sum().reset_index()
        daily['City'] = cities[i]
        results.append(daily)

    final_df = pd.concat(results)
    final_df.rename(columns={'date': 'date'}, inplace=True)
    return final_df

# --- HELPER: STORAGE ANALYTICS TRANSFORMS ---
def compute_storage_analytics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('period').reset_index(drop=True)

    df['week_of_year'] = df['period'].dt.isocalendar().week.astype(int)
    df['year'] = df['period'].dt.year
    df['delta'] = df['value'].diff()

    # Drop the first 5 years of data for a cleaner 5-year average calculation
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

# --- 4. MAP HELPERS (Pipeline Data) ---
def gdf_to_plotly_lines(gdf: gpd.GeoDataFrame):
    """
    Convert a GeoDataFrame of LineString / MultiLineString geometries
    into lon/lat lists suitable for a single Plotly Scattermapbox trace.
    """
    if gdf is None:
        return [], []
    if gdf.crs != "EPSG:4326":
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass # Use as is if projection fails

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
    """
    Load pipeline shapefile and boundary. Cached.
    """
    try:
        pipelines_gdf = gpd.read_file(SHAPEFILE_PATH)

        # Build bounding box from total extent
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

def create_satellite_map(gdf_pipelines, gdf_boundary, lng_df, storage_points_df):
    """
    Enhanced Map: Pipelines + LNG Terminals + Storage Bubbles
    """
    import os
    
    # --- 1. Mapbox Token Logic ---
    mapbox_token = None
    if "MAPBOX_TOKEN" in st.secrets:
        mapbox_token = st.secrets["MAPBOX_TOKEN"]
    if not mapbox_token:
        mapbox_token = os.getenv("MAPBOX_TOKEN")

    # Fallback to Carto if no token or invalid token
    use_satellite = False
    map_style = "carto-darkmatter"
    
    if mapbox_token and mapbox_token.startswith("pk."):
        use_satellite = True
        map_style = "satellite-streets"
    
    # --- 2. Base Geometry (Pipelines) ---
    pipeline_lons, pipeline_lats = gdf_to_plotly_lines(gdf_pipelines)
    
    try:
        gdf_boundary_4326 = gdf_boundary.to_crs(epsg=4326)
        center_point = gdf_boundary_4326.geometry.unary_union.centroid
        center_lat = center_point.y
        center_lon = center_point.x
    except:
        center_lat, center_lon = 39.8, -98.6 # Default US Center

    fig = go.Figure()

    # Layer 1: Pipelines (Thinner line for background context)
    fig.add_trace(
        go.Scattermapbox(
            mode="lines",
            lon=pipeline_lons,
            lat=pipeline_lats,
            name="Pipelines",
            line=dict(width=1, color="rgba(255, 75, 75, 0.6)"), # Streamlit Red, semi-transparent
            hoverinfo="none",
        )
    )

    # Layer 2: LNG Terminals (Green Squares)
    if not lng_df.empty:
        fig.add_trace(
            go.Scattermapbox(
                mode="markers",
                lon=lng_df['Lon'],
                lat=lng_df['Lat'],
                name="LNG Export Terminals",
                text=lng_df['Name'] + "<br>Cap: " + lng_df['Capacity_Bcfd'].astype(str) + " Bcfd",
                marker=dict(
                    size=12,
                    color='#00ff00', # Neon Green
                    symbol='square',
                    opacity=0.9
                ),
                hoverinfo='text'
            )
        )

    # Layer 3: Regional Storage Levels (Blue Bubbles)
    if storage_points_df is not None and not storage_points_df.empty:
        # Scale bubble size. Max storage ~1000 Bcf (SC Total) -> Size 40
        # Min storage ~200 Bcf -> Size 10
        fig.add_trace(
            go.Scattermapbox(
                mode="markers+text",
                lon=storage_points_df['lon'],
                lat=storage_points_df['lat'],
                name="Regional Storage (Bcf)",
                text=storage_points_df['value'].astype(int).astype(str) + " Bcf",
                textposition="bottom center",
                hovertext=storage_points_df['region'] + "<br>Current: " + storage_points_df['value'].astype(str) + " Bcf",
                marker=dict(
                    # Logarithmic scaling or simple linear scaling for size
                    size=storage_points_df['value'] / 25, 
                    sizemin=8,
                    color='#3399ff', # Light Blue
                    opacity=0.6,
                ),
                hoverinfo='text'
            )
        )

    # --- 3. Layout ---
    layout_args = dict(
        title="Infrastructure Map: Pipelines, LNG Terminals & Storage Hubs",
        mapbox=dict(
            style=map_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=3.5,
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=750,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    if use_satellite:
        layout_args["mapbox_accesstoken"] = mapbox_token

    fig.update_layout(**layout_args)
    return fig


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

    # 2A. Storage Level + Fan Chart
    fig_store = go.Figure()
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p90'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p10'], fill='tonexty', fillcolor='rgba(0, 123, 255, 0.1)', line=dict(width=0), name='10–90% band', hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p75'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p25'], fill='tonexty', fillcolor='rgba(0, 123, 255, 0.2)', line=dict(width=0), name='25–75% band', hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p50'], line=dict(color='rgba(0,0,0,0.4)', dash='dash'), name='Median (hist.)'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['value'], line=dict(color='blue', width=2), name='Actual Storage'))
    fig_store.update_layout(title=f"{selected_region} Storage vs Historical Distribution (Last 2 Years)", xaxis_title="Date", yaxis_title="Bcf", height=450, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_store, use_container_width=True)

    # 2B. Weekly Injection/Withdrawal vs 5-Year Avg
    st.markdown("#### Storage Analytics: Weekly Balances vs History (Last 2 Years)")

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Bar(x=recent['period'], y=recent['delta'], name='Actual Weekly Δ (Bcf)', marker_color=recent['delta'].apply(lambda x: 'red' if x < 0 else 'steelblue')))
    fig_delta.add_trace(go.Scatter(x=recent['period'], y=recent['delta_5y_avg'], mode='lines', name='5yr Avg Weekly Δ', line=dict(color='black', dash='dash')))
    fig_delta.update_layout(title=f"{selected_region}: Weekly Injection/Withdrawal vs 5-Year Average", xaxis_title="Date", yaxis_title="Bcf", height=400, barmode='group', margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_delta, use_container_width=True)

    # 2C. Deviation & Z-Score
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

    # 2D. Cumulative Deviation vs 5-Year Avg
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

st.markdown("---")

# --- 4. U.S. Pipeline & Infrastructure Map ---
st.subheader("4. U.S. Infrastructure Map (Pipelines, LNG, Storage)")

# A. Prepare LNG Data
lng_df = get_lng_terminals()

# B. Prepare Storage Map Data (Need to fetch current values for ALL regions)
# We use a simplified fetch here just for the latest datapoint to save API calls/time
storage_map_data = []
regions_to_map = ["East", "Midwest", "Mountain", "Pacific", "South Central Salt", "South Central Non-Salt"]
centroids = get_storage_centroids(None) # Get the dict

# Try to fetch latest data for the map (cached)
with st.spinner("Loading map layers..."):
    for reg in regions_to_map:
        # Check if we have this series ID
        if reg in EIA_SERIES:
            sid = EIA_SERIES[reg]
            # Fetch minimal history just to get latest
            df_reg = get_eia_series(EIA_API_KEY, sid, length_weeks=5) 
            if df_reg is not None and not df_reg.empty:
                latest_val = df_reg.iloc[-1]['value']
                coords = centroids.get(reg)
                if coords:
                    storage_map_data.append({
                        "region": reg,
                        "value": latest_val,
                        "lat": coords["Lat"],
                        "lon": coords["Lon"]
                    })

storage_points_df = pd.DataFrame(storage_map_data)

# C. Load Pipelines
pipelines_gdf, boundary_gdf = load_pipeline_data()

# D. Render Map
if pipelines_gdf is not None:
    fig_map = create_satellite_map(pipelines_gdf, boundary_gdf, lng_df, storage_points_df)
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Pipeline map components missing. Upload shapefile to view.")

# End of app






