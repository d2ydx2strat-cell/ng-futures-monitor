# app.py (New Main Application File)

import streamlit as st
import pandas as pd
import numpy as np
import os # <-- ADDED: For secure API key retrieval

# Import functions from custom modules
from constants import (EIA_SERIES, REGION_CAPACITY_BCF, get_storage_centroids, GOOGLE_WEATHER_API_KEY)
from data_loader import (
    get_price_data,
    get_eia_series,
    get_weather_demand,
    get_noaa_temp_anomaly_by_region,
    get_openweather_forecast_for_storage_regions,
    load_pipeline_data,
    get_google_weather_forecast, # <-- ASSUMED/RETAINED
)
from analytics import (
    compute_storage_analytics,
    build_weekly_merged_dataset,
    fit_fair_value_model,
)
from visualization import (
    plot_international_spreads,
    plot_storage_level,
    plot_storage_delta,
    plot_delta_deviation,
    plot_delta_zscore,
    plot_cumulative_deviation,
    plot_weather_demand,
    plot_fair_value,
    plot_mispricing,
    create_satellite_map,
    plot_forecast_map_with_slider, # <-- ADDED: New function for Google Map
)


# --- CONFIGURATION ---
st.set_page_config(page_title="NG Trading Monitor", layout="wide")
st.title("🔥 Global NG Spreads, Storage & Weather Monitor")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    if st.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    show_noaa = st.checkbox("Show NOAA 7‑Day Temp Anomaly on Map", value=True)

# --- 1. International Spreads ---
st.subheader("1. International Future Spreads (Arbitrage Window)")
try:
    price_df = get_price_data()
    latest = price_df.iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Henry Hub (US)", f"${latest['HenryHub_USD']:.2f}", f"{price_df['HenryHub_USD'].diff().iloc[-1]:.2f}")
    col2.metric("TTF Proxy (EU)", f"${latest['TTF_USD_MMBtu']:.2f}", f"{price_df['TTF_USD_MMBtu'].diff().iloc[-1]:.2f}")
    col3.metric("Spread (Export Arb)", f"${latest['Spread_TTF_HH']:.2f}", "High spread = Bullish US LNG")

    fig_price = plot_international_spreads(price_df)
    st.plotly_chart(fig_price, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load price data (Yahoo Finance might be throttling): {e}")

st.markdown("---")

# --- 2. US Storage Levels (EIA Weekly) ---
st.subheader("2. US Storage Levels (EIA Weekly)")

region_names = list(EIA_SERIES.keys())
default_region = "Lower 48 Total"
selected_region = st.selectbox("Select Region / South Central Detail", region_names, index=region_names.index(default_region))

series_id = EIA_SERIES[selected_region]
capacity_bcf = REGION_CAPACITY_BCF.get(selected_region)

storage_df = get_eia_series(series_id)

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

    # Plots
    st.plotly_chart(plot_storage_level(storage_df, selected_region), use_container_width=True)

    st.markdown("#### Storage Analytics: Weekly Balances vs History (Last 2 Years)")

    st.plotly_chart(plot_storage_delta(storage_df, selected_region), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_delta_deviation(storage_df, selected_region), use_container_width=True)

    with c2:
        st.plotly_chart(plot_delta_zscore(storage_df, selected_region), use_container_width=True)

    st.plotly_chart(plot_cumulative_deviation(storage_df, selected_region), use_container_width=True)

else:
    st.warning(f"⚠️ Could not load storage data for {selected_region}.")

st.markdown("---")

# --- 3. Weather (10-day HDD) ---
st.subheader("3. 10-Day HDD Forecast (Gas Demand Proxy)")
st.write("Projected Heating Degree Days (HDD) for key consumption hubs.")
try:
    weather_df = get_weather_demand()
    st.plotly_chart(plot_weather_demand(weather_df), use_container_width=True)
    
    chart_data = weather_df.pivot(index='date', columns='City', values='HDD')
    total_hdd = chart_data.sum(axis=1)
    st.metric("Total System Forecast HDD (Next 10 Days)", f"{total_hdd.sum():.0f}")
except Exception as e:
    st.error(f"Weather data error: {e}")

st.markdown("---")

# --- 4. FAIR VALUE MODEL SECTION ---
st.subheader("4. Storage-Based Fair Value Model for NG1")

weekly_df = build_weekly_merged_dataset()
if weekly_df is None or weekly_df.empty:
    st.info("Insufficient data to build fair value model.")
else:
    model, fv_df = fit_fair_value_model(weekly_df)

    latest_row = fv_df.iloc[-1]
    mispricing = latest_row['Mispricing']
    mispricing_pctile = (fv_df['Mispricing'] <= mispricing).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Current NG1", f"${latest_row['NG1']:.2f}")
    c2.metric("Model Fair Value", f"${latest_row['NG1_FV']:.2f}", f"Mispricing: {mispricing:+.2f}")
    c3.metric("Mispricing Percentile", f"{mispricing_pctile:.0f}th", "vs last 5–10 years")

    st.plotly_chart(plot_fair_value(fv_df), use_container_width=True)
    st.plotly_chart(plot_mispricing(fv_df), use_container_width=True)

st.markdown("---")

# --- 5. U.S. Infrastructure Map ---
st.subheader("5. U.S. Infrastructure Map (Pipelines, LNG, Storage, NOAA Outlook & 14‑Day Forecast)")

# Load Geo Data
pipelines_gdf, boundary_gdf, lng_df = load_pipeline_data()

if pipelines_gdf is None:
    st.info("Pipeline map components missing.")
else:
    # Prepare Storage & Weather Data for Map
    regions_to_map = ["East", "Midwest", "Mountain", "Pacific", "South Central Salt", "South Central Non-Salt"]
    centroids = get_storage_centroids()
    
    # get_noaa_temp_anomaly_by_region should likely take the centroids dictionary
    # Assuming the implementation handles it if called without args, 
    # but explicitly passing centroids is safer if the function is complex.
    noaa_regional_df = get_noaa_temp_anomaly_by_region(centroids)
    
    storage_map_data = []
    with st.spinner("Loading regional storage levels..."):
        for reg in regions_to_map:
            if reg in EIA_SERIES:
                sid = EIA_SERIES[reg]
                # Only need recent level for map, fetch minimal history
                df_reg = get_eia_series(sid, length_weeks=5)
                
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
        # Ensure columns exist even if no data
        storage_points_df["temp_bias"] = np.nan
        storage_points_df["forecast_mean_temp"] = np.nan
        storage_points_df["normal_temp_est"] = np.nan

    # OpenWeather 14-day forecast + slider
    # get_openweather_forecast_for_storage_regions should likely take the centroids dictionary
    ow_forecast_df = get_openweather_forecast_for_storage_regions(centroids, days=14) 
    ow_forecast_for_map = pd.DataFrame()

    if not ow_forecast_df.empty:
        forecast_dates = sorted(ow_forecast_df["date"].unique())
        idx = st.slider(
            "OpenWeather 14-Day Forecast – Select Day",
            min_value=0,
            max_value=len(forecast_dates) - 1,
            value=0,
            format="Day %d"
        )
        selected_forecast_date = forecast_dates[idx]

        st.caption(
            f"Showing OpenWeather forecast for **{selected_forecast_date}** "
            f"(storage regions colored by forecast mean temperature)."
        )

        ow_forecast_for_map = ow_forecast_df[ow_forecast_df["date"] == selected_forecast_date].copy()
    else:
        st.info("OpenWeather forecast not available (OPENWEATHER_API_KEY missing).")


    fig_map = create_satellite_map(
        pipelines_gdf,
        boundary_gdf,
        lng_df,
        storage_points_df,
        show_noaa=show_noaa,
        ow_forecast_for_map=ow_forecast_for_map
    )
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

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
3. Add the Display Section (Section 6)
The original issue was due to missing files/imports, which led to the removal of the entire Section 6. Now you need to re-add the new Section 6 to app.py (after the map section, around line 770).

Python

# app.py

# ... (End of Section 5 Map Logic) ...

# 6. GOOGLE AI FORECAST MAP (Hourly with Time Slider)
st.markdown("---")
st.subheader("6. Google AI 10-Day Hourly Forecast (High-Resolution)")

hourly_forecast_df = get_google_weather_forecast(centroids, GOOGLE_WEATHER_API_KEY)

if not hourly_forecast_df.empty:
    all_timestamps = sorted(hourly_forecast_df["Date_Time"].unique())
    
    time_slider_idx = st.slider(
        "Select Time Slot",
        min_value=0,
        max_value=len(all_timestamps) - 1,
        value=0,
        format="%Y-%m-%d %H:%M %Z"
    )
    selected_timestamp = all_timestamps[time_slider_idx]

    # You will need to re-add the plot_forecast_map_with_slider function 
    # to your visualization logic, similar to the OpenWeather map function.
    # Since you don't have a visualization.py, you need to integrate this:
    
    # 6A. Display the Map (Inline Plotting Function)
    
    df_filtered = hourly_forecast_df[hourly_forecast_df['Date_Time'] == selected_timestamp].copy()
    
    if not df_filtered.empty:
        temp_min = hourly_forecast_df['Temperature_F'].min()
        temp_max = hourly_forecast_df['Temperature_F'].max()
        clip_min = max(-10, temp_min - 5)
        clip_max = min(100, temp_max + 5)
        df_filtered["Temperature_F_Clipped"] = df_filtered["Temperature_F"].clip(clip_min, clip_max)

        fig_google = go.Figure(go.Scattermapbox(
            mode="markers",
            lon=df_filtered['Longitude'],
            lat=df_filtered['Latitude'],
            text=df_filtered['Region'] + "<br>" + df_filtered['Temperature_F'].round(1).astype(str) + "°F",
            hoverinfo='text',
            marker=dict(
                size=20,
                color=df_filtered['Temperature_F_Clipped'],
                colorscale='Jet',
                cmin=clip_min,
                cmax=clip_max,
                colorbar=dict(title="Temp (°F)", thickness=10),
                opacity=0.8
            )
        ))

        fig_google.update_layout(
            title=f"Hourly AI Weather Forecast: {selected_timestamp.strftime('%Y-%m-%d %H:%M %Z')}",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=df_filtered['Latitude'].mean(), lon=df_filtered['Longitude'].mean()),
                zoom=3,
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=750,
        )
        st.plotly_chart(fig_google, use_container_width=True)
    else:
        st.info("No data available for selected time.")
else:
    st.warning("Could not load Google Hourly Forecast data.")

st.markdown("---")

# --- 6. GOOGLE AI FORECAST MAP (Hourly with Time Slider) ---
st.subheader("6. Google AI Forecast Map (Hourly with Time Slider)")
st.caption("Visualizing hourly temperature forecasts for key US storage regions over the next 10 days using Google's Weather API.")

# 1. Retrieve the API Key securely using Streamlit secrets
# Use st.secrets first, then fall back to environment variable if needed
GOOGLE_WEATHER_API_KEY = st.secrets.get("GOOGLE_WEATHER_API_KEY", os.getenv("GOOGLE_WEATHER_API_KEY"))

if not GOOGLE_WEATHER_API_KEY:
    st.warning(
        "**Google Weather API Key Missing.** Please set your Google Weather API Key "
        "in your Streamlit secrets file (`.streamlit/secrets.toml`) or "
        "as a secret in your cloud deployment named `GOOGLE_WEATHER_API_KEY`."
    )
else:
    # 2. Fetch the data, passing the key and all centroids (from Section 5)
    with st.spinner("Fetching 10-day hourly AI weather forecast..."):
        weather_forecast_df = get_google_weather_forecast(
            locations_dict=centroids, 
            api_key=GOOGLE_WEATHER_API_KEY
        )
    
    if not weather_forecast_df.empty:
        # 3. Visualize the data using the new plot function
        forecast_map_fig = plot_forecast_map_with_slider(weather_forecast_df)
        st.plotly_chart(forecast_map_fig, use_container_width=True)
    else:
        st.error("Could not retrieve hourly forecast data. Check your Google Weather API Key and ensure the Weather API is enabled in your Google Cloud Project with billing activated.")

st.markdown("---")

# --- 7. Regional Trade Screen (Renumbered) ---
st.markdown("### 7. Regional Trade Screen: Storage vs NOAA 7‑Day Outlook") # <-- RENAMED SECTION 6 TO 7

if not storage_points_df.empty:
    trade_rows = []
    for reg in regions_to_map:
        sid = EIA_SERIES.get(reg)
        if not sid:
            continue
        # Get full history for accurate Z-score calculation
        df_reg_full = get_eia_series(sid, length_weeks=52*15)
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
        # Higher Bullish_Score is better for a long trade
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
            use_container_width=True,
        )

        st.caption(
            "Higher Bullish_Score = more supportive for long NG "
            "(low storage vs history + colder‑than‑normal NOAA 7‑day outlook). "
            "Negative scores tilt bearish (high storage + warm anomaly)."
        )
    else:
        st.info("Insufficient data to build regional trade screen.")
else:
    st.info("No storage map data available for trade screen.")
