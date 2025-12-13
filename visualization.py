# visualization.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import streamlit as st
import os

from analytics import gdf_to_plotly_lines
import numpy as np


# --- SECTION 1: PRICES ---
def plot_international_spreads(price_df: pd.DataFrame):
    """Plots Henry Hub, TTF prices, and the Spread over time."""
    fig_price = make_subplots(specs=[[{"secondary_y": True}]])
    fig_price.add_trace(go.Scatter(x=price_df.index, y=price_df['HenryHub_USD'], name="Henry Hub ($)"), secondary_y=False)
    fig_price.add_trace(go.Scatter(x=price_df.index, y=price_df['TTF_USD_MMBtu'], name="TTF EU ($/MMBtu)"), secondary_y=True)
    fig_price.update_layout(
        height=400, 
        margin=dict(l=10, r=10, t=30, b=10),
        title_text="Henry Hub vs. TTF (USD/MMBtu)"
    )
    return fig_price

# --- SECTION 2: STORAGE ---
def plot_storage_level(storage_df: pd.DataFrame, region_name: str, display_window_weeks: int = 52*2):
    """Plots actual storage level vs historical distribution."""
    display_df = storage_df.tail(display_window_weeks)

    fig_store = go.Figure()
    # 10-90% band
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p90'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p10'], fill='tonexty', fillcolor='rgba(0, 123, 255, 0.1)', line=dict(width=0), name='10–90% band', hoverinfo='skip'))
    # 25-75% band
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p75'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p25'], fill='tonexty', fillcolor='rgba(0, 123, 255, 0.2)', line=dict(width=0), name='25–75% band', hoverinfo='skip'))
    # Median
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['p50'], line=dict(color='rgba(0,0,0,0.4)', dash='dash'), name='Median (hist.)'))
    # Actual
    fig_store.add_trace(go.Scatter(x=display_df['period'], y=display_df['value'], line=dict(color='blue', width=2), name='Actual Storage'))
    
    fig_store.update_layout(title=f"{region_name} Storage vs Historical Distribution (Last 2 Years)", xaxis_title="Date", yaxis_title="Bcf", height=450, margin=dict(l=10, r=10, t=40, b=10))
    return fig_store

def plot_storage_delta(storage_df: pd.DataFrame, region_name: str, display_window_weeks: int = 52*2):
    """Plots weekly injection/withdrawal vs 5-year average."""
    recent = storage_df.tail(display_window_weeks)

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Bar(x=recent['period'], y=recent['delta'], name='Actual Weekly Δ (Bcf)', marker_color=recent['delta'].apply(lambda x: 'red' if x < 0 else 'steelblue')))
    fig_delta.add_trace(go.Scatter(x=recent['period'], y=recent['delta_5y_avg'], mode='lines', name='5yr Avg Weekly Δ', line=dict(color='black', dash='dash')))
    fig_delta.update_layout(title=f"{region_name}: Weekly Injection/Withdrawal vs 5-Year Average", xaxis_title="Date", yaxis_title="Bcf", height=400, barmode='group', margin=dict(l=10, r=10, t=40, b=10))
    return fig_delta

def plot_delta_deviation(storage_df: pd.DataFrame, region_name: str, display_window_weeks: int = 52*2):
    """Plots weekly deviation vs 5-year average."""
    recent = storage_df.tail(display_window_weeks)

    fig_dev = go.Figure()
    fig_dev.add_trace(go.Bar(x=recent['period'], y=recent['delta_dev_vs_5y'], name='Δ vs 5yr Avg (Bcf)', marker_color=recent['delta_dev_vs_5y'].apply(lambda x: 'red' if x < 0 else 'green')))
    fig_dev.update_layout(title=f"{region_name}: Weekly Deviation vs 5-Year Avg (Bcf)", xaxis_title="Date", yaxis_title="Bcf", height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig_dev

def plot_delta_zscore(storage_df: pd.DataFrame, region_name: str, display_window_weeks: int = 52*2):
    """Plots weekly delta Z-Score."""
    recent = storage_df.tail(display_window_weeks)

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=recent['period'], y=recent['delta_zscore'], mode='lines+markers', name='Weekly Δ Z-Score'))
    fig_z.add_hline(y=0, line=dict(color='black', width=1))
    fig_z.add_hline(y=1.5, line=dict(color='orange', width=1, dash='dash'))
    fig_z.add_hline(y=-1.5, line=dict(color='orange', width=1, dash='dash'))
    fig_z.update_layout(title=f"{region_name}: Weekly Injection/Withdrawal Z-Score", xaxis_title="Date", yaxis_title="Z-Score", height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig_z

def plot_cumulative_deviation(storage_df: pd.DataFrame, region_name: str):
    """Plots cumulative deviation vs 5-year average by gas year."""
    fig_cum = go.Figure()
    
    # Plot last 5 gas years
    max_gy = storage_df['gas_year'].max()
    for gy, sub in storage_df.groupby('gas_year'):
        if gy >= max_gy - 4:
            fig_cum.add_trace(go.Scatter(x=sub['period'], y=sub['cum_dev_vs_5y'], mode='lines', name=f"Gas Year {gy}"))
            
    fig_cum.add_hline(y=0, line=dict(color='black', width=1))
    fig_cum.update_layout(title=f"{region_name}: Cumulative Deviation vs 5-Year Avg (by Gas Year)", xaxis_title="Date", yaxis_title="Cumulative Δ vs 5-Year Avg (Bcf)", height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig_cum

# --- SECTION 3: WEATHER ---
def plot_weather_demand(weather_df: pd.DataFrame):
    """Plots 10-day HDD forecast for key cities."""
    chart_data = weather_df.pivot(index='date', columns='City', values='HDD')
    fig = go.Figure()
    for col in chart_data.columns:
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data[col], mode='lines+markers', name=col))
    
    fig.update_layout(title="10-Day HDD Forecast (Gas Demand Proxy)", xaxis_title="Date", yaxis_title="Daily HDD", height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# --- SECTION 4: FAIR VALUE MODEL ---
def plot_fair_value(fv_df: pd.DataFrame):
    """Plots actual NG1 price vs. model-derived fair value."""
    fig_fv = go.Figure()
    fig_fv.add_trace(go.Scatter(x=fv_df['week_date'], y=fv_df['NG1'], name="NG1 Actual", line=dict(color='blue')))
    fig_fv.add_trace(go.Scatter(x=fv_df['week_date'], y=fv_df['NG1_FV'], name="Model Fair Value", line=dict(color='orange')))
    fig_fv.update_layout(title="NG1 vs Storage-Based Fair Value", xaxis_title="Week", yaxis_title="$/MMBtu", height=450, margin=dict(l=10, r=10, t=40, b=10))
    return fig_fv

def plot_mispricing(fv_df: pd.DataFrame):
    """Plots the weekly mispricing (Actual - Fair Value)."""
    fig_mis = go.Figure()
    fig_mis.add_trace(go.Bar(x=fv_df['week_date'], y=fv_df['Mispricing'], name="Mispricing (NG1 - FV)", marker_color=fv_df['Mispricing'].apply(lambda x: 'red' if x < 0 else 'green')))
    fig_mis.add_hline(y=0, line=dict(color='black', width=1))
    fig_mis.update_layout(title="NG1 Mispricing vs Fair Value", xaxis_title="Week", yaxis_title="$/MMBtu", height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig_mis

# --- SECTION 6: GOOGLE HOURLY FORECAST MAP ---
def plot_forecast_map_with_slider(weather_df: pd.DataFrame, current_timestamp: datetime.datetime):
    """
    Plots the Google hourly weather forecast for a selected time on a map.
    """
    # Filter data for the selected timestamp
    df_filtered = weather_df[weather_df['Date_Time'] == current_timestamp].copy()

    if df_filtered.empty:
        # Fallback figure
        fig = go.Figure()
        fig.update_layout(title="No data available for selected time.", height=750)
        return fig

    # Determine colorscale properties (e.g., min/max temp)
    temp_min = weather_df['Temperature_F'].min()
    temp_max = weather_df['Temperature_F'].max()
    
    # Define a reasonable clip range for color scaling
    clip_min = max(-10, temp_min - 5)
    clip_max = min(100, temp_max + 5)
    
    df_filtered["Temperature_F_Clipped"] = df_filtered["Temperature_F"].clip(clip_min, clip_max)

    fig = go.Figure(go.Scattermapbox(
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
    
    # Use a simple map style for faster rendering without token dependency
    fig.update_layout(
        title=f"Hourly AI Weather Forecast: {current_timestamp.strftime('%Y-%m-%d %H:%M %Z')}",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=df_filtered['Latitude'].mean(), lon=df_filtered['Longitude'].mean()),
            zoom=3,
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=750,
    )
    
    return fig

# --- SECTION 5: INFRASTRUCTURE MAP ---
def create_satellite_map(
    gdf_pipelines: gpd.GeoDataFrame,
    gdf_boundary: gpd.GeoDataFrame,
    lng_df: pd.DataFrame,
    storage_points_df: pd.DataFrame,
    show_noaa: bool = True,
    ow_forecast_for_map: pd.DataFrame | None = None
):
    """Generates the main US infrastructure and weather map."""
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

    # Pipelines
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=pipeline_lons,
        lat=pipeline_lats,
        name="Pipelines",
        line=dict(width=1, color="rgba(255, 50, 50, 0.6)"),
        hoverinfo="none"
    ))

    # LNG terminals
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

    # Storage regions (size ~ volume, text = % full)
    if storage_points_df is not None and not storage_points_df.empty:
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=storage_points_df['lon'],
            lat=storage_points_df['lat'],
            name="Regional Storage",
            text=storage_points_df['pct_full'],
            textposition="middle center",
            textfont=dict(size=11, color="white", weight="bold"),
            hovertext=(
                storage_points_df['region']
                + "<br>Vol: " + storage_points_df['value'].astype(int).astype(str) + " Bcf"
                + "<br>Full: " + storage_points_df['pct_full']
            ),
            marker=dict(
                size=storage_points_df['value'] / 12, 
                sizemin=15,
                color='#003366',
                opacity=0.8,
            ),
            hoverinfo='text'
        ))

    # NOAA 7d anomaly layer
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

    # OpenWeather 14-day forecast overlay (per selected day)
    if ow_forecast_for_map is not None and not ow_forecast_for_map.empty:
        df_ow = ow_forecast_for_map.copy()
        df_ow["temp_f_clipped"] = df_ow["temp_f"].clip(-10, 100)

        fig.add_trace(
            go.Scattermapbox(
                mode="markers",
                lon=df_ow["lon"],
                lat=df_ow["lat"],
                name="OpenWeather 14d Forecast (Temp)",
                hovertext=(
                    df_ow["region"]
                    + "<br>Date: " + df_ow["date"].astype(str)
                    + "<br>Tmean: " + df_ow["temp_f"].round(1).astype(str) + "°F"
                    + "<br>Tmin/Tmax: "
                    + df_ow["temp_min_f"].round(1).astype(str) + " / "
                    + df_ow["temp_max_f"].round(1).astype(str) + "°F"
                    + "<br>HDD: " + df_ow["HDD"].round(1).astype(str)
                    + "<br>POP: " + (df_ow["pop"].fillna(0) * 100).round(0).astype(str) + "%"
                ),
                marker=dict(
                    size=26,
                    color=df_ow["temp_f_clipped"],
                    colorscale="Turbo",
                    cmin=-10,
                    cmax=100,
                    opacity=0.75,
                ),
                hoverinfo="text",
            )
        )

    layout_args = dict(
        title="US Natural Gas Infrastructure, Storage, NOAA 7‑Day & 14‑Day Forecast",
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
