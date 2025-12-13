# constants.py

import os
import streamlit as st

# --- API KEYS ---
# NOTE: In a real-world scenario, keys should be loaded from environment
# variables or Streamlit secrets, not hardcoded.
EIA_API_KEY = "KzzwPVmMSTVCI3pQbpL9calvF4CqGgEbwWy0qqXV"
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY")
GOOGLE_WEATHER_API_KEY = st.secrets.get("GOOGLE_WEATHER_API_KEY", os.getenv("GOOGLE_WEATHER_API_KEY", None))

# --- EIA STORAGE SERIES IDs ---
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

# --- REGIONAL STORAGE CAPACITY (BCF) ---
REGION_CAPACITY_BCF = {
    "Lower 48 Total": 4000, # Estimated/Placeholder
    "East": 950,
    "Midwest": 1100,
    "Mountain": 270,
    "Pacific": 420,
    "South Central Salt": 490,
    "South Central Non-Salt": 980,
    "South Central Total": 1470
}

# --- GEO ASSETS ---
SHAPEFILE_PATH = "Natural_Gas_Interstate_and_Intrastate_Pipelines.shp"

def get_lng_terminals():
    """LNG Terminal Data"""
    return [
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

def get_storage_centroids():
    """Storage Region Centroids"""
    return {
        "East": {"Lat": 40.5, "Lon": -78.0},
        "Midwest": {"Lat": 41.0, "Lon": -88.0},
        "Mountain": {"Lat": 42.0, "Lon": -108.0},
        "Pacific": {"Lat": 38.0, "Lon": -121.0},
        "South Central Salt": {"Lat": 30.0, "Lon": -92.0},
        "South Central Non-Salt": {"Lat": 34.0, "Lon": -99.0},
    }

# --- WEATHER CITIES FOR HDD FORECAST ---
HDD_CITIES = {
    "Chicago": {"lat": 41.85, "lon": -87.62},
    "New York": {"lat": 40.71, "lon": -74.00},
    "Houston": {"lat": 29.76, "lon": -95.36},
}
