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

# --- CONFIGURATION ---
st.set_page_config(page_title="NG Trading Monitor", layout="wide")
st.title("🔥 Global NG Spreads, Storage & Weather Monitor")

EIA_API_KEY = "KzzwPVmMSTVCI3pQbpL9calvF4CqGgEbwWy0qqXV"

# EIA weekly working gas series (v2 'series' facet values)
# Taken directly from your CSV dump
EIA_SERIES_V2 = {
    "Lower 48 Total":         "NW2_EPG0_SWO_R48_BCF",
    "East":                   "NW2_EPG0_SWO_R31_BCF",
    "Midwest":                "NW2_EPG0_SWO_R32_BCF",
    # EIA labels this as South Central Region in your CSV
    "South Central Total":    "NW2_EPG0_SWO_R33_BCF",
    "Mountain":               "NW2_EPG0_SWO_R34_BCF",
    "Pacific":                "NW2_EPG0_SWO_R35_BCF",
    # Salt / Nonsalt regions (Salt Region / Nonsalt Region in CSV)
    "South Central Salt":     "NW2_EPG0_SSO_R33_BCF",
    "South Central Non-Salt": "NW2_EPG0_SNO_R33_BCF",
}

# Optional: working gas capacity by region (Bcf) if you have it
REGION_CAPACITY_BCF = {
    "Lower 48 Total": None,
    "East": None,
    "Midwest": None,
    "South Central Total": None,
    "Mountain": None,
    "Pacific": None,
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
@st.cache_data(ttl=3600 * 24)
def get_price_data():
    """
    Fetches Henry Hub (US) and a proxy for TTF (EU) from Yahoo Finance.
    """
    tickers = ["NG=F", "TTF=F"]
    data = yf.download(tickers, period="1y", interval="1d")["Close"]

    data.rename(columns={"NG=F": "HenryHub_USD", "TTF=F": "TTF_EUR"}, inplace=True)

    # FX for EURUSD
    fx = yf.download("EURUSD=X", period="1d", interval="1d")["Close"].iloc[-1].item()

    # Convert TTF EUR/MWh to USD/MMBtu (1 MWh = 3.412 MMBtu)
    data["TTF_USD_MMBtu"] = (data["TTF_EUR"] * fx) / 3.412

    # Spread
    data["Spread_TTF_HH"] = data["TTF_USD_MMBtu"] - data["HenryHub_USD"]

    return data.sort_index(ascending=False)

# --- 2. DATA SOURCE: US STORAGE (EIA v2) ---
@st.cache_data(ttl=3600 * 24)
def get_eia_storage_v2(api_key: str, series_id: str, length_weeks: int = 52 * 15) -> pd.DataFrame | None:
    """
    Fetch weekly storage from EIA v2 API using the 'series' facet.
    series_id is like 'NW2_EPG0_SWO_R48_BCF', 'NW2_EPG0_SSO_R33_BCF', etc.
    """
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": series_id,
