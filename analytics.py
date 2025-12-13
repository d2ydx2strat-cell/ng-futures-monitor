# analytics.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st

from data_loader import get_eia_series, get_price_data
from constants import EIA_SERIES

# --- STORAGE ANALYTICS TRANSFORMS ---
def compute_storage_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes historical analytics for storage data (5yr avg, z-score, cumulative deviation)."""
    df = df.copy()
    df = df.sort_values('period').reset_index(drop=True)

    df['week_of_year'] = df['period'].dt.isocalendar().week.astype(int)
    df['year'] = df['period'].dt.year
    df['delta'] = df['value'].diff()

    # Base historical averages on data starting 5 years after the minimum year
    start_year = df['year'].min() + 5
    historical_df = df[df['year'] >= start_year].copy()
    
    grouped = historical_df.groupby('week_of_year')

    delta_mean = grouped['delta'].mean()
    delta_std = grouped['delta'].std(ddof=0)
    level_mean = grouped['value'].mean()
    level_std = grouped['value'].std(ddof=0)

    # Merge historical stats back to the full dataset
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

    # Gas Year starts April 1 (month >= 4)
    df['gas_year'] = np.where(df['period'].dt.month >= 4, df['period'].dt.year, df['period'].dt.year - 1)
    df['cum_dev_vs_5y'] = df.groupby('gas_year')['delta_dev_vs_5y'].cumsum()

    # Percentiles for plotting
    percentiles = grouped['value'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack(level=1)
    percentiles.columns = ['p10', 'p25', 'p50', 'p75', 'p90']
    df = df.merge(percentiles, on='week_of_year', how='left')

    return df

# --- FAIR VALUE MODEL DATA BUILD ---
@st.cache_data(ttl=3600*24)
def build_weekly_merged_dataset():
    """
    Build a weekly DataFrame with:
      - NG1 weekly close
      - Lower 48 storage level, z-score, cum deviation
      - TTF-HH spread (weekly avg)
    """
    # 1) Storage (Lower 48)
    stor_raw = get_eia_series(EIA_SERIES["Lower 48 Total"])
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
    # Align to EIA week-ending (usually Friday)
    ng_weekly = price_df['HenryHub_USD'].resample('W-FRI').last().to_frame('NG1')
    spread_weekly = price_df['Spread_TTF_HH'].resample('W-FRI').mean().to_frame('TTF_HH_Spread')

    price_weekly = ng_weekly.join(spread_weekly, how='inner')
    price_weekly.reset_index(inplace=True)
    price_weekly.rename(columns={'Date': 'week_date'}, inplace=True)

    # Placeholder for future HDD integration
    price_weekly['HDD'] = np.nan
    price_weekly['HDD_Dev'] = np.nan

    # Merge storage (EIA date) with prices (week-ending date)
    weekly = pd.merge_asof(
        stor.sort_values('week_date'),
        price_weekly.sort_values('week_date'),
        on='week_date',
        direction='backward'
    )

    weekly.dropna(subset=['NG1', 'Storage_Z', 'CumDev_Bcf', 'TTF_HH_Spread'], inplace=True)
    weekly.reset_index(drop=True, inplace=True)

    return weekly

# --- FAIR VALUE MODEL FIT ---
def fit_fair_value_model(weekly_df: pd.DataFrame):
    """
    Fit OLS: NG1 = α + β1*Storage_Z + β2*CumDev_Bcf + β3*TTF_HH_Spread
    """
    df = weekly_df.dropna(subset=['NG1', 'Storage_Z', 'CumDev_Bcf', 'TTF_HH_Spread']).copy()
    
    # Independent Variables
    X = df[['Storage_Z', 'CumDev_Bcf', 'TTF_HH_Spread']]
    X = sm.add_constant(X)
    
    # Dependent Variable
    y = df['NG1']
    
    model = sm.OLS(y, X).fit()
    df['NG1_FV'] = model.predict(X)
    df['Mispricing'] = df['NG1'] - df['NG1_FV']
    
    return model, df

# --- GEO HELPERS ---
def gdf_to_plotly_lines(gdf: gpd.GeoDataFrame):
    """Converts GeoPandas lines (pipelines) to Plotly line coordinates."""
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
