import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
from shapely.geometry import box

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="NG Pipeline Monitor"
)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
SHAPEFILE_PATH = "Natural_Gas_Interstate_and_Intrastate_Pipelines.shp"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def gdf_to_plotly_lines(gdf: gpd.GeoDataFrame):
    """
    Convert a GeoDataFrame of LineString / MultiLineString geometries
    into lon/lat lists suitable for a single Plotly Scattermapbox trace.
    Uses None separators between segments.
    """
    # Ensure WGS84 for Mapbox
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    lons = []
    lats = []

    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type in ["LineString", "MultiLineString"]:
            lines = [geom] if geom.geom_type == "LineString" else geom.geoms
            for line in lines:
                x, y = line.xy
                lons.extend(x.tolist())
                lats.extend(y.tolist())
                lons.append(None)
                lats.append(None)

    return lons, lats


@st.cache_data
def load_data():
    """
    Load the pipeline shapefile and build a simple boundary GeoDataFrame.
    Cached so it only loads once per session.
    """
    try:
        pipelines_gdf = gpd.read_file(SHAPEFILE_PATH)

        # Build bounding box f
