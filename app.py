import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
# import os # <-- Removed: no longer needed when using st.secrets
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

        # Build bounding box from total extent
        total_bounds = pipelines_gdf.total_bounds  # [minx, miny, maxx, maxy]
        bbox_polygon = box(
            total_bounds[0],
            total_bounds[1],
            total_bounds[2],
            total_bounds[3],
        )

        boundary_gdf = gpd.GeoDataFrame(
            {"name": ["Pipeline Extent"]},
            geometry=[bbox_polygon],
            crs=pipelines_gdf.crs,
        )

        return pipelines_gdf, boundary_gdf

    except Exception as e:
        st.error(f"Error loading geospatial data: {e}")
        st.warning(
            "Ensure all required shapefile components "
            "(.shp, .shx, .dbf, .prj, .cpg) are present."
        )
        return None, None


def create_satellite_map(gdf_pipelines: gpd.GeoDataFrame,
                         gdf_boundary: gpd.GeoDataFrame):
    """
    Build a Plotly Scattermapbox figure with:
    - Satellite basemap
    - Red pipeline lines
    - Transparent boundary polygon with yellow outline
    """
    # 1) Get Mapbox token from Streamlit secrets (BEST PRACTICE)
    mapbox_token = st.secrets.get("MAPBOX_TOKEN", None)

    if not mapbox_token:
        st.error("Mapbox token not found in Streamlit Secrets.")
        st.info("Please confirm the secret **MAPBOX_TOKEN** is set in your Streamlit Cloud app settings.")
        return None

    # 2) Prepare pipeline line coordinates
    pipeline_lons, pipeline_lats = gdf_to_plotly_lines(gdf_pipelines)

    # 3) Center map on boundary centroid (in WGS84)
    gdf_boundary_4326 = gdf_boundary.to_crs(epsg=4326)
    center_point = gdf_boundary_4326.geometry.unary_union.centroid
    center_lat = center_point.y
    center_lon = center_point.x

    fig = go.Figure()

    # Pipelines as a single line trace
    fig.add_trace(
        go.Scattermapbox(
            mode="lines",
            lon=pipeline_lons,
            lat=pipeline_lats,
            name="Pipelines",
            line=dict(width=2, color="red"),
            hoverinfo="none",
        )
    )

    # Boundary polygon as a Mapbox layer
    boundary_layer = {
        "source": gdf_boundary_4326.__geo_interface__,
        "type": "fill",
        "color": "rgba(0, 255, 0, 0.0)",  # transparent fill
        "line": {"color": "yellow", "width": 1.5},
        "below": "traces",
    }

    # 4) Layout with satellite basemap
    fig.update_layout(
        title="Natural Gas Pipelines on Satellite Imagery",
        mapbox=dict(
            style="satellite-streets",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=3,
            layers=[boundary_layer],
        ),
        mapbox_accesstoken=mapbox_token,  # Token passed directly to Plotly layout
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=700,
    )

    return fig


# ---------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------
st.title("🗺️ Natural Gas Pipeline Monitor")
st.markdown("Map overlay of interstate and intrastate pipeline routes on a satellite background.")

pipelines_gdf, boundary_gdf = load_data()

if pipelines_gdf is not None and boundary_gdf is not None:
    fig = create_satellite_map(pipelines_gdf, boundary_gdf)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
