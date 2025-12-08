import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
import os
from shapely.geometry import box

# --- Configuration ---
# Set the page to wide mode for a better map view
st.set_page_config(layout="wide", page_title="NG Pipeline Monitor")

# Name of the main shapefile (ensure all component files are in the same directory)
SHAPEFILE_PATH = "Natural_Gas_Interstate_and_Intrastate_Pipelines.shp"

# --- Helper Functions ---

# Function to prepare LineString data for Plotly
def gdf_to_plotly_lines(gdf):
    """
    Converts a GeoPandas GeoDataFrame of LineStrings into a format 
    (separate lists of longitudes and latitudes) suitable for a single 
    Plotly Scattermapbox trace. None is inserted to break lines.
    """
    # Reproject to WGS84 (EPSG:4326) which Mapbox/Plotly uses
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    lons = []
    lats = []
    
    # Iterate over geometry to build a continuous trace with None separators
    for geom in gdf.geometry:
        if geom.geom_type in ['LineString', 'MultiLineString']:
            # Handle both single and multi-line geometries
            lines = [geom] if geom.geom_type == 'LineString' else geom.geoms
            for line in lines:
                x, y = line.xy
                lons.extend(x.tolist())
                lats.extend(y.tolist())
                lons.append(None)  # Use None to break the line segment
                lats.append(None)
    
    return lons, lats

@st.cache_data
def load_data():
    """Loads the shapefile once and caches the result."""
    try:
        # Load the pipeline data (assuming the shapefile is in the same directory)
        pipelines_gdf = gpd.read_file(SHAPEFILE_PATH)
        
        # Create a simple bounding box polygon from the pipeline extent
        # This will be used as the 'boundary' layer and for centering the map
        total_bounds = pipelines_gdf.total_bounds
        # The bounding box is created in the shapefile's original CRS
        bbox_polygon = box(total_bounds[0], total_bounds[1], total_bounds[2], total_bounds[3])
        
        # Create a single-row GeoDataFrame for the boundary
        boundary_gdf = gpd.GeoDataFrame(
            {'name': ['Pipeline Extent']}, 
            geometry=[bbox_polygon], 
            crs=pipelines_gdf.crs
        )
        
        return pipelines_gdf, boundary_gdf
    except Exception as e:
        st.error(f"Error loading geospatial data: {e}")
        st.warning("Ensure all required shapefile components (.shp, .shx, .dbf, .prj, .cpg) are present.")
        return None, None

def create_satellite_map(gdf_pipelines, gdf_boundary):
    # Get token from secrets.toml (Streamlit will load it into os.environ)
    mapbox_token = os.environ.get("MAPBOX_TOKEN")

    if not mapbox_token:
        st.error("Mapbox Token not found.")
        st.info("Please set **MAPBOX_TOKEN** in a `.streamlit/secrets.toml` file.")
        return None

    # --- 1. Prepare Pipeline Data ---
    pipeline_lons, pipeline_lats = gdf_to_plotly_lines(gdf_pipelines)

    # --- 2. Calculate Map Center (for initial view) ---
    # Need to convert the boundary to 4326 (WGS84) for lat/lon calculations
    gdf_boundary_4326 = gdf_boundary.to_crs(epsg=4326)
    
    # Calculate the centroid of the boundary for the map center
    center_point = gdf_boundary_4326.geometry.unary_union.centroid
    center_lat = center_point.y
    center_lon = center_point.x

    # --- 3. Create the Plotly Figure ---
    fig = go.Figure()

    # Add Pipeline Lines (Scattermapbox trace)
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=pipeline_lons,
        lat=pipeline_lats,
        name='Pipelines',
        line=dict(width=2, color='red'),
        hoverinfo='none'
    ))

    # --- 4. Define Boundary as a Mapbox Layer (Polygons) ---
    # Use __geo_interface__ to easily convert the GeoDataFrame to GeoJSON dictionary
    boundary_layer = {
        'source': gdf_boundary_4326.__geo_interface__,
        'type': 'fill',
        'color': 'rgba(0, 255, 0, 0.0)', # Transparent fill for boundary
        'line': {'color': 'yellow', 'width': 1.5}, # Visible yellow boundary line
        'below': 'traces' # Ensures the boundary is beneath the red pipeline lines
    }

    # --- 5. Update Layout for Satellite Background ---
    fig.update_layout(
        title="Natural Gas Pipelines on Satellite Imagery",
        mapbox=dict(
            accesstoken=mapbox_token,
            # 'satellite-streets' is the Google Earth-like style
            style="satellite-streets",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=3, # Adjusted zoom level for a good initial continental view
            layers=[boundary_layer]
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=700
    )

    return fig

# --- Streamlit Main Logic ---
st.title("🗺️ Natural Gas Pipeline Monitor")
st.markdown("Map overlay of pipeline routes on a satellite background.")

pipelines_gdf, boundary_gdf = load_data()

if pipelines_gdf is not None:
    # Generate and display the satellite map
    satellite_fig = create_satellite_map(pipelines_gdf, boundary_gdf)
    
    if satellite_fig:
        st.plotly_chart(satellite_fig, use_container_width=True)
