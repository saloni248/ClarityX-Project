import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="US Real Estate Dashboard", layout="wide")
st.title("🏠 US Real Estate Asset Dashboard")

# ------------------------------------------------------------
# Caching Functions for Faster Load
# ------------------------------------------------------------
@st.cache_data
def load_excel(file_name: str):
    return pd.read_excel(file_name)

@st.cache_data
def load_csv(file_name: str):
    return pd.read_csv(file_name)

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
file_name = "assets_with_model_predictions.xlsx"
pred_file = "predictions.csv"

try:
    df = load_excel(file_name)
except Exception as e:
    st.error(f"❌ Error reading {file_name}: {e}")
    st.stop()

try:
    pred_df = load_csv(pred_file)
except Exception as e:
    st.error(f"❌ Error reading {pred_file}: {e}")
    st.stop()

# ------------------------------------------------------------
# Validate Columns
# ------------------------------------------------------------
required_columns = [
    "real property asset name", "zip code", "price_latest",
    "City", "State", "latitude", "longitude",
    "building status", "real property asset type", "predicted_price_model"
]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing columns in dataset: {missing_cols}")
    st.write("✅ Columns found in file:", list(df.columns))
    st.stop()

if not {"latitude", "longitude", "Estimated_Price"}.issubset(pred_df.columns):
    st.error("❌ Predictions CSV must include 'latitude', 'longitude', and 'Estimated_Price' columns.")
    st.stop()

# ------------------------------------------------------------
# Create Price Bins
# ------------------------------------------------------------
df["Price Bin"] = pd.qcut(
    df["predicted_price_model"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# ------------------------------------------------------------
# Sidebar Filters
# ------------------------------------------------------------
st.sidebar.header("🔎 Dashboard Filters")

building_options = st.sidebar.multiselect(
    "Building Status",
    options=df["building status"].dropna().unique().tolist(),
    default=df["building status"].dropna().unique().tolist()
)

asset_type_options = st.sidebar.multiselect(
    "Asset Type",
    options=df["real property asset type"].dropna().unique().tolist(),
    default=df["real property asset type"].dropna().unique().tolist()
)

price_bin_options = st.sidebar.multiselect(
    "Predicted Price Category",
    options=df["Price Bin"].dropna().unique().tolist(),
    default=df["Price Bin"].dropna().unique().tolist()
)

# ------------------------------------------------------------
# Filtered Data
# ------------------------------------------------------------
filtered_df = df[
    (df["building status"].isin(building_options)) &
    (df["real property asset type"].isin(asset_type_options)) &
    (df["Price Bin"].isin(price_bin_options))
]

# ------------------------------------------------------------
# Dashboard KPIs
# ------------------------------------------------------------
st.subheader("📊 Dashboard Summary (Based on Filters)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Assets", len(filtered_df))

with col2:
    avg_price = filtered_df["predicted_price_model"].mean()
    st.metric("Avg Predicted Price", f"${avg_price:,.0f}")

with col3:
    total_sqft = filtered_df.get("building rentable square feet", pd.Series([0])).fillna(0).sum()
    st.metric("Total Rentable SqFt", f"{total_sqft:,.0f}")

# ------------------------------------------------------------
# Visual Insights
# ------------------------------------------------------------
st.subheader("📈 Visual Insights")
colA, colB, colC = st.columns(3)

# Pie Chart - Asset Type Distribution
with colA:
    fig_pie = px.pie(
        filtered_df,
        names="real property asset type",
        title="Asset Type Distribution",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Heatmap - Price vs GSA Region
with colB:
    heatmap_data = filtered_df.groupby("gsa region")["predicted_price_model"].mean().reset_index()
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data["predicted_price_model"],
        x=heatmap_data["gsa region"],
        y=["Avg Predicted Price"] * len(heatmap_data),
        colorscale="Viridis"
    ))
    fig_heat.update_layout(title="Avg Predicted Price by GSA Region", xaxis_title="GSA Region")
    st.plotly_chart(fig_heat, use_container_width=True)

# Scatter Plot - Predicted vs Latest Price
with colC:
    filtered_df["building rentable square feet"] = filtered_df.get("building rentable square feet", 0).fillna(0)
    fig_scatter = px.scatter(
        filtered_df,
        x="price_latest",
        y="predicted_price_model",
        size="building rentable square feet",
        color="Price Bin",
        hover_name="real property asset name",
        title="Predicted vs Latest Price (Bubble = SqFt)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------------
# Extra Graph Row
# ------------------------------------------------------------
colX, colY, colZ = st.columns(3)

# Bar Chart - Top Cities by Asset Count
with colX:
    city_count = filtered_df["City"].value_counts().nlargest(10).reset_index()
    city_count.columns = ["City", "Count"]
    fig_bar = px.bar(
        city_count,
        x="City",
        y="Count",
        title="Top 10 Cities by Asset Count",
        color="Count",
        color_continuous_scale="OrRd"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Histogram - Distribution of Predicted Prices
with colY:
    fig_hist = px.histogram(
        filtered_df,
        x="predicted_price_model",
        nbins=20,
        title="Distribution of Predicted Prices",
        color_discrete_sequence=["#857D0B"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Box Plot - Predicted Price by Asset Type
with colZ:
    fig_box = px.box(
        filtered_df,
        x="real property asset type",
        y="predicted_price_model",
        title="Price Distribution by Asset Type",
        color="real property asset type",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------------------------
# Optimized Folium Maps
# ------------------------------------------------------------
MAX_POINTS = 500  # To avoid freezing Streamlit

st.subheader("🗺️ US Real Estate Asset Map (Sampled)")
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
for _, row in df.head(MAX_POINTS).iterrows():
    popup_text = f"""
    <b>Asset:</b> {row['real property asset name']}<br>
    <b>Status:</b> {row['building status']}<br>
    <b>Type:</b> {row['real property asset type']}<br>
    <b>Predicted Price:</b> ${row['predicted_price_model']:,}<br>
    <b>Zip:</b> {row['zip code']}<br>
    <b>Location:</b> {row['City']}, {row['State']}
    """
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color="purple",
        fill=True,
        fill_color="skyblue",
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(m)

st_folium(m, width=1200, height=700)

# ------------------------------------------------------------
# Predictions Map (Also Limited for Performance)
# ------------------------------------------------------------
st.subheader("🗺️ Estimated Price Map (Sampled Predictions)")
map_pred = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
for _, row in pred_df.head(MAX_POINTS).iterrows():
    popup_text = f"""
    <b>Asset:</b> {row.get('real property asset name', 'N/A')}<br>
    <b>Estimated Price:</b> ${row['Estimated_Price']:,}<br>
    <b>Latitude:</b> {row['latitude']}<br>
    <b>Longitude:</b> {row['longitude']}
    """
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color="blue",
        fill=True,
        fill_color="lightblue",
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(map_pred)

st_folium(map_pred, width=1200, height=700)
