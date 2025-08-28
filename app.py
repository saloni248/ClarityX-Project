import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="US Real Estate Dashboard", layout="wide")
st.title("🏠 US Real Estate Asset Dashboard")

# ------------------------------------------------------------
# Data loading helpers (tries .xlsx then .csv)
# ------------------------------------------------------------
@st.cache_data
def load_any(path_base: str):
    for ext in (".xlsx", ".xls", ".csv"):
        path = path_base if path_base.endswith(ext) else path_base + ext
        try:
            if path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(path)
            else:
                return pd.read_csv(path)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No file found for base path: {path_base}")

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
assets_path = "assets_with_model_predictions"
pred_path = "predictions"
clusters_path = "asset_clusters_with_macro"

try:
    df = load_any(assets_path)
except Exception as e:
    st.error(f"❌ Error reading assets file (tried .xlsx/.csv): {e}")
    st.stop()

try:
    pred_df = load_any(pred_path)
except Exception as e:
    st.error(f"❌ Error reading predictions file (tried .xlsx/.csv): {e}")
    st.stop()

clusters_df = None
try:
    clusters_df = load_any(clusters_path)
except Exception:
    # clusters are optional for Task 2
    clusters_df = None

# ------------------------------------------------------------
# Basic validation and cleaning
# ------------------------------------------------------------
required_columns = [
    "real property asset name", "zip code", "price_latest",
    "City", "State", "latitude", "longitude",
    "building status", "real property asset type", "predicted_price_model"
]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing columns in assets file: {missing_cols}")
    st.write("✅ Columns found in file:", list(df.columns))
    st.stop()

if not {"latitude", "longitude", "Estimated_Price"}.issubset(pred_df.columns):
    st.error("❌ Predictions CSV must include 'latitude', 'longitude', and 'Estimated_Price' columns.")
    st.stop()

# ensure numeric
for col in ["price_latest", "predicted_price_model"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# make sure prediction numeric column is numeric for descriptive stats
if "Estimated_Price" in pred_df.columns:
    pred_df["Estimated_Price"] = pd.to_numeric(pred_df["Estimated_Price"], errors="coerce")

# Price bins (tertiles) safely
try:
    mask = df["predicted_price_model"].notna()
    df.loc[mask, "Price Bin"] = pd.qcut(df.loc[mask, "predicted_price_model"], q=3, labels=["Low", "Medium", "High"]) 
except Exception:
    df["Price Bin"] = np.nan

# Sidebar navigation (no Original Maps page)
page = st.sidebar.selectbox("Select Page", ["Overview", "Maps", "Clusters & Diagnostics"]) 

# Shared sidebar filters (applied on pages that need them)
st.sidebar.header("🔎 Filters")
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

filtered_df = df[
    (df["building status"].isin(building_options)) &
    (df["real property asset type"].isin(asset_type_options)) &
    (df["Price Bin"].isin(price_bin_options))
]

# ---------- Page: Overview ----------
if page == "Overview":
    st.header("Overview — Key KPIs & Focused Visuals")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Assets", len(filtered_df))
    with col2:
        avg_price = filtered_df["predicted_price_model"].mean() if "predicted_price_model" in filtered_df.columns else np.nan
        st.metric("Avg Predicted Price", f"${avg_price:,.0f}" if not np.isnan(avg_price) else "N/A")
    with col3:
        total_sqft = filtered_df.get("building rentable square feet", pd.Series([0])).fillna(0).sum()
        st.metric("Total Rentable SqFt", f"{total_sqft:,.0f}")

    st.subheader("Distribution of Predicted Prices")
    if "predicted_price_model" in filtered_df.columns and filtered_df["predicted_price_model"].notna().any():
        fig_hist = px.histogram(filtered_df, x="predicted_price_model", nbins=30, title="Predicted Price Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.write("No predicted prices to plot.")

    st.subheader("Predicted Price by Asset Type")
    if "real property asset type" in filtered_df.columns:
        fig_box = px.box(filtered_df, x="real property asset type", y="predicted_price_model", title="Price by Asset Type")
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Top Cities by Asset Count")
    city_count = filtered_df["City"].value_counts().nlargest(10).reset_index()
    city_count.columns = ["City", "Count"]
    fig_bar = px.bar(city_count, x="City", y="Count", title="Top 10 Cities by Asset Count")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- NEW: Add two cluster count charts side-by-side ---
    st.markdown("---")
    st.subheader("Cluster counts — two sources")
    cA, cB = st.columns(2)

    with cA:
        st.markdown("**A: asset_clusters_with_macro.csv — direct cluster counts**")
        if clusters_df is not None:
            cluster_col = None
            for c in ("cluster_names", "cluster_name", "cluster"):
                if c in clusters_df.columns:
                    cluster_col = c
                    break
            if cluster_col:
                counts = clusters_df[cluster_col].value_counts().reset_index()
                counts.columns = ["cluster", "count"]
                fig1 = px.bar(counts, x="cluster", y="count", title="Asset counts by cluster (asset_clusters_with_macro)")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.write("No cluster column found in asset_clusters_with_macro.csv")
        else:
            st.write("asset_clusters_with_macro.csv not loaded.")

    with cB:
        st.markdown("**B: assets_with_model_predictions.csv — cluster counts (mapped)**")
        # Strategy: use cluster column in df if present; else try merge using asset_index
        df_cluster_col = None
        for c in ("cluster_names", "cluster_name", "cluster"):
            if c in df.columns:
                df_cluster_col = c
                break

        mapped_counts = None
        if df_cluster_col:
            mapped_counts = df[df_cluster_col].value_counts().reset_index()
            mapped_counts.columns = ["cluster", "count"]
        else:
            if clusters_df is not None and "asset_index" in clusters_df.columns and "asset_index" in df.columns:
                cluster_col = None
                for c in ("cluster_names", "cluster_name", "cluster"):
                    if c in clusters_df.columns:
                        cluster_col = c
                        break
                if cluster_col:
                    merged = df.merge(clusters_df[["asset_index", cluster_col]], on="asset_index", how="left")
                    if cluster_col in merged.columns:
                        mapped_counts = merged[cluster_col].value_counts().reset_index()
                        mapped_counts.columns = ["cluster", "count"]

        if mapped_counts is not None and not mapped_counts.empty:
            fig2 = px.bar(mapped_counts, x="cluster", y="count", title="Asset counts by cluster (mapped to assets_with_model_predictions)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("Could not determine cluster assignments for assets_with_model_predictions.csv — no cluster column and no asset_index to merge.")

    with st.expander("Top 10 assets by predicted_price_model (table)"):
        if "predicted_price_model" in df.columns:
            top10 = df.sort_values("predicted_price_model", ascending=False).head(10)[[
                "real property asset name", "City", "State", "zip code", "price_latest", "predicted_price_model"
            ]]
            st.dataframe(top10)
        else:
            st.write("predicted_price_model missing")

# ---------- Page: Maps ----------
# ---------- Page: Maps ----------
elif page == "Maps":
    st.header("Maps — Heatmaps & Density")
    st.markdown("**assets_with_model_predictions.csv — HeatMap (density of assets)**")
    try:
        heat_data = filtered_df[["latitude", "longitude"]].dropna().values.tolist()
        if heat_data:
            m_heat = folium.Map(location=[filtered_df["latitude"].mean(), filtered_df["longitude"].mean()], zoom_start=4)
            HeatMap(heat_data, radius=8, blur=10).add_to(m_heat)
            st_folium(m_heat, width=900, height=400, key="heat_assets_map")
        else:
            st.write("Not enough points for assets HeatMap.")
    except Exception as e:
        st.write("HeatMap generation failed:", e)

    # ----------------------------
    # NEW: Geo scatter 
    # ----------------------------
    st.markdown("---")
    col_geo, col_city = st.columns([2,1])
    with col_geo:
        st.subheader("Geo Scatter — Predicted Price (marker size by price)")
        try:
            geo_df = filtered_df.dropna(subset=['latitude','longitude','predicted_price_model']).copy()
            if not geo_df.empty:
                fig_geo = px.scatter_geo(
                    geo_df,
                    lat='latitude',
                    lon='longitude',
                    hover_name='real property asset name',
                    size='predicted_price_model',
                    title='Geographic Distribution of Predicted Prices',
                    scope='usa'
                )
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.write("Not enough geolocated points for geo scatter.")
        except Exception as e:
            st.write("Failed geo scatter:", e)

    st.markdown("---")

# ---------- Page: Clusters & Diagnostics ----------
elif page == "Clusters & Diagnostics":
    st.header("Clusters & Diagnostics — Focused analysis")

    # --- NEW: descriptive stats tables for assets and predictions ---
    st.subheader("Descriptive statistics — assets_with_model_predictions.csv")
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T.round(3))
        else:
            st.write("No numeric columns found in assets file for descriptive stats.")
    except Exception as e:
        st.write("Failed to compute descriptive stats for assets file:", e)

    st.subheader("Descriptive statistics — predictions.csv")
    try:
        numeric_cols_p = pred_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols_p:
            st.dataframe(pred_df[numeric_cols_p].describe().T.round(3))
        else:
            st.write("No numeric columns found in predictions file for descriptive stats.")
    except Exception as e:
        st.write("Failed to compute descriptive stats for predictions file:", e)

    st.markdown("---")

    st.subheader("Price Correlation")
    if {"price_latest", "predicted_price_model"}.issubset(df.columns):
        corr = df["price_latest"].corr(df["predicted_price_model"])
        st.write(f"Pearson correlation coefficient: **{corr:.3f}**")
    else:
        st.write("Required columns missing for correlation.")

    st.subheader("Largest % Deviations (top 10) — Predicted vs Latest")
    if {"predicted_price_model", "price_latest"}.issubset(df.columns):
        tmp = df.copy()
        tmp = tmp[tmp["price_latest"].notna() & (tmp["price_latest"] != 0) & tmp["predicted_price_model"].notna()].copy()
        tmp["pct_diff"] = (tmp["predicted_price_model"] - tmp["price_latest"]) / tmp["price_latest"] * 100
        top_dev = tmp.reindex(tmp["pct_diff"].abs().sort_values(ascending=False).index).head(10)[[
            "real property asset name", "City", "State", "price_latest", "predicted_price_model", "pct_diff"
        ]]
        top_dev["pct_diff"] = top_dev["pct_diff"].round(2)
        st.dataframe(top_dev)
    else:
        st.write("Required columns for deviation table not present.")

    # If cluster information is available, show a compact summary and a meaningful graph
    if clusters_df is not None and "cluster_names" in clusters_df.columns:
        st.subheader("Cluster-level Summary")
        price_col = None
        for c in ("predicted_price_model", "Estimated_price", "Estimated_Price", "Estimated Price"):
            if c in clusters_df.columns:
                price_col = c
                break
        agg = {price_col:["count","mean"]} if price_col else {}
        if "building rentable square feet" in clusters_df.columns:
            agg["building rentable square feet"] = ["mean"]
        if agg:
            cs = clusters_df.groupby("cluster_names").agg(agg)
            cs.columns = ["_" . join(col).strip() for col in cs.columns.values]
            st.dataframe(cs.reset_index())

        # Visualization: average price per cluster (if price present)
        if price_col:
            fig_cluster = px.bar(clusters_df.groupby("cluster_names")[price_col].mean().reset_index(),
                                 x="cluster_names", y=price_col, title="Avg Price by Cluster")
            st.plotly_chart(fig_cluster, use_container_width=True)

    else:
        st.write("No cluster file or 'cluster_names' column available.")

# End
st.sidebar.markdown("---")
st.sidebar.caption("Task 2: descriptive + spatial summaries. Original maps removed; cluster & predictions descriptive tables added.")
