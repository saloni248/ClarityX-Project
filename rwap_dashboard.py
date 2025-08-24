import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="RWAP Dashboard", layout="wide")

# ----------------- Helper Functions -----------------
def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # --- Ownership ---
    if "ownership" in df.columns:
        df["ownership_clean"] = df["ownership"].astype(str).str.strip().str.lower().map({
            "owned": "Owned", "own": "Owned", "o": "Owned",
            "lease": "Leased", "leased": "Leased", "l": "Leased"
        }).fillna(df["ownership"].astype(str).str.title())
    else:
        df["ownership_clean"] = np.nan

    # --- Asset Age ---
    if "construction_date" in df.columns:
        df["construction_date_parsed"] = pd.to_datetime(df["construction_date"], errors="coerce")
        df["asset_age_years"] = ((pd.Timestamp(date.today()) - df["construction_date_parsed"]).dt.days) / 365.25
    else:
        df["asset_age_years"] = np.nan

    # --- Rental SF bins ---
    if "rental_sqft" in df.columns:
        df["rental_sqft"] = pd.to_numeric(df["rental_sqft"], errors="coerce")
        bins = [-np.inf, 1000, 5000, 20000, 50000, 100000, np.inf]
        labels = ["<1k", "1k-5k", "5k-20k", "20k-50k", "50k-100k", "100k+"]
        df["rental_sqft_bin"] = pd.cut(df["rental_sqft"], bins=bins, labels=labels)
    else:
        df["rental_sqft_bin"] = np.nan

    # --- Building Status ---
    if "building_status" in df.columns:
        bs = df["building_status"].astype(str).str.strip().str.lower()
        map_status = {
            "active": "Active", "excess": "Excess",
            "decommissioned": "Decommissioned", "retired": "Decommissioned"
        }
        df["building_status_clean"] = bs.map(map_status)
        df.loc[df["building_status_clean"].isna(), "building_status_clean"] = df["building_status"].astype(str).str.title()
    else:
        df["building_status_clean"] = np.nan

    # --- Asset Type ---
    if "real_property_asset_type" in df.columns:
        at = df["real_property_asset_type"].astype(str).str.lower()
        df["asset_type_group"] = np.select(
            [at.str.contains("build|bldg"),
             at.str.contains("land|plot"),
             at.str.contains("structure|infra|tower|bridge")],
            ["Building", "Land", "Structure"],
            default=df["real_property_asset_type"].astype(str).str.title()
        )
    else:
        df["asset_type_group"] = np.nan

    return df

def show_dashboard(df, title="Dataset"):
    st.header(f"📊 Analytics for {title}")

    # 1. Ownership
    st.subheader("1. Ownership")
    if df["ownership_clean"].notna().any():
        st.bar_chart(df["ownership_clean"].value_counts())

    # 2. State & City
    st.subheader("2. State & City Breakdown")
    if {"state","city"}.issubset(df.columns):
        st.write("Cities per State")
        st.dataframe(df.groupby("state")["city"].nunique().rename("Unique Cities"))
        st.write("Assets by City")
        st.dataframe(df.groupby(["state","city"]).size().rename("Assets").reset_index())
    
    # 3. Asset Age
    st.subheader("3. Asset Age (years)")
    if df["asset_age_years"].notna().any():
        st.write(df["asset_age_years"].describe())
        fig, ax = plt.subplots()
        ax.hist(df["asset_age_years"].dropna(), bins=20)
        ax.set_xlabel("Years")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # 4. Rental SF
    st.subheader("4. Rental Square Feet Ranges")
    if df["rental_sqft_bin"].notna().any():
        st.bar_chart(df["rental_sqft_bin"].value_counts().sort_index())

    # 5. Status & Region
    st.subheader("5. Building Status & Regions")
    if {"region","building_status_clean"}.issubset(df.columns):
        pivot = df.pivot_table(index="region", columns="building_status_clean", values="asset_age_years", aggfunc="count", fill_value=0)
        st.dataframe(pivot)

    # 6. Asset Type
    st.subheader("6. Asset Type by Region")
    if {"region","asset_type_group"}.issubset(df.columns):
        type_pivot = df.pivot_table(index="region", columns="asset_type_group", values="asset_age_years", aggfunc="count", fill_value=0)
        st.dataframe(type_pivot)

    # 7. GIS
    st.subheader("7. GIS & Spatial Analysis")
    if {"latitude","longitude"}.issubset(df.columns):
        st.map(df.rename(columns={"latitude":"lat","longitude":"lon"})[["lat","lon"]].dropna().head(1000))

# ----------------- Main App -----------------
st.title("🏢 RWAP Analytical Dashboard")

# Load datasets with the exact names you have
df1 = preprocess(load_data("rwap25_gis_dataset1.csv"))
df2 = preprocess(load_data("rwap25_gis_dataset2.csv"))

# Sidebar choice
choice = st.sidebar.radio("Select Dataset to Explore", ["Dataset 1", "Dataset 2"])
if choice == "Dataset 1":
    show_dashboard(df1, "Dataset 1 (rwap25_gis_dataset1.csv)")
else:
    show_dashboard(df2, "Dataset 2 (rwap25_gis_dataset2.csv)")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.cluster import DBSCAN

# ---------------- Page Config ----------------
st.set_page_config(page_title="RWAP Dashboard", layout="wide")
st.title("🏢 RWAP Analytical Dashboard — rwap25_gis_dataset1")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("rwap25_gis_dataset1.csv", low_memory=False)

df = load_data()
st.write("### Preview of Dataset")
st.dataframe(df.head())

# ---------------- Data Cleaning ----------------
# Ownership
if "ownership" in df.columns:
    df["ownership_clean"] = df["ownership"].astype(str).str.lower().map({
        "owned": "Owned", "own": "Owned", "o": "Owned",
        "leased": "Leased", "lease": "Leased", "l": "Leased"
    }).fillna(df["ownership"])
else:
    df["ownership_clean"] = np.nan

# Construction Date → Age
if "construction_date" in df.columns:
    df["construction_date"] = pd.to_datetime(df["construction_date"], errors="coerce")
    df["asset_age_years"] = (pd.Timestamp(date.today()) - df["construction_date"]).dt.days / 365.25
else:
    df["asset_age_years"] = np.nan

# Rental Square Feet
if "rental_sqft" in df.columns:
    df["rental_sqft"] = pd.to_numeric(df["rental_sqft"], errors="coerce")
    bins = [-np.inf, 1000, 5000, 20000, 50000, 100000, np.inf]
    labels = ["<1k", "1k-5k", "5k-20k", "20k-50k", "50k-100k", "100k+"]
    df["rental_sqft_bin"] = pd.cut(df["rental_sqft"], bins=bins, labels=labels)
else:
    df["rental_sqft_bin"] = np.nan

# Building Status
if "building_status" in df.columns:
    df["building_status_clean"] = df["building_status"].astype(str).str.lower().map({
        "active": "Active",
        "excess": "Excess",
        "decommissioned": "Decommissioned",
        "retired": "Decommissioned"
    }).fillna(df["building_status"])
else:
    df["building_status_clean"] = np.nan

# Asset Type
if "real_property_asset_type" in df.columns:
    at = df["real_property_asset_type"].astype(str).str.lower()
    df["asset_type_group"] = np.select(
        [at.str.contains("build|bldg", na=False),
         at.str.contains("land|plot", na=False),
         at.str.contains("structure|infra|tower|bridge", na=False)],
        ["Building", "Land", "Structure"],
        default=df["real_property_asset_type"]
    )
else:
    df["asset_type_group"] = np.nan

# ---------------- Dashboard Sections ----------------
st.header("1. Owned vs Leased")
st.bar_chart(df["ownership_clean"].value_counts())

st.header("2. Cities per State & Assets per City")
if {"state", "city"}.issubset(df.columns):
    st.write("Unique Cities per State")
    st.dataframe(df.groupby("state")["city"].nunique().rename("Unique Cities"))
    st.write("Assets by City")
    st.dataframe(df.groupby(["state", "city"]).size().rename("Assets").reset_index())

st.header("3. Asset Age (Years)")
if df["asset_age_years"].notna().any():
    st.write(df["asset_age_years"].describe())
    fig, ax = plt.subplots()
    ax.hist(df["asset_age_years"].dropna(), bins=20, color="skyblue")
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

st.header("4. Rental Square Feet Ranges")
st.bar_chart(df["rental_sqft_bin"].value_counts().sort_index())

st.header("5. Building Status by Region")
if {"region", "building_status_clean"}.issubset(df.columns):
    pivot = df.pivot_table(index="region", columns="building_status_clean",
                           values="ownership_clean", aggfunc="count", fill_value=0)
    st.dataframe(pivot)
    if "Active" in pivot.columns:
        st.write("✅ Top Active Regions")
        st.dataframe(pivot.sort_values("Active", ascending=False).head(10))
    if "Decommissioned" in pivot.columns:
        st.write("⚠️ Top Decommissioned Regions")
        st.dataframe(pivot.sort_values("Decommissioned", ascending=False).head(10))

st.header("6. Asset Type by Region")
if {"region", "asset_type_group"}.issubset(df.columns):
    type_pivot = df.pivot_table(index="region", columns="asset_type_group",
                                values="ownership_clean", aggfunc="count", fill_value=0)
    st.dataframe(type_pivot)
    st.bar_chart(df["asset_type_group"].value_counts())

st.header("7. GIS Map")
if {"latitude", "longitude"}.issubset(df.columns):
    st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]].dropna().head(1000))
