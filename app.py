import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PipelineWatch-NG",
    page_icon="satellite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR = "data/cached"
MODEL_DIR = "data/models"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_risk_scored():
    path = os.path.join(CACHE_DIR, "m3_risk_scored.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_fire_hotspots():
    path = os.path.join(CACHE_DIR, "fire_hotspots.geojson")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_clusters():
    path = os.path.join(CACHE_DIR, "m2_refinery_clusters.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_sar_points():
    path = os.path.join(CACHE_DIR, "sar_dark_spots.geojson")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_model_config():
    path = os.path.join(CACHE_DIR, "m3_model_config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_m1_metadata():
    path = os.path.join(CACHE_DIR, "m1_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

df_risk    = load_risk_scored()
firms_gj   = load_fire_hotspots()
df_clusters = load_clusters()
sar_gj     = load_sar_points()
model_cfg  = load_model_config()
m1_meta    = load_m1_metadata()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Crude_Oil_Tanker.jpg/320px-Crude_Oil_Tanker.jpg",
                 use_column_width=True)
st.sidebar.title("PipelineWatch-NG")
st.sidebar.caption("Satellite-based crude oil theft monitoring")
st.sidebar.markdown("---")

st.sidebar.markdown("**Study area**")
st.sidebar.markdown("Trans Niger Pipeline (TNP) corridor")
st.sidebar.markdown("5.0-5.8 N, 6.5-7.2 E")
st.sidebar.markdown("---")

st.sidebar.markdown("**Analysis period**")
st.sidebar.markdown("Baseline: Jan-Jun 2023")
st.sidebar.markdown("Recent:   Jan-Jun 2024")
st.sidebar.markdown("---")

st.sidebar.markdown("**Sensor stack**")
st.sidebar.markdown("- Sentinel-1 SAR (cloud-free, 24/7)")
st.sidebar.markdown("- FIRMS/VIIRS thermal IR (night-capable)")
st.sidebar.markdown("- TROPOMI SO2 (chemical fingerprint)")
st.sidebar.markdown("- Sentinel-2 NDVI/NDWI")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by Emmanuel Ibekwe")
st.sidebar.markdown("[GitHub](https://github.com/Ibekwemmanuel7/pipelinewatch-ng)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("PipelineWatch-NG")
st.markdown("#### Satellite-based crude oil theft and pipeline monitoring — Niger Delta, Nigeria")
st.markdown("---")

# ── KPI metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

n_high   = int((df_risk["risk_tier"] == "HIGH").sum())   if not df_risk.empty else 0
n_medium = int((df_risk["risk_tier"] == "MEDIUM").sum()) if not df_risk.empty else 0
n_fire   = len(firms_gj["features"]) if firms_gj else 0
n_clust  = len(df_clusters)
cv_acc   = model_cfg.get("cv_accuracy", 0)

col1.metric("HIGH risk zones",    str(n_high),   delta="Active alerts")
col2.metric("MEDIUM risk zones",  str(n_medium), delta="Monitor")
col3.metric("Fire hotspots",      str(n_fire),   delta="VIIRS detected")
col4.metric("Refinery clusters",  str(n_clust),  delta="DBSCAN sites")
col5.metric("Model CV accuracy",  str(round(cv_acc * 100, 1)) + "%", delta="XGBoost")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk Map",
    "Fire Clusters",
    "Feature Importance",
    "Alert Table",
    "About"
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — RISK MAP
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Combined Risk Score Map")
    st.caption("XGBoost + Isolation Forest combined risk index over the TNP corridor")

    layer_choice = st.selectbox(
        "Display layer",
        ["Risk tiers (HIGH / MEDIUM / LOW)",
         "Continuous risk score",
         "SAR dark spots",
         "Fire hotspots"]
    )

    m = folium.Map(
        location=[5.40, 6.85],
        zoom_start=9,
        tiles="CartoDB dark_matter"
    )

    # ROI boundary
    folium.Rectangle(
        bounds=[[5.00, 6.50], [5.80, 7.20]],
        color="#378ADD",
        fill=False,
        weight=2,
        dash_array="8 4",
        tooltip="TNP corridor study area"
    ).add_to(m)

    tier_colors = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#B5D4F4"}

    if not df_risk.empty:
        if layer_choice == "Risk tiers (HIGH / MEDIUM / LOW)":
            for _, row in df_risk.iterrows():
                color  = tier_colors.get(row["risk_tier"], "#888780")
                radius = 8 if row["risk_tier"] == "HIGH" else 5 if row["risk_tier"] == "MEDIUM" else 3
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=("Risk: " + str(row["risk_tier"]) +
                             " | Score: " + str(round(row["combined_risk_score"], 3)) +
                             " | VV: " + str(round(row["VV"], 2)) + " dB")
                ).add_to(m)

        elif layer_choice == "Continuous risk score":
            if "combined_risk_score" in df_risk.columns:
                max_score = df_risk["combined_risk_score"].max()
                for _, row in df_risk.iterrows():
                    intensity = row["combined_risk_score"] / max_score if max_score > 0 else 0
                    r = int(255 * intensity)
                    b = int(255 * (1 - intensity))
                    color = "#{:02x}00{:02x}".format(r, b)
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=4,
                        color=color,
                        fill=True,
                        fill_opacity=0.75,
                        tooltip="Score: " + str(round(row["combined_risk_score"], 3))
                    ).add_to(m)

        elif layer_choice == "SAR dark spots" and sar_gj:
            for feat in sar_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                props  = feat["properties"]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=6,
                    color="#E24B4A",
                    fill=True,
                    fill_opacity=0.8,
                    tooltip="VV: " + str(round(props.get("VV", 0), 2)) + " dB"
                ).add_to(m)

        elif layer_choice == "Fire hotspots" and firms_gj:
            for feat in firms_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                props  = feat["properties"]
                t21    = props.get("T21_max_K", 330)
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=5,
                    color="#EF9F27",
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=("T21: " + str(round(t21, 1)) + "K" +
                             " | Source: " + str(props.get("likely_source", "?")))
                ).add_to(m)

    st_folium(m, height=520, use_container_width=True)

    # Legend
    leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
    leg_col1.markdown(":red_circle: HIGH risk")
    leg_col2.markdown(":orange_circle: MEDIUM risk")
    leg_col3.markdown(":blue_circle: LOW risk")
    leg_col4.markdown(":large_blue_square: TNP corridor boundary")

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — FIRE CLUSTERS
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("DBSCAN Fire Hotspot Clusters")
    st.caption("50 VIIRS fire hotspots grouped into 8 candidate refinery/flare sites")

    if not df_clusters.empty:
        # Cluster map
        m2 = folium.Map(
            location=[5.40, 6.85],
            zoom_start=9,
            tiles="CartoDB positron"
        )

        folium.Rectangle(
            bounds=[[5.00, 6.50], [5.80, 7.20]],
            color="#185FA5",
            fill=False,
            weight=2,
            dash_array="8 4"
        ).add_to(m2)

        risk_colors = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#1D9E75"}

        for _, row in df_clusters.iterrows():
            color = risk_colors.get(row["risk_label"], "#888780")
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=max(6, int(row["n_hotspots"]) * 2),
                color=color,
                fill=True,
                fill_opacity=0.75,
                tooltip=("Cluster " + str(int(row["cluster_id"])) +
                         " | " + str(int(row["n_hotspots"])) + " hotspots" +
                         " | T21_max: " + str(round(row["max_T21_K"], 1)) + "K" +
                         " | Risk: " + str(row["risk_label"]) +
                         " | Source: " + str(row["dominant_source"]))
            ).add_to(m2)
            folium.Marker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                icon=folium.DivIcon(
                    html="<div style=font-size:10px;font-weight:bold;color:#333>C" +
                         str(int(row["cluster_id"])) + "</div>",
                    icon_size=(20, 12)
                )
            ).add_to(m2)

        # Individual hotspots
        if firms_gj:
            for feat in firms_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=3,
                    color="#FAC775",
                    fill=True,
                    fill_opacity=0.5,
                    tooltip="Individual hotspot"
                ).add_to(m2)

        st_folium(m2, height=480, use_container_width=True)

        # Cluster table
        st.markdown("**Cluster summary table**")
        display_cols = ["cluster_id","n_hotspots","centroid_lat",
                        "centroid_lon","max_T21_K","mean_fire_count",
                        "dominant_source","risk_label"]
        available = [c for c in display_cols if c in df_clusters.columns]
        st.dataframe(
            df_clusters[available].round(3),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Cluster data not found. Run Module 2 first.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — FEATURE IMPORTANCE
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Feature Importance — Which sensor drives the risk score?")

    if model_cfg and "feature_cols" in model_cfg:
        try:
            import xgboost as xgb
            import joblib

            model_path = os.path.join(MODEL_DIR, "xgb_risk_scorer.json")
            if os.path.exists(model_path):
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                importances = model.feature_importances_
                feat_cols   = model_cfg["feature_cols"]

                feat_df = pd.DataFrame({
                    "Feature":    feat_cols,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)

                fig = px.bar(
                    feat_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#B5D4F4","#378ADD","#E24B4A"],
                    title="XGBoost Feature Importance (Gain)",
                    labels={"Importance": "Importance Score", "Feature": ""}
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis={"autorange": "reversed"},
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Interpretation**")
                top_feat = feat_df.iloc[0]["Feature"]
                st.markdown("- Top signal: **" + top_feat + "**")
                st.markdown("- SAR_change_dB: how much darker recent SAR is vs baseline — new spill indicator")
                st.markdown("- NDVI_change: vegetation loss along pipeline ROW — oil contamination indicator")
                st.markdown("- VV/VH: raw SAR backscatter — oil suppresses radar return on water")
            else:
                st.warning("Model file not found. Run Module 3 first.")
        except Exception as e:
            st.error("Could not load model: " + str(e))
    else:
        st.warning("Model config not found. Run Module 3 first.")

    # CV accuracy metric
    if model_cfg:
        st.markdown("---")
        m1c, m2c, m3c = st.columns(3)
        m1c.metric("CV Accuracy",  str(round(model_cfg.get("cv_accuracy", 0) * 100, 1)) + "%")
        m2c.metric("Training samples", str(model_cfg.get("n_samples", 0)))
        m3c.metric("Model", "XGBoost v" + str(model_cfg.get("model_version", "1.0")))

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — ALERT TABLE
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Active Alerts — Highest Risk Locations")
    st.caption("Sorted by combined risk score. Filter by tier using the selector below.")

    if not df_risk.empty:
        tier_filter = st.selectbox(
            "Filter by risk tier",
            ["All", "HIGH only", "MEDIUM and above"]
        )

        if tier_filter == "HIGH only":
            df_alerts = df_risk[df_risk["risk_tier"] == "HIGH"]
        elif tier_filter == "MEDIUM and above":
            df_alerts = df_risk[df_risk["risk_tier"].isin(["HIGH", "MEDIUM"])]
        else:
            df_alerts = df_risk

        df_alerts = df_alerts.sort_values("combined_risk_score", ascending=False)

        alert_cols = ["lat","lon","combined_risk_score","risk_tier",
                      "VV","dark_spot_mask","SAR_change_dB","NDVI_change"]
        available  = [c for c in alert_cols if c in df_alerts.columns]

        def color_tier(val):
            colors = {"HIGH": "background-color: #FCEBEB; color: #A32D2D",
                      "MEDIUM": "background-color: #FAEEDA; color: #633806",
                      "LOW": "background-color: #E6F1FB; color: #0C447C"}
            return colors.get(val, "")

        styled = (df_alerts[available]
                  .round(3)
                  .style.applymap(color_tier, subset=["risk_tier"]))

        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(str(len(df_alerts)) + " locations shown")

        # Download button
        csv = df_alerts[available].round(3).to_csv(index=False)
        st.download_button(
            label="Download alert table as CSV",
            data=csv,
            file_name="pipelinewatch_alerts_" + datetime.now().strftime("%Y%m%d") + ".csv",
            mime="text/csv"
        )
    else:
        st.warning("Risk scored data not found. Run Module 3 first.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("About PipelineWatch-NG")

    st.markdown("""
Nigeria loses an estimated **150,000+ barrels per day** to crude oil theft.
The Niger Delta terrain — mangrove creeks, remote waterways — makes ground
surveillance nearly impossible.

PipelineWatch-NG uses free space-based sensors to detect oil spills,
illegal refineries, and suspicious activity entirely from the cloud,
with no satellite data downloads.
""")

    st.markdown("**Sensor stack**")
    sensor_df = pd.DataFrame([
        {"Sensor": "Sentinel-1 SAR",  "Signal": "Oil spill dark spots",        "Cloud-free": "Yes", "Night": "Yes", "Cost": "Free"},
        {"Sensor": "FIRMS/VIIRS",     "Signal": "Illegal refinery fire hotspots","Cloud-free": "Yes", "Night": "Yes", "Cost": "Free"},
        {"Sensor": "TROPOMI SO2",     "Signal": "Chemical plumes from crude burning","Cloud-free": "Partial","Night": "No","Cost": "Free"},
        {"Sensor": "Sentinel-2 MSI",  "Signal": "Vegetation dieback along ROW",  "Cloud-free": "No",  "Night": "No",  "Cost": "Free"},
    ])
    st.dataframe(sensor_df, use_container_width=True, hide_index=True)

    st.markdown("**ML pipeline**")
    st.markdown("- XGBoost risk classifier trained on multi-sensor feature table")
    st.markdown("- Isolation Forest unsupervised anomaly detection")
    st.markdown("- Combined weighted risk index (0-1 scale)")
    st.markdown("- DBSCAN spatial clustering of fire hotspots")

    st.markdown("**Target stakeholders**")
    st.markdown("- NNPC (Nigerian National Petroleum Corporation)")
    st.markdown("- Nigerian Ministry of Petroleum Resources")
    st.markdown("- International oil companies: Shell, Chevron, TotalEnergies")
    st.markdown("- World Bank / donor organisations funding Niger Delta monitoring")

    st.markdown("**Data sources**")
    st.markdown("All data via Google Earth Engine — zero downloads, free tier.")

    if m1_meta:
        st.markdown("**Last ingestion run**")
        det = m1_meta.get("detections", {})
        sc  = m1_meta.get("scene_counts", {})
        run_df = pd.DataFrame([
            {"Sensor": "Sentinel-1 SAR",  "Scenes (recent)": sc.get("s1_recent", "-"),     "Detections": "see risk map"},
            {"Sensor": "FIRMS/VIIRS",     "Scenes (recent)": sc.get("firms_recent", "-"),   "Detections": str(det.get("fire_hotspots", "-"))},
            {"Sensor": "TROPOMI SO2",     "Scenes (recent)": sc.get("tropomi_recent", "-"), "Detections": str(det.get("so2_anomalies", "-"))},
        ])
        st.dataframe(run_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("Built by **Emmanuel Ibekwe** | [GitHub](https://github.com/Ibekwemmanuel7/pipelinewatch-ng)")
