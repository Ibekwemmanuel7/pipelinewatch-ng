"""
PipelineWatch-NG — Streamlit Dashboard
Satellite-based crude oil theft monitoring · Niger Delta, Nigeria

Live demo: https://pipelinewatch-ng.streamlit.app
GitHub:    https://github.com/Ibekwemmanuel7/pipelinewatch-ng
"""

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
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CACHE_DIR = "data/cached"
MODEL_DIR = "data/models"

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_risk_scored():
    path = os.path.join(CACHE_DIR, "m3_risk_scored.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_geojson(filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_clusters():
    path = os.path.join(CACHE_DIR, "m2_refinery_clusters.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_validation():
    path = os.path.join(CACHE_DIR, "m2b_tropomi_validation.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_model_config():
    path = os.path.join(CACHE_DIR, "m3_model_config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_nrt():
    path = os.path.join(CACHE_DIR, "m5_nrt_latest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

df_risk     = load_risk_scored()
firms_gj    = load_geojson("fire_hotspots.geojson")
sar_gj      = load_geojson("sar_dark_spots.geojson")
df_clusters = load_clusters()
df_val      = load_validation()
model_cfg   = load_model_config()
nrt         = load_nrt()
nrt_alert   = nrt.get("alert_level", "PENDING") if nrt else "PENDING"
nrt_window  = nrt.get("nrt_window", "Awaiting first scheduled update") if nrt else "Awaiting first scheduled update"
nrt_run_date = nrt.get("run_date", "") if nrt else ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("PipelineWatch-NG")
    st.caption("Near-real-time satellite crude oil theft monitoring")
    st.markdown("---")
    st.markdown("**Study area**")
    st.markdown("Trans Niger Pipeline (TNP) corridor")
    st.markdown("5.0–5.8°N, 6.5–7.2°E")
    st.markdown("---")
    st.markdown("**Monitoring mode**")
    st.markdown("Near-real-time rolling alert pipeline")
    st.markdown("Latest window: " + nrt_window)
    if nrt_run_date:
        st.markdown("Last run: " + nrt_run_date[:10])
    st.markdown("")
    st.markdown("**Model calibration**")
    st.markdown("Baseline: Jan–Jun 2023")
    st.markdown("Historical comparison: Jan–Jun 2024")
    st.markdown("---")
    st.markdown("**Sensor stack**")
    st.markdown("- Sentinel-1 SAR (cloud-free, 24/7)")
    st.markdown("- FIRMS/VIIRS (thermal IR, night-capable)")
    st.markdown("- TROPOMI SO₂ (chemical fingerprint)")
    st.markdown("- Sentinel-2 NDVI/NDWI")
    st.markdown("---")

    # NRT alert badge
    color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "PENDING": "⚪"}.get(nrt_alert, "⚪")
    st.markdown("**Latest NRT alert**")
    st.markdown(color + " " + nrt_alert)

    st.markdown("---")
    st.markdown("Built by **Emmanuel Ibekwe**")
    st.markdown("[GitHub](https://github.com/Ibekwemmanuel7/pipelinewatch-ng)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("PipelineWatch-NG")
st.markdown("#### Near-real-time satellite crude oil theft and pipeline monitoring — Niger Delta, Nigeria")
st.markdown("---")

# ── KPI metrics ───────────────────────────────────────────────────────────────
n_high      = int((df_risk["risk_tier"] == "HIGH").sum())   if not df_risk.empty else 0
n_medium    = int((df_risk["risk_tier"] == "MEDIUM").sum()) if not df_risk.empty else 0
n_fire      = len(firms_gj["features"]) if firms_gj else 0
n_confirmed = int(df_val["SO2_confirmed"].sum()) if not df_val.empty else 0
cv_acc      = model_cfg.get("cv_accuracy", 0)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("NRT alert",                 nrt_alert,        delta=nrt_window)
col2.metric("HIGH risk zones",           str(n_high),      delta="Model alerts")
col3.metric("MEDIUM risk zones",         str(n_medium),    delta="Monitor")
col4.metric("Fire hotspots (VIIRS)",     str(n_fire),      delta="Baseline detections")
col5.metric("Confirmed sites",           str(n_confirmed), delta="SO₂ + fire")
col6.metric("Model CV accuracy",         str(round(cv_acc * 100, 1)) + "%", delta="XGBoost")

if not nrt:
    st.info(
        "Near-real-time mode is enabled. The latest rolling alert will appear after "
        "the first GitHub Actions NRT run writes data/cached/m5_nrt_latest.json."
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Risk Map",
    "Fire Clusters",
    "SO₂ Validation",
    "Feature Importance",
    "Alert Table",
    "About"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RISK MAP
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Combined Risk Score Map")
    st.caption("XGBoost + Isolation Forest combined risk index over the TNP corridor")

    layer_choice = st.selectbox(
        "Display layer",
        ["Risk tiers (HIGH / MEDIUM / LOW)",
         "Continuous risk score",
         "SAR dark spots",
         "Fire hotspots (VIIRS)"]
    )

    m = folium.Map(location=[5.40, 6.85], zoom_start=9, tiles="CartoDB dark_matter")

    folium.Rectangle(
        bounds=[[5.00, 6.50], [5.80, 7.20]],
        color="#378ADD", fill=False, weight=2,
        dash_array="8 4", tooltip="TNP corridor study area"
    ).add_to(m)

    tier_colors = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#B5D4F4"}
    tier_radii  = {"HIGH": 8, "MEDIUM": 5, "LOW": 3}

    if not df_risk.empty:
        if layer_choice == "Risk tiers (HIGH / MEDIUM / LOW)":
            for _, row in df_risk.iterrows():
                color  = tier_colors.get(row["risk_tier"], "#888780")
                radius = tier_radii.get(row["risk_tier"], 3)
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=radius, color=color, fill=True, fill_opacity=0.8,
                    tooltip=("Risk: " + str(row["risk_tier"]) +
                             " | Score: " + str(round(row["combined_risk_score"], 3)) +
                             " | VV: " + str(round(row["VV"], 2)) + " dB")
                ).add_to(m)

        elif layer_choice == "Continuous risk score":
            max_score = df_risk["combined_risk_score"].max()
            for _, row in df_risk.iterrows():
                intensity = row["combined_risk_score"] / max_score if max_score > 0 else 0
                r = int(255 * intensity)
                b = int(255 * (1 - intensity))
                color = "#{:02x}00{:02x}".format(r, b)
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=4, color=color, fill=True, fill_opacity=0.75,
                    tooltip="Score: " + str(round(row["combined_risk_score"], 3))
                ).add_to(m)

        elif layer_choice == "SAR dark spots" and sar_gj:
            for feat in sar_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                props  = feat["properties"]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=6, color="#E24B4A", fill=True, fill_opacity=0.8,
                    tooltip="VV: " + str(round(props.get("VV", 0), 2)) + " dB"
                ).add_to(m)

        elif layer_choice == "Fire hotspots (VIIRS)" and firms_gj:
            for feat in firms_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                props  = feat["properties"]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=5, color="#EF9F27", fill=True, fill_opacity=0.8,
                    tooltip=("T21: " + str(round(props.get("T21_max_K", 0), 1)) + "K" +
                             " | Source: " + str(props.get("likely_source", "?")))
                ).add_to(m)

    st_folium(m, height=520, use_container_width=True)

    leg1, leg2, leg3 = st.columns(3)
    leg1.markdown("🔴 HIGH risk zone")
    leg2.markdown("🟠 MEDIUM risk zone")
    leg3.markdown("🔵 LOW risk / background")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FIRE CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("DBSCAN Fire Hotspot Clusters")
    st.caption("50 VIIRS fire hotspots grouped into 8 candidate illegal refinery / flare sites")

    if not df_clusters.empty:
        m2 = folium.Map(location=[5.40, 6.85], zoom_start=9, tiles="CartoDB positron")
        folium.Rectangle(
            bounds=[[5.00, 6.50], [5.80, 7.20]],
            color="#185FA5", fill=False, weight=2, dash_array="8 4"
        ).add_to(m2)

        risk_colors = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#1D9E75"}

        for _, row in df_clusters.iterrows():
            color = risk_colors.get(row["risk_label"], "#888780")
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=max(6, int(row["n_hotspots"]) * 2),
                color=color, fill=True, fill_opacity=0.75,
                tooltip=("Cluster " + str(int(row["cluster_id"])) +
                         " | " + str(int(row["n_hotspots"])) + " hotspots" +
                         " | T21_max: " + str(round(row["max_T21_K"], 1)) + "K" +
                         " | " + str(row["risk_label"]) +
                         " | " + str(row["dominant_source"]))
            ).add_to(m2)
            folium.Marker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                icon=folium.DivIcon(
                    html="<div style='font-size:10px;font-weight:bold;color:#333'>C" +
                         str(int(row["cluster_id"])) + "</div>",
                    icon_size=(20, 12)
                )
            ).add_to(m2)

        if firms_gj:
            for feat in firms_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=3, color="#FAC775", fill=True, fill_opacity=0.5
                ).add_to(m2)

        st_folium(m2, height=480, use_container_width=True)

        st.markdown("**Cluster summary**")
        display_cols = [c for c in ["cluster_id","n_hotspots","centroid_lat","centroid_lon",
                                     "max_T21_K","mean_fire_count","dominant_source","risk_label"]
                        if c in df_clusters.columns]
        st.dataframe(df_clusters[display_cols].round(3), use_container_width=True, hide_index=True)
    else:
        st.warning("Cluster data not found. Run Module 2.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SO2 VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("TROPOMI Dry Season Validation")
    st.caption("Oct–Dec 2023 | SO₂ co-location with fire clusters")

    st.markdown("""
**Key finding:** 6 out of 8 fire clusters show episodic SO₂ elevation > 3 Dobson Units
during the dry season — the chemical fingerprint of illegal crude oil burning.

**Why episodic?** Artisanal refineries burn intermittently (batch processing).
TROPOMI captures background on most days but spikes of 3–5+ DU on active burn days.
This low-mean / high-max pattern is the correct signature for intermittent sources.
""")

    if not df_val.empty:
        n_conf = int(df_val["SO2_confirmed"].sum())
        st.metric("Confirmed refinery candidates", str(n_conf) + " / " + str(len(df_val)),
                  delta="Fire + SO₂ co-located")

        # Validation bar chart
        fig = go.Figure()
        bar_colors = ["#E24B4A" if v else "#B5D4F4" for v in df_val["SO2_confirmed"]]
        fig.add_trace(go.Bar(
            x=["C" + str(int(i)) for i in df_val["cluster_id"]],
            y=df_val["SO2_mean_DU"],
            marker_color=bar_colors, opacity=0.85, name="SO₂ mean",
            text=[str(round(v, 3)) for v in df_val["SO2_mean_DU"]],
            textposition="outside"
        ))
        fig.add_trace(go.Bar(
            x=["C" + str(int(i)) for i in df_val["cluster_id"]],
            y=df_val["SO2_max_DU"],
            marker_color="#FAC775", opacity=0.5, name="SO₂ max (episodic)"
        ))
        fig.add_hline(y=1.5, line_dash="dash", line_color="#854F0B",
                      annotation_text="Mean threshold (1.5 DU)")
        fig.add_hline(y=3.0, line_dash="dot", line_color="#E24B4A",
                      annotation_text="Episodic threshold (3.0 DU)")
        fig.update_layout(
            title="SO₂ per fire cluster (Oct–Dec 2023 dry season)",
            yaxis_title="SO₂ (Dobson Units)", height=400, barmode="group",
            plot_bgcolor="white", paper_bgcolor="white"
        )
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Validation table**")
        display_cols = [c for c in ["cluster_id","SO2_mean_DU","SO2_max_DU",
                                     "n_hotspots","max_T21_K","SO2_confirmed","verdict"]
                        if c in df_val.columns]

        def highlight_confirmed(val):
            if val is True or val == "True":
                return "background-color: #FCEBEB; color: #A32D2D"
            return ""

        st.dataframe(
            df_val[display_cols].round(3).style.map(
                highlight_confirmed, subset=["SO2_confirmed"]
            ),
            use_container_width=True, hide_index=True
        )

        if n_conf > 0:
            st.markdown("**Recommended NNPC field inspection coordinates:**")
            for _, row in df_val[df_val["SO2_confirmed"]].iterrows():
                st.markdown("- **Cluster " + str(int(row["cluster_id"])) + ":** " +
                            str(round(row["centroid_lat"], 4)) + "°N, " +
                            str(round(row["centroid_lon"], 4)) + "°E  —  " +
                            str(row["verdict"]))
    else:
        st.warning("Validation data not found. Run Module 2b.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Feature Importance — Which sensor drives the risk score?")

    if model_cfg and "feature_cols" in model_cfg:
        try:
            import xgboost as xgb_lib
            import joblib
            model_path = os.path.join(MODEL_DIR, "xgb_risk_scorer.json")
            if os.path.exists(model_path):
                model = xgb_lib.XGBClassifier()
                model.load_model(model_path)
                importances = model.feature_importances_
                feat_cols   = model_cfg["feature_cols"]
                feat_df = pd.DataFrame({
                    "Feature": feat_cols, "Importance": importances
                }).sort_values("Importance", ascending=True)

                fig = px.bar(
                    feat_df, x="Importance", y="Feature", orientation="h",
                    color="Importance",
                    color_continuous_scale=["#B5D4F4","#378ADD","#E24B4A"],
                    title="XGBoost Feature Importance (Gain)"
                )
                fig.update_layout(
                    height=400, showlegend=False,
                    yaxis={"autorange": "reversed"},
                    coloraxis_showscale=False,
                    plot_bgcolor="white", paper_bgcolor="white"
                )
                fig.update_xaxes(gridcolor="#f0f0f0")
                st.plotly_chart(fig, use_container_width=True)

                top_feat = feat_df.iloc[-1]["Feature"]
                st.markdown("**Interpretation**")
                st.markdown("- **Top signal: " + top_feat + "** — change in SAR backscatter vs baseline")
                st.markdown("- **SAR_change_dB:** New darkening in recent period = new oil on water")
                st.markdown("- **NDVI_change:** Vegetation loss along pipeline ROW = oil contamination")
                st.markdown("- **VV / VH:** Raw SAR backscatter — oil suppresses radar return on water")

                m1c, m2c, m3c = st.columns(3)
                m1c.metric("CV Accuracy",       str(round(model_cfg.get("cv_accuracy", 0) * 100, 1)) + "%")
                m2c.metric("Training samples",  str(model_cfg.get("n_samples", 0)))
                m3c.metric("Model version",      model_cfg.get("model_version", "1.0"))
        except Exception as e:
            st.error("Could not load model: " + str(e))
    else:
        st.warning("Model config not found. Run Module 3.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ALERT TABLE
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Active Alerts — Highest Risk Locations")

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
        alert_cols = [c for c in ["lat","lon","combined_risk_score","risk_tier",
                                   "VV","dark_spot_mask","SAR_change_dB","NDVI_change"]
                      if c in df_alerts.columns]

        def color_tier(val):
            colors = {"HIGH":   "background-color: #FCEBEB; color: #A32D2D",
                      "MEDIUM": "background-color: #FAEEDA; color: #633806",
                      "LOW":    "background-color: #E6F1FB; color: #0C447C"}
            return colors.get(val, "")

        st.dataframe(
            df_alerts[alert_cols].round(3).style.map(color_tier, subset=["risk_tier"]),
            use_container_width=True, hide_index=True
        )
        st.caption(str(len(df_alerts)) + " locations shown")

        csv = df_alerts[alert_cols].round(3).to_csv(index=False)
        st.download_button(
            label="Download alert table as CSV",
            data=csv,
            file_name="pipelinewatch_alerts_" + datetime.now().strftime("%Y%m%d") + ".csv",
            mime="text/csv"
        )
    else:
        st.warning("Risk scored data not found. Run Module 3.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("About PipelineWatch-NG")
    st.markdown("""
Nigeria loses an estimated **150,000+ barrels per day** to crude oil theft.
The Niger Delta's mangrove creek terrain makes ground surveillance nearly impossible.
Existing commercial monitoring systems cost **$50,000–$500,000 per year**,
placing them beyond the reach of Nigerian government agencies.

PipelineWatch-NG is the first open-source, multi-sensor, cloud-native pipeline
for automated crude oil theft detection in the Niger Delta — deployable at **zero cost**.
""")

    st.markdown("**Sensor stack**")
    sensor_df = pd.DataFrame([
        {"Sensor": "Sentinel-1 SAR",  "Signal": "Oil spill dark spots",         "Cloud-free": "Yes", "Night": "Yes",  "Cost": "Free"},
        {"Sensor": "FIRMS/VIIRS",     "Signal": "Illegal refinery fire hotspots","Cloud-free": "Yes", "Night": "Yes",  "Cost": "Free"},
        {"Sensor": "TROPOMI SO₂",     "Signal": "Chemical plumes (crude burning)","Cloud-free": "Partial","Night": "No","Cost": "Free"},
        {"Sensor": "Sentinel-2 MSI",  "Signal": "Vegetation dieback along ROW",  "Cloud-free": "No",  "Night": "No",   "Cost": "Free"},
    ])
    st.dataframe(sensor_df, use_container_width=True, hide_index=True)

    st.markdown("**ML pipeline**")
    st.markdown("- XGBoost classifier — 98.1% cross-validated accuracy")
    st.markdown("- Isolation Forest — unsupervised anomaly detection (10% contamination)")
    st.markdown("- DBSCAN spatial clustering — haversine metric, 5.5 km search radius")
    st.markdown("- Combined risk index — 0.6 × XGBoost + 0.4 × Isolation Forest")

    st.markdown("**Key results**")
    st.markdown("- 50 persistent fire hotspots detected over the TNP corridor")
    st.markdown("- 8 DBSCAN cluster sites identified")
    st.markdown("- **6/8 clusters chemically confirmed** by TROPOMI SO₂ dry season validation")
    st.markdown("- 11,685 chronic SAR spill pixels mapped")
    st.markdown("- 2 HIGH-risk zones at 5.637°N/6.625°E and 5.727°N/6.625°E")

    st.markdown("**Target stakeholders**")
    st.markdown("- NNPC — Nigerian National Petroleum Corporation")
    st.markdown("- Nigerian Ministry of Petroleum Resources")
    st.markdown("- International oil companies: Shell, Chevron, TotalEnergies")
    st.markdown("- World Bank / donor organisations")

    st.markdown("---")
    st.markdown("Built by **Emmanuel Ibekwe** | [GitHub](https://github.com/Ibekwemmanuel7/pipelinewatch-ng) | ibekwemmanuel@gmail.com")
