"""
PipelineWatch-NG - Streamlit Dashboard
Satellite-based crude oil theft monitoring, Niger Delta, Nigeria

Live demo: https://pipelinewatch-ng.streamlit.app
GitHub:    https://github.com/Ibekwemmanuel7/pipelinewatch-ng
"""

from datetime import datetime, timezone
import json
import os

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(
    page_title="PipelineWatch-NG",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CACHE_DIR = "data/cached"
MODEL_DIR = "data/models"


@st.cache_data
def load_risk_scored():
    path = os.path.join(CACHE_DIR, "m3_risk_scored.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_geojson(filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_nrt():
    path = os.path.join(CACHE_DIR, "m5_nrt_latest.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


df_risk = load_risk_scored()
firms_gj = load_geojson("fire_hotspots.geojson")
sar_gj = load_geojson("sar_dark_spots.geojson")
df_clusters = load_clusters()
df_val = load_validation()
model_cfg = load_model_config()
nrt = load_nrt()
nrt_alert = nrt.get("alert_level", "PENDING") if nrt else "PENDING"
nrt_window = nrt.get("nrt_window", "Awaiting first scheduled update") if nrt else "Awaiting first scheduled update"
nrt_run_date = nrt.get("run_date", "") if nrt else ""


with st.sidebar:
    st.title("PipelineWatch-NG")
    st.caption("Near-real-time satellite crude oil theft monitoring")
    st.markdown("---")
    st.markdown("**Study area**")
    st.markdown("Trans Niger Pipeline (TNP) corridor")
    st.markdown("5.0-5.8 N, 6.5-7.2 E")
    st.markdown("---")
    st.markdown("**Monitoring mode**")
    st.markdown("Near-real-time rolling alert pipeline")
    st.markdown(f"Latest window: {nrt_window}")
    if nrt_run_date:
        st.markdown(f"Last run: {nrt_run_date[:10]}")
    st.markdown("")
    st.markdown("**Model calibration**")
    st.markdown("Baseline: Jan-Jun 2023")
    st.markdown("Historical comparison: Jan-Jun 2024")
    st.markdown("---")
    st.markdown("**Sensor stack**")
    st.markdown("- Sentinel-1 SAR (cloud-free, 24/7)")
    st.markdown("- FIRMS/VIIRS (thermal IR, night-capable)")
    st.markdown("- TROPOMI SO2 (chemical fingerprint)")
    st.markdown("- Sentinel-2 NDVI/NDWI")

    label = {"HIGH": "RED", "MEDIUM": "AMBER", "LOW": "GREEN", "PENDING": "PENDING"}.get(
        nrt_alert, "UNKNOWN"
    )
    st.markdown("---")
    st.markdown("**Latest NRT alert**")
    st.markdown(f"{label}: {nrt_alert}")

    st.markdown("---")
    st.markdown("Built by **Emmanuel Ibekwe**")
    st.markdown("[GitHub](https://github.com/Ibekwemmanuel7/pipelinewatch-ng)")


st.title("PipelineWatch-NG")
st.markdown("#### Near-real-time satellite crude oil theft and pipeline monitoring - Niger Delta, Nigeria")
st.markdown("---")

n_high = int((df_risk["risk_tier"] == "HIGH").sum()) if not df_risk.empty else 0
n_medium = int((df_risk["risk_tier"] == "MEDIUM").sum()) if not df_risk.empty else 0
n_fire = len(firms_gj["features"]) if firms_gj else 0
n_confirmed = int(df_val["SO2_confirmed"].sum()) if not df_val.empty and "SO2_confirmed" in df_val else 0
cv_acc = model_cfg.get("cv_accuracy", 0)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("NRT alert", nrt_alert, delta=nrt_window)
col2.metric("HIGH risk zones", str(n_high), delta="Model alerts")
col3.metric("MEDIUM risk zones", str(n_medium), delta="Monitor")
col4.metric(
    "Thermal hotspots (VIIRS)",
    str(n_fire),
    delta="Baseline detections",
    help=(
        "VIIRS thermal radiance hotspots from NASA FIRMS. The sensor measures "
        "mid-IR brightness temperature; persistent thermal anomalies are "
        "candidate combustion sources (illegal refinery flares, gas flares, "
        "vegetation burning, or other industrial heat). Not all hotspots are "
        "fire - see About tab for the FIRMS disclaimer."
    ),
)
col5.metric(
    "Co-located sites",
    str(n_confirmed),
    delta="Thermal + SO₂",
    help=(
        "Thermal hotspot clusters whose locations also show episodic SO₂ "
        "elevation in TROPOMI dry-season retrievals - the two-signature "
        "pattern consistent with crude burning."
    ),
)
col6.metric(
    "Model CV accuracy",
    f"{round(cv_acc * 100, 1)}%",
    delta="XGBoost",
    help=(
        "Cross-validated against weak proxy labels derived from domain rules, "
        "not ground-truth theft incidents. Indicates internal model "
        "self-consistency rather than confirmed real-world detection rate. "
        "Supervised retraining will require collaboration with NNPC field teams."
    ),
)

if not nrt:
    st.info(
        "Near-real-time mode is enabled. The latest rolling alert will appear after "
        "the first GitHub Actions NRT run writes data/cached/m5_nrt_latest.json. "
        "The pipeline runs every 6 hours once the GEE service-account secrets are "
        "configured in the GitHub repository."
    )
else:
    # Per-sensor freshness panel: auditable data age for ops/procurement review.
    try:
        last_run_dt = datetime.fromisoformat(nrt["run_date"])
        if last_run_dt.tzinfo is None:
            last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)
        secs = (datetime.now(timezone.utc) - last_run_dt).total_seconds()
        if secs < 3600:
            age_str = f"{int(secs / 60)} min ago"
        elif secs < 86400:
            age_str = f"{round(secs / 3600, 1)} h ago"
        else:
            age_str = f"{round(secs / 86400, 1)} d ago"
        is_stale = secs > 6 * 3600 * 1.5  # 1.5x the cron interval
    except Exception:
        age_str = "unknown"
        is_stale = False

    f1, f2, f3, f4 = st.columns(4)
    age_color = "#E24B4A" if is_stale else "#1D9E75"
    f1.markdown(
        f"**Pipeline last run**  \n"
        f"<span style='color:{age_color};font-weight:600'>{age_str}</span>  \n"
        f"<span style='color:#666;font-size:0.85em'>Cron: every 6 h</span>",
        unsafe_allow_html=True,
    )
    f2.markdown(
        f"**Window**  \n{nrt.get('nrt_window', '-')}  \n"
        f"<span style='color:#666;font-size:0.85em'>Rolling 7-day analysis</span>",
        unsafe_allow_html=True,
    )
    f3.markdown(
        f"**VIIRS thermal scenes**  \n{nrt.get('firms_images', 0)} in window  \n"
        f"<span style='color:#666;font-size:0.85em'>Thermal radiance scenes via NASA FIRMS - written when anomalies present</span>",
        unsafe_allow_html=True,
    )
    f4.markdown(
        f"**TROPOMI SO₂ retrievals**  \n{nrt.get('tropomi_images', 0)} in window  \n"
        f"<span style='color:#666;font-size:0.85em'>Daily global coverage; cloud-masked in wet season</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Risk Map",
        "Thermal Clusters",
        "SO₂ Validation",
        "Feature Importance",
        "Alert Table",
        "About",
    ]
)


with tab1:
    st.subheader("Combined Risk Score Map")
    st.caption("XGBoost + Isolation Forest combined risk index over the TNP corridor")

    layer_choice = st.selectbox(
        "Display layer",
        [
            "Risk tiers (HIGH / MEDIUM / LOW)",
            "Continuous risk score",
            "SAR dark spots",
            "Thermal hotspots (VIIRS)",
        ],
    )

    m = folium.Map(location=[5.40, 6.85], zoom_start=9, tiles="CartoDB dark_matter")
    folium.Rectangle(
        bounds=[[5.00, 6.50], [5.80, 7.20]],
        color="#378ADD",
        fill=False,
        weight=2,
        dash_array="8 4",
        tooltip="TNP corridor study area",
    ).add_to(m)

    tier_colors = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#B5D4F4"}
    tier_radii = {"HIGH": 8, "MEDIUM": 5, "LOW": 3}

    if not df_risk.empty:
        if layer_choice == "Risk tiers (HIGH / MEDIUM / LOW)":
            for _, row in df_risk.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=tier_radii.get(row["risk_tier"], 3),
                    color=tier_colors.get(row["risk_tier"], "#888780"),
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=(
                        f"Risk: {row['risk_tier']} | "
                        f"Score: {round(row['combined_risk_score'], 3)} | "
                        f"VV: {round(row['VV'], 2)} dB"
                    ),
                ).add_to(m)

        elif layer_choice == "Continuous risk score":
            max_score = df_risk["combined_risk_score"].max()
            for _, row in df_risk.iterrows():
                intensity = row["combined_risk_score"] / max_score if max_score > 0 else 0
                color = "#{:02x}00{:02x}".format(int(255 * intensity), int(255 * (1 - intensity)))
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.75,
                    tooltip=f"Score: {round(row['combined_risk_score'], 3)}",
                ).add_to(m)

        elif layer_choice == "SAR dark spots" and sar_gj:
            for feat in sar_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                props = feat["properties"]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=6,
                    color="#E24B4A",
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=f"VV: {round(props.get('VV', 0), 2)} dB",
                ).add_to(m)

        elif layer_choice == "Thermal hotspots (VIIRS)" and firms_gj:
            for feat in firms_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                props = feat["properties"]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=5,
                    color="#EF9F27",
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=(
                        f"T21: {round(props.get('T21_max_K', 0), 1)}K | "
                        f"Source: {props.get('likely_source', '?')}"
                    ),
                ).add_to(m)

    st_folium(m, height=520, use_container_width=True)

    leg1, leg2, leg3 = st.columns(3)
    leg1.markdown("RED: HIGH risk zone")
    leg2.markdown("AMBER: MEDIUM risk zone")
    leg3.markdown("BLUE: LOW risk / background")


with tab2:
    st.subheader("DBSCAN Thermal Anomaly Clusters")
    st.caption(
        "50 persistent VIIRS thermal radiance hotspots grouped into 8 candidate "
        "combustion sites (illegal refinery flares, gas flares, or vegetation burns) "
        "via DBSCAN spatial clustering"
    )

    if not df_clusters.empty:
        m2 = folium.Map(location=[5.40, 6.85], zoom_start=9, tiles="CartoDB positron")
        folium.Rectangle(
            bounds=[[5.00, 6.50], [5.80, 7.20]],
            color="#185FA5",
            fill=False,
            weight=2,
            dash_array="8 4",
        ).add_to(m2)

        risk_colors = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#1D9E75"}

        for _, row in df_clusters.iterrows():
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=max(6, int(row["n_hotspots"]) * 2),
                color=risk_colors.get(row["risk_label"], "#888780"),
                fill=True,
                fill_opacity=0.75,
                tooltip=(
                    f"Cluster {int(row['cluster_id'])} | "
                    f"{int(row['n_hotspots'])} hotspots | "
                    f"T21_max: {round(row['max_T21_K'], 1)}K | "
                    f"{row['risk_label']} | {row['dominant_source']}"
                ),
            ).add_to(m2)
            folium.Marker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                icon=folium.DivIcon(
                    html=(
                        "<div style='font-size:10px;font-weight:bold;color:#333'>"
                        f"C{int(row['cluster_id'])}</div>"
                    ),
                    icon_size=(20, 12),
                ),
            ).add_to(m2)

        if firms_gj:
            for feat in firms_gj["features"]:
                coords = feat["geometry"]["coordinates"]
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=3,
                    color="#FAC775",
                    fill=True,
                    fill_opacity=0.5,
                ).add_to(m2)

        st_folium(m2, height=480, use_container_width=True)
        st.markdown("**Cluster summary**")
        display_cols = [
            c
            for c in [
                "cluster_id",
                "n_hotspots",
                "centroid_lat",
                "centroid_lon",
                "max_T21_K",
                "mean_fire_count",
                "dominant_source",
                "risk_label",
            ]
            if c in df_clusters.columns
        ]
        st.dataframe(df_clusters[display_cols].round(3), width="stretch", hide_index=True)
    else:
        st.warning("Cluster data not found. Run Module 2.")


with tab3:
    st.subheader("TROPOMI Dry Season Validation")
    st.caption("Oct-Dec 2023 | SO₂ co-location with thermal anomaly clusters")

    st.markdown(
        """
**Key finding:** 6 out of 8 thermal anomaly clusters show episodic SO₂ elevation
> 3 Dobson Units during the dry season - the chemical signature consistent with
illegal crude oil burning, layered on top of the thermal signal.

**Why episodic?** Artisanal refineries burn intermittently (batch processing).
TROPOMI captures background on most days but spikes of 3-5+ DU on active burn days.
This low-mean / high-max pattern is the expected signature for intermittent sources.
The dual signature (persistent thermal anomaly + episodic SO₂) is what raises a
cluster from "candidate" to "high-confidence combustion source."
"""
    )

    if not df_val.empty:
        n_conf = int(df_val["SO2_confirmed"].sum()) if "SO2_confirmed" in df_val else 0
        st.metric(
            "Dual-signature candidates",
            f"{n_conf} / {len(df_val)}",
            delta="Thermal + SO₂ co-located",
            help=(
                "Clusters where a persistent thermal anomaly co-locates with "
                "episodic SO₂ elevation > 3 DU. The combination is consistent "
                "with crude burning but still requires ground verification."
            ),
        )

        fig = go.Figure()
        bar_colors = ["#E24B4A" if bool(v) else "#B5D4F4" for v in df_val["SO2_confirmed"]]
        fig.add_trace(
            go.Bar(
                x=["C" + str(int(i)) for i in df_val["cluster_id"]],
                y=df_val["SO2_mean_DU"],
                marker_color=bar_colors,
                opacity=0.85,
                name="SO2 mean",
                text=[str(round(v, 3)) for v in df_val["SO2_mean_DU"]],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                x=["C" + str(int(i)) for i in df_val["cluster_id"]],
                y=df_val["SO2_max_DU"],
                marker_color="#FAC775",
                opacity=0.5,
                name="SO2 max (episodic)",
            )
        )
        fig.add_hline(y=1.5, line_dash="dash", line_color="#854F0B", annotation_text="Mean threshold (1.5 DU)")
        fig.add_hline(y=3.0, line_dash="dot", line_color="#E24B4A", annotation_text="Episodic threshold (3.0 DU)")
        fig.update_layout(
            title="SO₂ per thermal anomaly cluster (Oct-Dec 2023 dry season)",
            yaxis_title="SO2 (Dobson Units)",
            height=400,
            barmode="group",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Validation table**")
        display_cols = [
            c
            for c in [
                "cluster_id",
                "SO2_mean_DU",
                "SO2_max_DU",
                "n_hotspots",
                "max_T21_K",
                "SO2_confirmed",
                "verdict",
            ]
            if c in df_val.columns
        ]
        df_validation = df_val[display_cols].copy()
        if "SO2_confirmed" in df_validation:
            df_validation["Confirmed"] = df_validation["SO2_confirmed"].map(
                {True: "Confirmed", False: "Not confirmed"}
            )
        st.dataframe(df_validation.round(3), width="stretch", hide_index=True)

        if n_conf > 0 and {"SO2_confirmed", "centroid_lat", "centroid_lon"}.issubset(df_val.columns):
            st.markdown("**Recommended NNPC field inspection coordinates:**")
            for _, row in df_val[df_val["SO2_confirmed"]].iterrows():
                st.markdown(
                    f"- **Cluster {int(row['cluster_id'])}:** "
                    f"{round(row['centroid_lat'], 4)} N, "
                    f"{round(row['centroid_lon'], 4)} E - {row.get('verdict', '')}"
                )
    else:
        st.warning("Validation data not found. Run Module 2b.")


with tab4:
    st.subheader("Feature Importance - Which sensor drives the risk score?")

    if model_cfg and "feature_cols" in model_cfg:
        try:
            import xgboost as xgb_lib

            model_path = os.path.join(MODEL_DIR, "xgb_risk_scorer.json")
            if os.path.exists(model_path):
                model = xgb_lib.XGBClassifier()
                model.load_model(model_path)
                importances = model.feature_importances_
                feat_cols = model_cfg["feature_cols"]
                feat_df = pd.DataFrame(
                    {"Feature": feat_cols, "Importance": importances}
                ).sort_values("Importance", ascending=True)

                fig = px.bar(
                    feat_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#B5D4F4", "#378ADD", "#E24B4A"],
                    title="XGBoost Feature Importance (Gain)",
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis={"autorange": "reversed"},
                    coloraxis_showscale=False,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                fig.update_xaxes(gridcolor="#f0f0f0")
                st.plotly_chart(fig, width="stretch")

                top_feat = feat_df.iloc[-1]["Feature"]
                st.markdown("**Interpretation**")
                st.markdown(f"- **Top signal: {top_feat}** - change in SAR backscatter vs baseline")
                st.markdown("- **SAR_change_dB:** New darkening in recent period = new oil on water")
                st.markdown("- **NDVI_change:** Vegetation loss along pipeline ROW = oil contamination")
                st.markdown("- **VV / VH:** Raw SAR backscatter - oil suppresses radar return on water")

                m1c, m2c, m3c = st.columns(3)
                m1c.metric("CV Accuracy", f"{round(model_cfg.get('cv_accuracy', 0) * 100, 1)}%")
                m2c.metric("Training samples", str(model_cfg.get("n_samples", 0)))
                m3c.metric("Model version", str(model_cfg.get("model_version", "1.0")))
            else:
                st.warning("Model file not found. Run Module 3.")
        except Exception as e:
            st.error("Could not load model: " + str(e))
    else:
        st.warning("Model config not found. Run Module 3.")


with tab5:
    st.subheader("Active Alerts - Highest Risk Locations")

    if not df_risk.empty:
        tier_filter = st.selectbox("Filter by risk tier", ["All", "HIGH only", "MEDIUM and above"])
        if tier_filter == "HIGH only":
            df_alerts = df_risk[df_risk["risk_tier"] == "HIGH"].copy()
        elif tier_filter == "MEDIUM and above":
            df_alerts = df_risk[df_risk["risk_tier"].isin(["HIGH", "MEDIUM"])].copy()
        else:
            df_alerts = df_risk.copy()

        df_alerts = df_alerts.sort_values("combined_risk_score", ascending=False)
        df_alerts["Risk"] = df_alerts["risk_tier"].map(
            {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}
        )
        alert_cols = [
            c
            for c in [
                "lat",
                "lon",
                "combined_risk_score",
                "Risk",
                "VV",
                "dark_spot_mask",
                "SAR_change_dB",
                "NDVI_change",
            ]
            if c in df_alerts.columns
        ]

        st.dataframe(df_alerts[alert_cols].round(3), width="stretch", hide_index=True)
        st.caption(f"{len(df_alerts)} locations shown")

        csv_cols = [c for c in alert_cols if c != "Risk"] + ["risk_tier"]
        csv_cols = [c for c in dict.fromkeys(csv_cols) if c in df_alerts.columns]
        csv = df_alerts[csv_cols].round(3).to_csv(index=False)
        st.download_button(
            label="Download alert table as CSV",
            data=csv,
            file_name="pipelinewatch_alerts_" + datetime.now().strftime("%Y%m%d") + ".csv",
            mime="text/csv",
        )
    else:
        st.warning("Risk scored data not found. Run Module 3.")


with tab6:
    st.subheader("About PipelineWatch-NG")
    st.markdown(
        """
Nigeria loses an estimated **150,000+ barrels per day** to crude oil theft.
The Niger Delta's mangrove creek terrain makes ground surveillance nearly impossible.
Existing commercial monitoring systems cost **$50,000-$500,000 per year**,
placing them beyond the reach of Nigerian government agencies.

PipelineWatch-NG is an open-source, multi-sensor, cloud-native pipeline for
automated crude oil theft detection in the Niger Delta - deployable at zero cost.
"""
    )

    st.markdown("**Sensor stack**")
    sensor_df = pd.DataFrame(
        [
            {"Sensor": "Sentinel-1 SAR", "Measures": "C-band radar backscatter", "Interpretation": "Dark spots on water = candidate oil slick", "Cloud-free": "Yes", "Night": "Yes", "Cost": "Free"},
            {"Sensor": "VIIRS (FIRMS)", "Measures": "Mid-IR thermal radiance", "Interpretation": "Persistent thermal anomaly = candidate combustion source", "Cloud-free": "Yes", "Night": "Yes", "Cost": "Free"},
            {"Sensor": "TROPOMI SO₂", "Measures": "UV column density of SO₂", "Interpretation": "Episodic plume = chemical fingerprint of crude burning", "Cloud-free": "Partial", "Night": "No", "Cost": "Free"},
            {"Sensor": "Sentinel-2 MSI", "Measures": "Optical reflectance (NDVI/NDWI)", "Interpretation": "Vegetation dieback = candidate ROW contamination", "Cloud-free": "No", "Night": "No", "Cost": "Free"},
        ]
    )
    st.dataframe(sensor_df, width="stretch", hide_index=True)

    st.markdown("**ML pipeline**")
    st.markdown("- XGBoost classifier - 98.1% cross-validated accuracy")
    st.markdown("- Isolation Forest - unsupervised anomaly detection (10% contamination)")
    st.markdown("- DBSCAN spatial clustering - haversine metric, 5.5 km search radius")
    st.markdown("- Combined risk index - 0.6 x XGBoost + 0.4 x Isolation Forest")

    st.markdown("**Key results**")
    st.markdown("- 50 persistent thermal radiance hotspots detected over the TNP corridor (VIIRS)")
    st.markdown("- 8 DBSCAN cluster sites identified as candidate persistent combustion sources")
    st.markdown("- **6/8 clusters show co-located episodic SO₂ elevation (>3 DU)** consistent with crude burning, via TROPOMI dry-season validation")
    st.markdown("- 11,685 chronic SAR dark-spot pixels mapped (candidate oil-on-water signatures)")
    st.markdown("- 2 HIGH-risk zones identified at 5.637 N / 6.625 E and 5.727 N / 6.625 E")

    st.markdown("**Target stakeholders**")
    st.markdown("- NNPC - Nigerian National Petroleum Corporation")
    st.markdown("- Nigerian Ministry of Petroleum Resources")
    st.markdown("- International oil companies: Shell, Chevron, TotalEnergies")
    st.markdown("- World Bank / donor organisations")

    st.markdown("**Data products and interpretation chain**")
    st.markdown(
        "PipelineWatch-NG consumes raw geophysical measurements and applies "
        "domain-rule interpretation layers to produce risk indicators. The chain is:"
    )
    st.markdown("- **Sensor measurement** → what the satellite physically records")
    st.markdown("- **Thermal anomaly / SAR dark spot / SO₂ column** → first-order interpretation of the radiance or backscatter signal")
    st.markdown("- **Candidate combustion / oil signature** → cross-checked when two or more independent signals co-locate")
    st.markdown("- **Risk tier (HIGH / MEDIUM / LOW)** → final operational alert, derived from the combined signal index")

    st.markdown("**Important disclaimer on FIRMS / VIIRS thermal data**")
    st.info(
        "VIIRS thermal hotspots from NASA FIRMS measure mid-IR brightness temperature, "
        "not fire directly. Persistent thermal anomalies are *candidate* combustion "
        "sources - they may also be agricultural burns, gas flares, volcanic activity, "
        "or other industrial heat sources. Per the NASA / ESDIS terms of use, FIRMS "
        "data is provided 'as is' and is **not intended for tactical decision-making "
        "or local-scale conditions**. PipelineWatch-NG applies SO₂ co-location and SAR "
        "cross-checks to raise candidate hotspots to higher-confidence alerts, but "
        "ground verification by field teams is required before any operational action.\n\n"
        "Reference: [NASA FIRMS VIIRS Active Fire documentation]"
        "(https://firms.modaps.eosdis.nasa.gov/descriptions/FIRMS_VIIRS_Firehotspots.html)"
    )

    st.markdown("**Data attribution**")
    st.markdown(
        "- FIRMS / VIIRS active-fire data: NASA / ESDIS / LANCE - "
        "[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov)"
    )
    st.markdown(
        "- Sentinel-1 SAR, Sentinel-2 MSI, Sentinel-5P TROPOMI: "
        "ESA Copernicus Programme, accessed via Google Earth Engine"
    )
    st.markdown("- Compute: Google Earth Engine (free for noncommercial / research use)")

    st.markdown("---")
    st.markdown(
        "Built by **Emmanuel Ibekwe** | "
        "[GitHub](https://github.com/Ibekwemmanuel7/pipelinewatch-ng) | "
        "ibekwemmanuel@gmail.com"
    )
