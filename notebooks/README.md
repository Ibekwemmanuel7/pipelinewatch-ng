# PipelineWatch-NG

**Satellite-based crude oil theft monitoring — Niger Delta, Nigeria**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pipelinewatch-ng.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Powered-green.svg)](https://earthengine.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Nigeria loses an estimated **150,000+ barrels per day** to crude oil theft. The Niger Delta's mangrove creek terrain makes ground surveillance nearly impossible. Existing commercial monitoring systems cost $50,000–$500,000 per year - beyond the reach of Nigerian agencies. PipelineWatch-NG is the first open-source, multi-sensor, cloud-native pipeline for automated crude oil theft detection in the Niger Delta, deployable at zero data cost.

---

## Live Demo

**[https://pipelinewatch-ng.streamlit.app](https://pipelinewatch-ng.streamlit.app)**

No login required. No installation. Open in any browser.

---

## Key Results

| Metric | Value |
|--------|-------|
| Sentinel-1 SAR scenes processed | 59 (30 baseline + 29 recent) |
| FIRMS/VIIRS images processed | 361 (180 + 181) |
| TROPOMI SO₂ images processed | 591 (313 + 278) |
| Persistent fire hotspots detected | 50 |
| DBSCAN refinery site clusters | 8 |
| **Clusters confirmed by TROPOMI SO₂** | **6 / 8** |
| Chronic SAR spill pixels | 11,685 |
| XGBoost CV accuracy | 98.1% |
| HIGH-risk zones identified | 2 |
| Top predictive signal | SAR_change_dB (0.28 importance) |

### TROPOMI Dry Season Validation

6 out of 8 fire clusters were chemically confirmed by **episodic SO₂ elevation > 3 Dobson Units** during the Oct–Dec 2023 dry season. This low-mean / high-max SO₂ pattern is the signature of **intermittent artisanal crude burning** ("kpofire" operations), which fire up for batch processing then go dark — consistent with the known operating pattern of illegal refineries in the Niger Delta.

| Cluster | SO₂ mean (DU) | SO₂ max (DU) | Hotspots | T21 max (K) | Status |
|---------|:------------:|:------------:|:--------:|:-----------:|--------|
| C0 | 0.071 | 4.263 | 2 | 349.3 | ✅ CONFIRMED (episodic) |
| C1 | 0.027 | 3.991 | 12 | 348.9 | ✅ CONFIRMED (episodic) |
| C2 | 0.106 | 3.451 | 2 | 341.0 | ✅ CONFIRMED (episodic) |
| C3 | -0.009 | 2.171 | 3 | 332.5 | ❌ Unconfirmed |
| C4 | -0.016 | 3.070 | 2 | 341.8 | ✅ CONFIRMED (episodic) |
| C5 | 0.129 | 4.391 | 4 | 346.3 | ✅ CONFIRMED (episodic) |
| C6 | 0.064 | 3.211 | 3 | 338.3 | ✅ CONFIRMED (episodic) |
| C7 | 0.048 | 2.767 | 15 | 346.9 | ❌ Unconfirmed |

---

## All-Weather Detection Stack

| Sensor | Signal | Cloud-penetrating | Night-capable | Cost |
|--------|--------|:-----------------:|:-------------:|------|
| Sentinel-1 SAR | Oil spill dark spots | ✅ C-band radar | ✅ Active sensor | Free |
| FIRMS / VIIRS | Illegal refinery fire hotspots | ✅ Thermal IR | ✅ Thermal IR | Free |
| TROPOMI SO₂ | Chemical plumes from crude burning | ✅ UV backscatter | ❌ Daytime only | Free |
| Sentinel-2 MSI | Vegetation dieback along pipeline ROW | ❌ Optical | ❌ Daytime only | Free |

All data accessed via **Google Earth Engine** — no downloads, no storage costs.

---

## Study Area

**Trans Niger Pipeline (TNP) corridor**
Rumuekpe → Oguta → Bonny Terminal, Rivers/Imo States, Nigeria
Bounding box: 5.0°–5.8°N, 6.5°–7.2°E

One of Nigeria's highest crude oil theft corridors, traversing mangrove creek
networks that make ground surveillance nearly impossible.

---

## Module Structure

```
Module 1 — Data Ingestion
  notebooks/01_module1_ingestion.ipynb
  Sentinel-1 SAR (59 scenes), FIRMS/VIIRS (361 images), TROPOMI SO₂ (591 images)
  Outputs: GeoJSON feature collections, 50 fire hotspots, summary charts

Module 2 — Processing and Feature Engineering
  notebooks/02_module2_processing.ipynb
  Sentinel-2 NDVI/NDWI, SAR change detection (11,685 chronic spill px),
  DBSCAN clustering (8 sites), 270-sample × 11-feature table

Module 2b — TROPOMI Dry Season Validation
  notebooks/02b_tropomi_validation.ipynb
  Oct-Dec 2023 dry season SO₂ analysis → 6/8 clusters confirmed

Module 3 — ML Anomaly Detection
  notebooks/03_module3_ml.ipynb
  XGBoost (98.1% CV), Isolation Forest, combined 0-1 risk index
  2 HIGH-risk zones at 5.637°N/6.625°E and 5.727°N/6.625°E

Module 4 — Dashboard
  app.py
  Streamlit + Folium → https://pipelinewatch-ng.streamlit.app

Module 5 — Weekly NRT Update
  notebooks/05_module5_nrt_update.ipynb
  Run every 7 days → auto-pulls FIRMS + TROPOMI → updates dashboard
```

---

## ML Pipeline

```
Feature table (270 samples × 11 features)
         ↓
Rule-based weak labels (LOW / MEDIUM / HIGH)
         ↓
XGBoost classifier   (n_estimators=200, 5-fold CV, 98.1% accuracy)
         ↓
Isolation Forest     (contamination=0.10, 27 anomalies detected)
         ↓
Combined risk index  (0.6 × XGBoost + 0.4 × Isolation Forest)
         ↓
Risk tier assignment (HIGH ≥ 0.65 | MEDIUM ≥ 0.35 | LOW < 0.35)
```

**Top features by XGBoost gain:**

| Rank | Feature | Importance | Interpretation |
|------|---------|:----------:|----------------|
| 1 | SAR_change_dB | 0.282 | New darkening vs baseline — strongest new spill indicator |
| 2 | NDVI_change | 0.239 | Vegetation loss along pipeline ROW |
| 3 | VV | 0.093 | Raw SAR backscatter — oil suppresses radar return |
| 4 | VH | 0.088 | Cross-pol — helps discriminate oil from wind shadow |

---

## Quickstart

### Prerequisites

- Windows, Linux, or macOS with Anaconda
- Google account (for free GEE access at [code.earthengine.google.com](https://code.earthengine.google.com))
- ~500 MB disk space

### Setup

```bash
git clone https://github.com/Ibekwemmanuel7/pipelinewatch-ng.git
cd pipelinewatch-ng

conda create -n pipelinewatch python=3.10 -y
conda activate pipelinewatch
conda install -c conda-forge geopandas rasterio -y
pip install -r requirements.txt

earthengine authenticate
```

### Run the analysis

Open Jupyter and run notebooks in order:

```bash
conda activate pipelinewatch
jupyter notebook
```

1. `notebooks/01_module1_ingestion.ipynb`
2. `notebooks/02_module2_processing.ipynb`
3. `notebooks/03_module3_ml.ipynb`
4. `notebooks/02b_tropomi_validation.ipynb`

### Run the dashboard locally

```bash
streamlit run app.py
```

### Weekly NRT monitoring

Open `notebooks/05_module5_nrt_update.ipynb` → **Kernel → Restart and Run All**

Pulls the last 7 days of FIRMS and TROPOMI, raises an alert level, and pushes
results to GitHub. Streamlit auto-redeploys within 2 minutes.

---

## Repository Structure

```
pipelinewatch-ng/
├── app.py                              Streamlit dashboard
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml                     Theme configuration
├── config/
│   └── roi_gee.py                      Study area and threshold config
├── notebooks/
│   ├── 01_module1_ingestion.ipynb      SAR + FIRMS + TROPOMI ingestion
│   ├── 02_module2_processing.ipynb     S2 NDVI, SAR change, DBSCAN
│   ├── 02b_tropomi_validation.ipynb    Dry season SO₂ validation
│   ├── 03_module3_ml.ipynb             XGBoost + Isolation Forest
│   └── 05_module5_nrt_update.ipynb     Weekly NRT alert pipeline
├── data/
│   ├── cached/                         Pre-computed outputs (GeoJSON, CSV, JSON)
│   │   ├── sar_dark_spots.geojson
│   │   ├── fire_hotspots.geojson
│   │   ├── so2_anomalies.geojson
│   │   ├── m2_feature_table.csv
│   │   ├── m2_refinery_clusters.csv
│   │   ├── m2b_tropomi_validation.csv
│   │   ├── m3_risk_scored.csv
│   │   └── m3_model_config.json
│   └── models/                         Trained ML models
│       ├── xgb_risk_scorer.json
│       ├── isolation_forest.pkl
│       └── feature_scaler.pkl
└── outputs/                            Interactive charts (HTML)
    ├── m1_fire_summary.html
    ├── m1_so2_comparison.html
    ├── m2_dbscan_clusters.html
    ├── m2_ndvi_analysis.html
    ├── m2b_tropomi_validation.html
    ├── m3_feature_importance.html
    └── m3_risk_score_map.html
```

---

## Limitations and Future Work

**SO₂ spatial resolution mismatch:** TROPOMI's 5.5 × 3.5 km footprint is coarser
than SAR (10 m) and VIIRS (375 m). Fusion at different spatial scales introduces
co-location uncertainty. Future work should use TROPOMI OFFL (offline) products
for improved retrieval accuracy.

**Ground truth:** No confirmed theft incident database exists for the TNP corridor.
Weak proxy labels were used for ML training. Collaboration with NNPC field teams
to map known incident locations would enable fully supervised training and
significantly improve precision.

**AIS vessel tracking:** Bunkering vessels in the creeks are the strongest
corroborating evidence of active theft. MarineTraffic API integration is the
next priority for Module 1.

**SAR speckle filtering:** Lee filtering was deferred from interactive notebooks
due to kernel memory constraints on resource-limited machines. A GEE Export Task
approach would enable proper filtering at scale.

**Temporal resolution:** Current pipeline runs on 6-month composites. The NRT
module (Module 5) provides weekly alerting but is not yet validated against
known incident reports.

---

## Target Stakeholders

- **NNPC** — Nigerian National Petroleum Corporation
- **Nigerian Ministry of Petroleum Resources**
- **International oil companies** — Shell, Chevron, TotalEnergies (Niger Delta operations)
- **World Bank / donor organisations** funding Niger Delta environmental monitoring

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Cloud compute | Google Earth Engine (Python API) |
| SAR processing | GEE COPERNICUS/S1_GRD |
| Optical processing | GEE COPERNICUS/S2_SR_HARMONIZED |
| Fire detection | GEE FIRMS / VIIRS |
| Gas detection | GEE COPERNICUS/S5P/NRTI/L3_SO2 |
| ML | XGBoost, scikit-learn Isolation Forest |
| Clustering | DBSCAN (scikit-learn, haversine metric) |
| Dashboard | Streamlit + Folium + Plotly |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.10 |

---

## Author

**Emmanuel Ibekwe**
[GitHub](https://github.com/Ibekwemmanuel7) · ibekwemmanuel@gmail.com · College Station, TX

---

## Acknowledgements

- ESA Copernicus Programme — Sentinel-1, Sentinel-2, Sentinel-5P data
- NASA FIRMS — VIIRS active fire data
- Google Earth Engine — cloud compute platform
