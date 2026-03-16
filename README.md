# PipelineWatch-NG

**Satellite-based crude oil theft and pipeline monitoring — Niger Delta, Nigeria**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pipelinewatch-ng.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Powered-green.svg)](https://earthengine.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Nigeria loses an estimated **150,000+ barrels per day** to crude oil theft. The Niger Delta's mangrove creek terrain makes ground surveillance nearly impossible. PipelineWatch-NG uses free space-based sensors to detect oil spills, illegal refineries, and suspicious activity - entirely from the cloud, with zero satellite downloads.

---

## Live Demo

**[https://pipelinewatch-ng.streamlit.app](https://pipelinewatch-ng.streamlit.app)**

No login required. No installation. Open in any browser.

---

## What makes this different

Most existing approaches to Niger Delta monitoring are:
- Single-sensor (SAR only, or optical only) — blind in clouds or at night
- Static maps published in academic papers — not operational
- Commercial systems costing $50k–$500k/year — inaccessible to NNPC or Nigerian government agencies

PipelineWatch-NG is the first open-source, multi-sensor, cloud-native pipeline for automated crude oil theft detection in the Niger Delta — combining SAR spill detection, thermal refinery identification, and SO₂ chemical fingerprinting into a single reproducible framework deployable at zero data cost.

---

## All-weather detection stack

| Sensor | Signal detected | Cloud-penetrating | Night-capable | Cost |
|--------|----------------|:-----------------:|:-------------:|------|
| Sentinel-1 SAR | Oil spill dark spots on water | ✅ C-band radar | ✅ Active sensor | Free |
| FIRMS / VIIRS | Illegal refinery fire hotspots | ✅ Thermal IR | ✅ Thermal IR | Free |
| TROPOMI SO₂ | Chemical plumes from crude burning | ✅ UV backscatter | ❌ Daytime | Free |
| Sentinel-2 MSI | Vegetation dieback along pipeline ROW | ❌ Optical | ❌ Daytime | Free |

All data accessed via **Google Earth Engine** — no downloads, no storage costs.

---

## Study area

**Trans Niger Pipeline (TNP) corridor**
Rumuekpe → Oguta → Bonny Terminal
Bounding box: 5.0°–5.8°N, 6.5°–7.2°E

One of the highest crude oil theft corridors in Nigeria, traversing mangrove creek networks that make ground surveillance impractical.

---

## Module structure

```
Module 1 — Data Ingestion
  Sentinel-1 SAR (30 scenes), FIRMS/VIIRS (181 images), TROPOMI SO₂ (278 images)
  Output: GeoJSON feature collections, 50 fire hotspots detected

Module 2 — Processing & Feature Engineering
  Sentinel-2 NDVI/NDWI, SAR change detection, DBSCAN clustering
  Output: 270-sample feature table, 8 refinery site clusters, 11,685 chronic spill pixels

Module 3 — ML Anomaly Detection
  XGBoost risk classifier (98.1% CV accuracy), Isolation Forest anomaly detection
  Output: Risk-scored dataset, trained models, feature importance

Module 4 — Risk Fusion Dashboard
  Streamlit + Folium interactive dashboard, live public deployment
  Output: https://pipelinewatch-ng.streamlit.app
```

---

## Key results

| Metric | Value |
|--------|-------|
| S1 SAR scenes processed | 59 (30 baseline + 29 recent) |
| FIRMS/VIIRS images | 361 (180 baseline + 181 recent) |
| TROPOMI SO₂ images | 591 (313 baseline + 278 recent) |
| Fire hotspots detected | 50 |
| Refinery site clusters (DBSCAN) | 8 |
| Chronic spill pixels | 11,685 |
| XGBoost CV accuracy | 98.1% |
| HIGH risk zones identified | 2 |
| Top predictive signal | SAR_change_dB |

---

## ML pipeline

```
Feature table (270 samples × 11 features)
        ↓
Weak label assignment (domain-rule proxy labels: LOW / MEDIUM / HIGH)
        ↓
XGBoost classifier (n_estimators=200, 5-fold CV, 98.1% accuracy)
        ↓
Isolation Forest (contamination=0.10, 27 anomalies detected)
        ↓
Combined risk index (0.6 × XGBoost + 0.4 × IsoForest)
        ↓
Risk tier assignment (HIGH ≥ 0.65 | MEDIUM ≥ 0.35 | LOW < 0.35)
```

Top features by XGBoost gain:
1. `SAR_change_dB` — new darkening vs baseline (new spill indicator)
2. `NDVI_change` — vegetation loss along pipeline ROW
3. `VV` / `VH` — raw SAR backscatter

---

## Quickstart

### Prerequisites
- Windows / Anaconda
- Google account (for free GEE access)
- ~500 MB disk space

### Setup

```bat
git clone https://github.com/Ibekwemmanuel7/pipelinewatch-ng.git
cd pipelinewatch-ng

conda create -n pipelinewatch python=3.10 -y
conda activate pipelinewatch
conda install -c conda-forge geopandas rasterio -y
pip install -r requirements.txt

earthengine authenticate
```

### Run the notebooks in order

```bat
conda activate pipelinewatch
cd notebooks
jupyter notebook
```

1. `01_module1_ingestion.ipynb` — set your GEE project ID, run all cells
2. `02_module2_processing.ipynb` — run all cells
3. `03_module3_ml.ipynb` — run all cells

### Run the dashboard locally

```bat
streamlit run app.py
```

---

## Repository structure

```
pipelinewatch-ng/
├── app.py                              Streamlit dashboard
├── requirements.txt
├── .streamlit/config.toml              Theme configuration
├── notebooks/
│   ├── 01_module1_ingestion.ipynb      SAR + FIRMS + TROPOMI ingestion
│   ├── 02_module2_processing.ipynb     S2 NDVI, SAR change, DBSCAN
│   └── 03_module3_ml.ipynb             XGBoost + Isolation Forest
├── data/
│   ├── cached/                         Pre-computed GeoJSON + CSV outputs
│   │   ├── sar_dark_spots.geojson
│   │   ├── fire_hotspots.geojson
│   │   ├── so2_anomalies.geojson
│   │   ├── m2_feature_table.csv
│   │   ├── m2_refinery_clusters.csv
│   │   ├── m3_risk_scored.csv
│   │   └── m3_model_config.json
│   └── models/                         Trained ML models
│       ├── xgb_risk_scorer.json
│       ├── isolation_forest.pkl
│       └── feature_scaler.pkl
└── outputs/                            Charts and figures
```

---

## Target stakeholders

- **NNPC** — Nigerian National Petroleum Corporation
- **Nigerian Ministry of Petroleum Resources**
- **International oil companies** — Shell, Chevron, TotalEnergies operating in Nigeria
- **World Bank / donor organisations** funding Niger Delta environmental monitoring

---

## Technology stack

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

## Limitations and future work

- **SO₂ detection**: TROPOMI retrievals are cloud-masked during the Niger Delta wet season (Jun–Nov). Future work should test the Oct–Dec dry season window where SO₂ signals are cleaner.
- **Ground truth**: No confirmed theft incident database exists for the TNP corridor. Weak proxy labels were used for ML training. Collaboration with NNPC field teams would enable supervised training.
- **AIS vessel tracking**: Module 1 architecture includes AIS bunkering vessel detection — not yet implemented. MarineTraffic API integration is the next priority.
- **SAR speckle filtering**: Deferred from interactive notebooks due to memory constraints. A GEE Export Task approach in Module 2 will enable proper Lee filtering at scale.
- **Temporal resolution**: Current pipeline runs on 6-month composites. A near-real-time weekly alert mode is feasible with the same architecture.

---

## Author

**Emmanuel Ibekwe**
[GitHub](https://github.com/Ibekwemmanuel7) · ibekwemmanuel@gmail.com

---

## Acknowledgements

- ESA Copernicus Programme — Sentinel-1, Sentinel-2, Sentinel-5P data
- NASA FIRMS — VIIRS active fire data
- Google Earth Engine — cloud compute platform
- NNPC and Niger Delta environmental monitoring community
