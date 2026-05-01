# PipelineWatch-NG

**Satellite-based crude oil theft and pipeline monitoring - Niger Delta, Nigeria**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pipelinewatch-ng.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Powered-green.svg)](https://earthengine.google.com)
[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/License-PolyForm--NC--1.0.0-blue.svg)](LICENSE)

> Nigeria loses an estimated **150,000+ barrels per day** to crude oil theft. The Niger Delta's mangrove creek terrain makes ground surveillance nearly impossible. PipelineWatch-NG uses free space-based sensors to detect oil spills, illegal refineries, and suspicious activity - entirely from the cloud, with zero satellite downloads.

---

## Live Demo

**[https://pipelinewatch-ng.streamlit.app](https://pipelinewatch-ng.streamlit.app)**

No login required. No installation. Open in any browser.

---

## What makes this different

Most existing approaches to Niger Delta monitoring are:
- Single-sensor (SAR only, or optical only) - blind in clouds or at night
- Static maps published in academic papers - not operational
- Commercial systems costing $50k–$500k/year - inaccessible to NNPC or Nigerian government agencies

PipelineWatch-NG is the first open-source, multi-sensor, cloud-native pipeline for automated crude oil theft detection in the Niger Delta. It combines SAR spill detection, thermal refinery identification, and SO₂ chemical fingerprinting into a single reproducible framework deployable at zero data cost.

---

## All-weather detection stack

The pipeline is built around a deliberate separation between what each sensor *measures* (raw geophysical signal) and what those measurements *imply* (the operational indicator of interest). This separation matters because the underlying products carry well-documented limitations and disclaimers - see the FIRMS note in the [Data products and disclaimers](#data-products-and-disclaimers) section below.

| Sensor | What it measures | First-order interpretation | Cloud-penetrating | Night-capable | Cost |
|--------|----------------|----------------------------|:-----------------:|:-------------:|------|
| Sentinel-1 SAR | C-band radar backscatter (VV, VH) | Dark spots on water = candidate oil slick | ✅ C-band radar | ✅ Active sensor | Free |
| VIIRS (FIRMS) | Mid-IR brightness temperature | Persistent thermal anomaly = candidate combustion source | ✅ Thermal IR | ✅ Thermal IR | Free |
| TROPOMI SO₂ | UV column density of SO₂ | Episodic plume = chemical signature consistent with crude burning | ✅ UV backscatter | ❌ Daytime | Free |
| Sentinel-2 MSI | Optical reflectance (NDVI, NDWI) | Vegetation dieback = candidate ROW contamination | ❌ Optical | ❌ Daytime | Free |

All data accessed via **Google Earth Engine** - no downloads, no storage costs. Single-signature alerts are *candidates*; the pipeline raises confidence by requiring two or more independent signals to co-locate (e.g. persistent thermal anomaly + episodic SO₂ elevation).

---

## Study area

**Trans Niger Pipeline (TNP) corridor**
Rumuekpe → Oguta → Bonny Terminal
Bounding box: 5.0°–5.8°N, 6.5°–7.2°E

One of the highest crude oil theft corridors in Nigeria, traversing mangrove creek networks that make ground surveillance impractical.

---

## Module structure

```
Module 1 - Data Ingestion
  Sentinel-1 SAR (30 scenes), VIIRS thermal scenes via FIRMS (181), TROPOMI SO₂ (278)
  Output: GeoJSON feature collections, 50 persistent thermal anomaly hotspots detected

Module 2 - Processing & Feature Engineering
  Sentinel-2 NDVI/NDWI, SAR change detection, DBSCAN clustering
  Output: 270-sample feature table, 8 candidate combustion-source clusters, 11,685 chronic SAR dark-spot pixels

Module 3 - ML Anomaly Detection
  XGBoost risk classifier (98.1% CV accuracy), Isolation Forest anomaly detection
  Output: Risk-scored dataset, trained models, feature importance

Module 4 - Risk Fusion Dashboard
  Streamlit + Folium interactive dashboard, live public deployment
  Output: https://pipelinewatch-ng.streamlit.app
```

---

## Key results

| Metric | Value |
|--------|-------|
| S1 SAR scenes processed | 59 (30 baseline + 29 recent) |
| VIIRS thermal scenes (via FIRMS) | 361 (180 baseline + 181 recent) |
| TROPOMI SO₂ retrievals | 591 (313 baseline + 278 recent) |
| Persistent thermal hotspots detected | 50 |
| Candidate combustion-source clusters (DBSCAN) | 8 |
| Clusters with co-located episodic SO₂ (>3 DU) | 6 of 8 |
| Chronic SAR dark-spot pixels mapped | 11,685 |
| XGBoost CV accuracy (weak-label proxy) | 98.1% |
| HIGH-risk zones identified | 2 |
| Top predictive signal | `SAR_change_dB` |

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
1. `SAR_change_dB` - new darkening vs baseline (new spill indicator)
2. `NDVI_change` - vegetation loss along pipeline ROW
3. `VV` / `VH` - raw SAR backscatter

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

1. `01_module1_ingestion.ipynb` - set your GEE project ID, run all cells
2. `02_module2_processing.ipynb` - run all cells
3. `03_module3_ml.ipynb` - run all cells

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

- **NNPC** - Nigerian National Petroleum Corporation
- **Nigerian Ministry of Petroleum Resources**
- **International oil companies** - Shell, Chevron, TotalEnergies operating in Nigeria
- **World Bank / donor organisations** funding Niger Delta environmental monitoring

---

## Technology stack

| Component | Technology |
|-----------|-----------|
| Cloud compute | Google Earth Engine (Python API) |
| SAR processing | GEE COPERNICUS/S1_GRD |
| Optical processing | GEE COPERNICUS/S2_SR_HARMONIZED |
| Thermal anomaly detection | GEE FIRMS (VIIRS active-fire product) |
| SO₂ retrieval | GEE COPERNICUS/S5P/NRTI/L3_SO2 |
| ML | XGBoost, scikit-learn Isolation Forest |
| Clustering | DBSCAN (scikit-learn, haversine metric) |
| Dashboard | Streamlit + Folium + Plotly |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.10 |

---

## Limitations and future work

- **SO₂ detection**: TROPOMI retrievals are cloud-masked during the Niger Delta wet season (Jun–Nov). Future work should test the Oct–Dec dry season window where SO₂ signals are cleaner.
- **Ground truth**: No confirmed theft incident database exists for the TNP corridor. Weak proxy labels were used for ML training. Collaboration with NNPC field teams would enable supervised training.
- **AIS vessel tracking**: Module 1 architecture includes AIS bunkering vessel detection - not yet implemented. MarineTraffic API integration is the next priority.
- **SAR speckle filtering**: Deferred from interactive notebooks due to memory constraints. A GEE Export Task approach in Module 2 will enable proper Lee filtering at scale.
- **Temporal resolution**: Current pipeline runs on 6-month composites. A near-real-time weekly alert mode is feasible with the same architecture.

---

## Data products and disclaimers

PipelineWatch-NG consumes raw geophysical measurements from public satellite missions and applies its own interpretation layer to produce operational risk indicators. The chain is deliberately staged so that each step's evidentiary weight is auditable:

```
Sensor measurement (radiance / backscatter / column density)
        ↓
First-order interpretation (thermal anomaly / SAR dark spot / SO₂ plume)
        ↓
Cross-signature confirmation (e.g. persistent thermal anomaly + episodic SO₂)
        ↓
Operational risk tier (HIGH / MEDIUM / LOW)
```

### A note on FIRMS / VIIRS thermal data

The VIIRS data accessed via NASA's Fire Information for Resource Management System (FIRMS) measures **mid-IR brightness temperature**, not fire directly. Persistent thermal anomalies are *candidate* combustion sources - they may also be agricultural burns, gas flares, volcanic activity, or other industrial heat sources. PipelineWatch-NG explicitly raises the confidence of a thermal anomaly only when a second independent signal (typically TROPOMI SO₂ co-location, or SAR change adjacent to the hotspot) supports the combustion interpretation.

Per NASA / ESDIS terms of use, FIRMS data is provided **"as is"** and is **not intended for tactical decision-making or local-scale conditions**. Ground verification by qualified field teams is required before any operational action is taken on any alert produced by this system.

Reference: [NASA FIRMS - VIIRS Active Fire data product description](https://firms.modaps.eosdis.nasa.gov/descriptions/FIRMS_VIIRS_Firehotspots.html)

### A note on the 98.1% CV accuracy figure

The XGBoost classifier reports 98.1% cross-validated accuracy against **weak proxy labels** derived from domain rules (not against ground-truth crude theft incidents). This number reflects internal model self-consistency, not confirmed real-world detection rate. Supervised retraining with a verified incident dataset - ideally from NNPC, IOC field teams, or Tantita Security operations - would be required before this figure could be claimed as operational accuracy.

### Data attribution

- **FIRMS / VIIRS active-fire data**: NASA / ESDIS / LANCE - [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov)
- **Sentinel-1 SAR, Sentinel-2 MSI, Sentinel-5P TROPOMI**: ESA Copernicus Programme, accessed via Google Earth Engine
- **Compute platform**: Google Earth Engine (free for noncommercial / research use)

---

## License

Copyright © 2024–2026 Emmanuel Ibekwe. All rights reserved.

This project is licensed under the **PolyForm Noncommercial License 1.0.0** - see the `LICENSE` file for the full terms. In summary: you may view, run, and modify this code for non-commercial purposes (research, teaching, personal use, internal evaluation). You may **not** use it, in whole or in part, in any commercial product or service without a separate commercial license from the author.

For commercial licensing, partnership inquiries, or pilot programs with national oil companies, 