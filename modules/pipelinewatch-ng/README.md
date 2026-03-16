# PipelineWatch-NG

**Satellite-based crude oil theft and pipeline monitoring — Niger Delta, Nigeria**

> Nigeria loses an estimated 150,000+ barrels per day to crude oil theft. The Niger Delta's
> mangrove creek terrain makes ground surveillance nearly impossible. PipelineWatch-NG uses
> free space-based sensors to detect oil spills, illegal refineries, and suspicious vessels —
> entirely from the cloud, with no data downloads.

---

## Live demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pipelinewatch-ng.streamlit.app)

---

## All-weather detection stack

| Sensor | Signal | Cloud? | Night? | Source |
|--------|--------|:------:|:------:|--------|
| Sentinel-1 SAR | Oil spill dark spots (VV backscatter) | ✅ | ✅ | ESA / GEE free |
| FIRMS / VIIRS | Illegal refinery fire hotspots (FRP) | ✅ | ✅ | NASA / GEE free |
| TROPOMI SO₂ | Chemical plumes from crude burning | ✅ | ❌ | ESA / GEE free |
| Sentinel-2 MSI | Vegetation dieback along pipeline ROW | ❌ | ❌ | ESA / GEE free |
| AIS vessel data | Suspicious bunkering vessels | ✅ | ✅ | MarineTraffic |

**Study area:** Trans Niger Pipeline (TNP) corridor — Rumuekpe to Bonny Terminal  
**Bounding box:** 5.0°–5.8°N, 6.5°–7.2°E

---

## Module structure

```
Module 1 — Data Ingestion        notebooks/01_module1_ingestion.ipynb
Module 2 — Processing            notebooks/02_module2_processing.ipynb   [coming]
Module 3 — ML Anomaly Detection  notebooks/03_module3_ml.ipynb           [coming]
Module 4 — Risk Fusion Dashboard notebooks/04_module4_dashboard.ipynb    [coming]
```

---

## Quickstart (Windows / Anaconda)

```bat
REM 1. Clone the repo
git clone https://github.com/yourusername/pipelinewatch-ng.git
cd pipelinewatch-ng

REM 2. Run the setup script from Anaconda Prompt
setup_windows.bat

REM 3. Open the notebook
conda activate pipelinewatch
cd notebooks
jupyter notebook
```

Then open `01_module1_ingestion.ipynb`, set your GEE project ID in Cell 2, and run all cells.

### GEE project setup
1. Go to https://code.earthengine.google.com
2. Accept the Terms of Service
3. Create a new project (free for non-commercial research)
4. Copy your Project ID into Cell 2 of the notebook

---

## Tech stack

- **Cloud compute:** Google Earth Engine (Python API) — no satellite downloads
- **ML:** PyTorch (CNN oil spill classifier), XGBoost (fire/gas risk scorer)
- **Dashboard:** Streamlit + Folium — interactive map, deployed to Streamlit Community Cloud
- **Language:** Python 3.10, Jupyter, Anaconda

---

## Target stakeholders

- NNPC (Nigerian National Petroleum Corporation)
- Nigerian Ministry of Petroleum Resources
- International oil companies: Shell, Chevron, TotalEnergies
- World Bank / donor organisations funding Niger Delta environmental monitoring

---

## Repository structure

```
pipelinewatch-ng/
├── app.py                       Streamlit dashboard entry point
├── requirements.txt
├── setup_windows.bat            One-click Windows setup
├── config/
│   └── roi_gee.py               Study area, dates, thresholds
├── modules/
│   └── m1_ingestion/
│       ├── gee_sentinel1.py     SAR ingestion + dark spot detection
│       └── gee_fire_gas.py      FIRMS/VIIRS + TROPOMI SO₂ ingestion
├── notebooks/
│   └── 01_module1_ingestion.ipynb
├── data/
│   └── cached/                  Pre-computed GeoJSON outputs (committed)
│       ├── sar_dark_spots.geojson
│       ├── fire_hotspots.geojson
│       ├── so2_anomalies.geojson
│       └── m1_metadata.json
└── outputs/
    ├── m1_fire_timeseries.png
    └── m1_so2_comparison.png
```
