# ============================================================
# PipelineWatch-NG — Study Area Configuration
# Trans Niger Pipeline (TNP) corridor, Niger Delta, Nigeria
# ============================================================

# Bounding box: [west, south, east, north]
ROI_BOUNDS = [6.50, 5.00, 7.20, 5.80]

# Human-readable
ROI_COORDS = {
    "west":  6.50,
    "east":  7.20,
    "south": 5.00,
    "north": 5.80,
}

# For Folium / geemap map centering [lat, lon]
ROI_CENTER = [5.40, 6.85]
ROI_ZOOM   = 9

# Analysis periods
BASELINE_START = "2023-01-01"
BASELINE_END   = "2023-06-30"
RECENT_START   = "2024-01-01"
RECENT_END     = "2024-06-30"

# Sensor thresholds (tuned for Niger Delta)
SO2_THRESHOLD_DU       = 3.0    # Dobson Units — above = anomalous SO2 elevation
SAR_DARK_SPOT_SIGMA    = 1.5    # stddev below mean VV = dark spot candidate
FIRMS_BRIGHTNESS_K     = 330    # Kelvin — VIIRS T21 threshold for fire detection
VESSEL_LOITER_SPEED_KN = 2.0    # knots — below = loitering / bunkering behaviour
