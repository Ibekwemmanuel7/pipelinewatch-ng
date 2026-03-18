# =============================================================================
# PipelineWatch-NG  —  Study area and analysis configuration
# Trans Niger Pipeline (TNP) corridor, Niger Delta, Nigeria
# =============================================================================

# Bounding box [west, south, east, north]
ROI_BOUNDS = [6.50, 5.00, 7.20, 5.80]

ROI_CENTER = [5.40, 6.85]   # [lat, lon] for map centering
ROI_ZOOM   = 9

# Analysis periods
BASELINE_START = "2023-01-01"
BASELINE_END   = "2023-06-30"
RECENT_START   = "2024-01-01"
RECENT_END     = "2024-06-30"

# Dry season validation window (TROPOMI retrievals cleaner Oct-Dec)
DRY_SEASON_START = "2023-10-01"
DRY_SEASON_END   = "2023-12-31"

# Detection thresholds (tuned for Niger Delta)
SAR_DARK_SPOT_SIGMA  = 1.5    # stddev below mean VV = oil spill candidate
FIRMS_BRIGHTNESS_K   = 330.0  # Kelvin — VIIRS T21 fire threshold
SO2_THRESHOLD_DU     = 3.0    # Dobson Units — wet season threshold
SO2_DRY_THRESHOLD_DU = 1.5    # Dobson Units — dry season threshold (more sensitive)
SO2_MAX_CONFIRM_DU   = 3.0    # Episodic max threshold for refinery confirmation
SO2_MIN_OBS          = 10     # Minimum observations for episodic confirmation
