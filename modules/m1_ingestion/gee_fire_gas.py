"""
PipelineWatch-NG — Module 1
FIRMS/VIIRS fire hotspots + TROPOMI SO₂ gas plume ingestion via Google Earth Engine

All-weather / night-time detection strategy
--------------------------------------------
The Niger Delta has persistent cloud cover for 6-9 months of the year.
Optical sensors (Sentinel-2, Landsat) are blind during these periods.

This module uses two cloud-independent, night-capable sensors:

1. VIIRS (Visible Infrared Imaging Radiometer Suite) — on Suomi-NPP and NOAA-20
   - Thermal infrared at 375 m resolution
   - Detects fire radiative power (FRP) day and night
   - "kpofire" artisanal refineries burn continuously — persistent FRP signature

2. TROPOMI (TROPOspheric Monitoring Instrument) — on Sentinel-5P
   - UV/visible backscatter, not dependent on scene illumination
   - Detects SO₂ column density at 5.5 × 3.5 km resolution
   - Artisanal crude cracking releases H₂S → SO₂
   - Persistent SO₂ elevation over fixed location = strong refinery signal
   - Works through thin-to-moderate cloud (UV penetrates better than visible)

Combined SO₂ + FRP signature = high-confidence illegal refinery identification
"""

import ee


# ─────────────────────────────────────────────
# FIRMS / VIIRS
# ─────────────────────────────────────────────

def get_firms_collection(roi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """
    Query NASA FIRMS VIIRS active fire detections.

    GEE dataset: "FIRMS"
    Source:      VIIRS S-NPP 375 m (I-band thermal infrared)
    Key band:    T21 — brightness temperature at 3.75 µm (fire channel), in Kelvin
                 confidence — "l" (low), "n" (nominal), "h" (high)

    Niger Delta context: legitimate uses of fire include:
    - Agricultural burning (dry season, Jan-Mar): patchy, short-duration
    - Gas flaring (NNPC/IOC infrastructure): fixed locations, continuous
    - Illegal artisanal refineries: fixed locations, continuous, often lower FRP
                                    than flares but elevated SO₂ co-signal

    We use FRP persistence and co-location with SO₂ to discriminate.
    """
    return (
        ee.ImageCollection("FIRMS")
        .filterDate(start, end)
        .filterBounds(roi)
        .select(["T21", "confidence", "line_number"])
    )


def compute_firms_composite(collection: ee.ImageCollection,
                             roi: ee.Geometry) -> ee.Image:
    """
    Build a fire persistence composite.

    max() composite: captures the highest brightness temperature observed
    at each pixel over the period. Persistent fire sources (refineries, flares)
    will show up as consistently hot pixels; transient fires will only appear
    in the max if they were ever very hot.

    count() composite: how many times each pixel was detected as a fire.
    High count + moderate max T21 = persistent low-intensity source (artisanal refinery)
    Low count + high max T21 = possible flare or agricultural burn
    """
    # Max brightness temperature
    t21_max = collection.select("T21").max().rename("T21_max").clip(roi)

    # Detection count (number of scenes where fire was detected)
    t21_count = collection.select("T21").count().rename("fire_count").clip(roi)

    # Mean brightness temperature (persistent sources have stable mean)
    t21_mean = collection.select("T21").mean().rename("T21_mean").clip(roi)

    return t21_max.addBands(t21_count).addBands(t21_mean)


def extract_fire_hotspots(firms_composite: ee.Image,
                           roi: ee.Geometry,
                           t21_threshold_k: float = 330.0,
                           min_count: int = 3) -> ee.FeatureCollection:
    """
    Extract persistent fire hotspot clusters as point features.

    Filters
    -------
    t21_threshold_k : only pixels with max T21 > 330 K (background ~290-300 K)
    min_count       : detected fire on at least 3 separate acquisition dates
                      (filters single-event agricultural fires)

    Returns a FeatureCollection with attributes:
    - T21_max     : peak brightness temperature (K)
    - T21_mean    : mean brightness temperature (K)
    - fire_count  : number of detections in period
    - persistence : fire_count / total_days (0-1 scale)
    - signal_type : "FIRMS_fire_hotspot"
    - likely_source: "refinery_candidate" | "flare" | "agricultural" (heuristic)
    """
    # Mask to hot, persistent pixels
    hot_mask = (firms_composite.select("T21_max").gt(t21_threshold_k)
                .And(firms_composite.select("fire_count").gte(min_count)))

    hotspot_image = firms_composite.updateMask(hot_mask)

    # Vectorise to points (centroid of connected hot pixel clusters)
    vectors = hotspot_image.select("T21_max").reduceToVectors(
        geometry=roi,
        scale=375,
        geometryType="centroid",
        eightConnected=True,
        maxPixels=1e9,
        bestEffort=True
    )

    def annotate_hotspot(feature):
        geom = feature.geometry()

        stats = firms_composite.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.max(), sharedInputs=True
            ),
            geometry=geom.buffer(500),
            scale=375,
            maxPixels=1e6
        )

        t21_max   = ee.Number(stats.get("T21_max_max"))
        t21_mean  = ee.Number(stats.get("T21_mean_mean"))
        count     = ee.Number(stats.get("fire_count_mean"))

        # Heuristic source classification
        # Gas flares: very high T21 (>380K), fixed location
        # Artisanal refineries: moderate T21 (330-370K), persistent, will have SO2
        # Agricultural: sporadic, lower count
        likely_source = ee.Algorithms.If(
            t21_max.gt(380),
            "flare_candidate",
            ee.Algorithms.If(
                count.gte(10),
                "refinery_candidate",
                "agricultural_or_other"
            )
        )

        return feature.set({
            "T21_max_K":      t21_max,
            "T21_mean_K":     t21_mean,
            "fire_count":     count,
            "signal_type":    "FIRMS_fire_hotspot",
            "likely_source":  likely_source,
            "confidence":     ee.Algorithms.If(count.gte(10), "high", "nominal")
        })

    return vectors.map(annotate_hotspot)


# ─────────────────────────────────────────────
# TROPOMI SO₂
# ─────────────────────────────────────────────

def get_tropomi_so2_collection(roi: ee.Geometry,
                                start: str, end: str) -> ee.ImageCollection:
    """
    Query Sentinel-5P TROPOMI SO₂ total column density.

    GEE dataset: "COPERNICUS/S5P/NRTI/L3_SO2"
    Band:        SO2_column_number_density (mol/m²)
    Resolution:  5.5 × 3.5 km at nadir
    Revisit:     Daily (13:30 local solar time, descending node)

    Niger Delta background SO₂: ~0.5–1.5 DU (natural + background pollution)
    Gas flare SO₂:               5–20 DU
    Dense illegal refinery:      3–50+ DU

    1 Dobson Unit = 2.69 × 10²⁰ molecules/cm² = 4.46 × 10⁻⁴ mol/m²
    Conversion: DU = mol/m² × (1 / 4.46e-4) = mol/m² × 2241.5
    """
    return (
        ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_SO2")
        .filterDate(start, end)
        .filterBounds(roi)
        .select([
            "SO2_column_number_density",
            "SO2_column_number_density_amf",
            "cloud_fraction"
        ])
    )


def compute_so2_composite(collection: ee.ImageCollection,
                           roi: ee.Geometry,
                           cloud_fraction_max: float = 0.3) -> ee.Image:
    """
    Build a clean SO₂ composite with cloud masking and unit conversion.

    Cloud masking: TROPOMI SO₂ retrievals are unreliable when cloud_fraction > 0.3
    because clouds shield the boundary layer from UV measurement.
    We mask high-cloud scenes before compositing.

    We build two composites:
    - so2_mean_du  : temporal mean → identifies persistent sources
    - so2_max_du   : temporal max  → captures episodic high-emission events
    - so2_p75_du   : 75th percentile → robust to outliers, shows typical elevated days
    """
    def mask_clouds(image):
        cloud = image.select("cloud_fraction")
        mask  = cloud.lt(cloud_fraction_max)
        return image.updateMask(mask)

    def to_dobson_units(image):
        # mol/m² → Dobson Units
        so2_du = (image.select("SO2_column_number_density")
                       .multiply(2241.5)
                       .rename("SO2_DU"))
        return image.addBands(so2_du)

    clean = collection.map(mask_clouds).map(to_dobson_units)

    so2_mean = clean.select("SO2_DU").mean().rename("SO2_mean_DU").clip(roi)
    so2_max  = clean.select("SO2_DU").max().rename("SO2_max_DU").clip(roi)
    so2_p75  = clean.select("SO2_DU").reduce(
        ee.Reducer.percentile([75])
    ).rename("SO2_p75_DU").clip(roi)
    so2_count = clean.select("SO2_DU").count().rename("SO2_obs_count").clip(roi)

    print(f"  Clean TROPOMI scenes after cloud masking: using percentile composite")
    return so2_mean.addBands(so2_max).addBands(so2_p75).addBands(so2_count)


def extract_so2_anomalies(so2_composite: ee.Image,
                           roi: ee.Geometry,
                           threshold_du: float = 3.0,
                           min_obs: int = 5) -> ee.FeatureCollection:
    """
    Extract SO₂ anomaly zones exceeding the background threshold.

    Threshold rationale
    -------------------
    Niger Delta background: ~1-2 DU (from natural sources + distant industrial)
    3 DU threshold: ~2x background — conservative to capture all refinery signals
    Tune upward to 5 DU if too many false positives in the area.

    min_obs: require at least 5 clean observations (filters areas with poor
    data coverage, e.g. consistently cloudy sub-regions)
    """
    # Mask to anomalous, well-observed pixels
    anomaly_mask = (so2_composite.select("SO2_mean_DU").gt(threshold_du)
                    .And(so2_composite.select("SO2_obs_count").gte(min_obs)))

    anomaly_image = so2_composite.select("SO2_mean_DU").updateMask(anomaly_mask)

    vectors = anomaly_image.reduceToVectors(
        geometry=roi,
        scale=5500,    # ~TROPOMI native resolution
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e9,
        bestEffort=True
    )

    def annotate_so2(feature):
        geom = feature.geometry()
        stats = so2_composite.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.max(), sharedInputs=True
            ),
            geometry=geom,
            scale=5500,
            maxPixels=1e6
        )

        so2_mean = ee.Number(stats.get("SO2_mean_DU_mean"))
        so2_max  = ee.Number(stats.get("SO2_max_DU_max"))
        obs      = ee.Number(stats.get("SO2_obs_count_mean"))
        area     = geom.area(maxError=100)

        confidence = ee.Algorithms.If(
            so2_mean.gt(8),  "high",
            ee.Algorithms.If(so2_mean.gt(5), "nominal", "low")
        )

        return feature.set({
            "SO2_mean_DU":  so2_mean,
            "SO2_max_DU":   so2_max,
            "obs_count":    obs,
            "area_m2":      area,
            "signal_type":  "TROPOMI_SO2_anomaly",
            "confidence":   confidence
        })

    return vectors.map(annotate_so2)


# ─────────────────────────────────────────────
# Combined risk scoring
# ─────────────────────────────────────────────

def compute_fire_gas_risk_score(so2_composite: ee.Image,
                                 firms_composite: ee.Image,
                                 roi: ee.Geometry,
                                 so2_threshold: float = 3.0,
                                 t21_threshold: float = 330.0) -> ee.Image:
    """
    Combine SO₂ and fire signals into a 0–3 illegal refinery risk score.

    Score
    -----
    0 : no signal
    1 : fire only (could be agricultural / flare)
    2 : SO₂ only (could be distant industrial)
    3 : fire + SO₂ co-located → high-confidence illegal refinery candidate

    Resampled to 1 km grid for consistent risk fusion in Module 4.
    """
    so2_mask  = so2_composite.select("SO2_mean_DU").gt(so2_threshold)
    fire_mask = firms_composite.select("T21_max").gt(t21_threshold)

    # Resample fire to SO₂ resolution for spatial overlap test
    fire_resampled = fire_mask.reproject(crs="EPSG:4326", scale=5500)
    so2_resampled  = so2_mask.reproject(crs="EPSG:4326", scale=5500)

    score = (so2_resampled.multiply(2)
             .add(fire_resampled.multiply(1))
             .rename("fire_gas_risk_score")
             .clip(roi))

    return score


def get_firms_viz_params() -> dict:
    return {
        "T21_max": {
            "min": 300, "max": 420,
            "palette": ["E6F1FB", "FAC775", "EF9F27", "E24B4A", "791F1F"],
            "label": "VIIRS max brightness temp (K)"
        }
    }


def get_so2_viz_params() -> dict:
    return {
        "SO2_mean_DU": {
            "min": 0, "max": 15,
            "palette": ["E1F5EE", "9FE1CB", "1D9E75", "085041", "04342C"],
            "label": "TROPOMI SO₂ mean column (Dobson Units)"
        }
    }
