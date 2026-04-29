"""
PipelineWatch-NG — Module 1
Sentinel-1 SAR ingestion and preprocessing via Google Earth Engine

Physics background
------------------
SAR (Synthetic Aperture Radar) transmits microwave pulses and measures the
backscattered energy. Over water:
  - Calm / oily water:  very LOW backscatter (specular reflection away from sensor)
  - Rough water:        HIGH backscatter (Bragg scattering returns signal)

Oil on water dampens capillary waves → locally suppresses Bragg scattering →
appears as a DARK PATCH in VV polarisation imagery.

This module:
  1. Queries the Sentinel-1 GRD archive for the TNP corridor
  2. Applies a Lee speckle filter (reduces salt-and-pepper SAR noise)
  3. Computes a dark-spot anomaly mask (candidate oil spill zones)
  4. Builds temporal composites for baseline vs recent comparison
  5. Exports results as GeoJSON feature collections for the dashboard
"""

import ee
import sys


def get_s1_collection(roi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """
    Query Sentinel-1 GRD IW scenes over the study area.

    Parameters
    ----------
    roi   : ee.Geometry  — study area bounding box
    start : str          — "YYYY-MM-DD"
    end   : str          — "YYYY-MM-DD"

    Filter rationale
    ----------------
    - GRD (Ground Range Detected): amplitude backscatter, terrain-corrected,
      ready for analysis without complex phase processing
    - IW mode: 250 km swath at 10 m resolution — optimal for wide-area surveillance
    - VV + VH dual-pol: VV best for ocean/water oil spill detection;
      VH useful for vegetation change (Module 2)
    - DESCENDING orbit: more consistent incidence angle over the Niger Delta
      (empirically found to give better water/oil contrast than ascending passes)
    """
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select(["VV", "VH"])
    )


def apply_lee_speckle_filter(image: ee.Image, kernel_size: int = 7) -> ee.Image:
    """
    Lee speckle filter for SAR imagery.

    SAR images suffer from multiplicative speckle noise (random pixel-to-pixel
    variation). The Lee filter estimates local statistics and applies an
    adaptive weight:

        W = σ²_signal / (σ²_signal + σ²_noise)
        filtered = mean + W * (pixel - mean)

    Where W → 1 preserves detail in heterogeneous areas,
    and W → 0 smooths homogeneous areas (open water).

    kernel_size=7 is the standard for oil spill detection per ESA guidelines.
    """
    def filter_single_band(band_name):
        band = image.select([band_name])
        kernel = ee.Kernel.square(kernel_size // 2)

        band_mean = band.reduceNeighborhood(
            reducer=ee.Reducer.mean(), kernel=kernel
        )
        band_var = band.reduceNeighborhood(
            reducer=ee.Reducer.variance(), kernel=kernel
        )

        # Estimate noise variance as the mean of local variances
        # (assumption: noise is stationary across the scene)
        noise_var_dict = band_var.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=image.geometry(),
            scale=500,
            maxPixels=1e8,
            bestEffort=True
        )
        noise_var = ee.Number(noise_var_dict.values().get(0))

        # Adaptive weight
        weight = band_var.divide(band_var.add(noise_var))
        filtered = band_mean.add(weight.multiply(band.subtract(band_mean)))
        return filtered.rename(band_name)

    vv = filter_single_band("VV")
    vh = filter_single_band("VH")
    return (vv.addBands(vh)
              .copyProperties(image, ["system:time_start", "system:index"]))


def compute_sar_features(image: ee.Image, roi: ee.Geometry,
                         sigma_threshold: float = 1.5) -> ee.Image:
    """
    Compute oil-spill-related feature bands from calibrated SAR backscatter.

    Bands added
    -----------
    dark_spot_mask      : binary — 1 where VV is anomalously low (spill candidate)
    dark_spot_magnitude : how many stddev below mean (higher = darker = more suspicious)
    vv_vh_ratio         : VV/VH — oil slicks have characteristic ratio vs wind shadows
    cross_pol_diff      : VV - VH in dB — another discriminator
    """
    vv = image.select("VV")
    vh = image.select("VH")

    # Scene-level statistics for dark spot detection
    stats = vv.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.stdDev(), sharedInputs=True
        ),
        geometry=roi,
        scale=100,
        maxPixels=1e9,
        bestEffort=True
    )

    mean_vv = ee.Number(stats.get("VV_mean"))
    std_vv  = ee.Number(stats.get("VV_stdDev"))

    # Dark spot threshold: pixels below (mean - sigma * stdDev)
    threshold = mean_vv.subtract(std_vv.multiply(sigma_threshold))

    dark_mask = vv.lt(threshold).rename("dark_spot_mask")
    dark_mag  = (mean_vv.subtract(vv)).divide(std_vv).rename("dark_spot_magnitude")

    # VV/VH ratio (linear, not dB — convert first)
    vv_lin = ee.Image(10).pow(vv.divide(10))
    vh_lin = ee.Image(10).pow(vh.divide(10))
    ratio  = vv_lin.divide(vh_lin).rename("vv_vh_ratio")

    cross_pol_diff = vv.subtract(vh).rename("cross_pol_diff")

    return (image
            .addBands(dark_mask)
            .addBands(dark_mag)
            .addBands(ratio)
            .addBands(cross_pol_diff))


def build_s1_composite(roi: ee.Geometry, start: str, end: str,
                       apply_filter: bool = True,
                       sigma_threshold: float = 1.5) -> tuple:
    """
    Build a temporal median composite over a date range.

    Median compositing rationale: ships, rain cells, and wind roughness
    create transient bright/dark anomalies. Median over many scenes
    suppresses these transients, leaving persistent dark features
    (chronic spill zones, seeps) visible in the composite.

    Returns
    -------
    composite   : ee.Image   — median composite with all feature bands
    collection  : ee.ImageCollection — individual scenes (for time series)
    scene_count : int        — number of scenes used
    """
    collection = get_s1_collection(roi, start, end)
    scene_count = collection.size().getInfo()
    print(f"  S1 scenes found ({start} → {end}): {scene_count}")

    if scene_count == 0:
        print("  WARNING: No Sentinel-1 scenes found for this ROI/date range.")
        print("  Check that your ROI intersects a Sentinel-1 coverage area.")
        return None, None, 0

    if apply_filter:
        print("  Applying Lee speckle filter...")
        collection = collection.map(apply_lee_speckle_filter)

    print("  Computing SAR feature bands...")
    collection = collection.map(
        lambda img: compute_sar_features(img, roi, sigma_threshold)
    )

    composite = collection.median().clip(roi)
    return composite, collection, scene_count


def extract_dark_spot_vectors(composite: ee.Image, roi: ee.Geometry,
                               min_area_m2: float = 10000) -> ee.FeatureCollection:
    """
    Convert the binary dark_spot_mask raster to polygon vectors.

    Each polygon represents a candidate oil spill zone with attributes:
    - mean_vv            : mean VV backscatter in the zone (dB)
    - dark_spot_magnitude: how anomalous (stddev units)
    - area_m2            : zone area in square metres

    min_area_m2 = 10,000 m² = 1 hectare minimum — filters pixel-level noise.
    A meaningful oil spill in the Niger Delta would typically be ≥ 1 ha.
    """
    mask = composite.select("dark_spot_mask")
    vv   = composite.select("VV")
    mag  = composite.select("dark_spot_magnitude")

    # Vectorise the mask
    vectors = mask.selfMask().reduceToVectors(
        geometry=roi,
        scale=40,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e9,
        bestEffort=True
    )

    # Annotate with VV stats
    def annotate(feature):
        geom = feature.geometry()
        stats = vv.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=40,
            maxPixels=1e6
        )
        mag_stats = mag.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=40,
            maxPixels=1e6
        )
        area = geom.area(maxError=10)
        return feature.set({
            "mean_vv_db":          stats.get("VV"),
            "dark_spot_magnitude": mag_stats.get("dark_spot_magnitude"),
            "area_m2":             area,
            "signal_type":         "SAR_dark_spot",
            "confidence":          ee.Algorithms.If(
                                       mag_stats.get("dark_spot_magnitude"),
                                       "nominal", "low"
                                   )
        })

    annotated = vectors.map(annotate)

    # Filter by minimum area
    filtered = annotated.filter(ee.Filter.gte("area_m2", min_area_m2))
    return filtered


def get_s1_viz_params() -> dict:
    """Visualisation parameters for geemap / folium display."""
    return {
        "VV": {
            "min": -25, "max": 0,
            "palette": ["000000", "404040", "808080", "c0c0c0", "ffffff"],
            "label": "Sentinel-1 VV backscatter (dB)"
        },
        "dark_spot_mask": {
            "min": 0, "max": 1,
            "palette": ["00000000", "E24B4A"],
            "opacity": 0.7,
            "label": "SAR dark spot candidates (oil spill)"
        },
        "dark_spot_magnitude": {
            "min": 0, "max": 4,
            "palette": ["E6F1FB", "85B7EB", "E24B4A"],
            "label": "Dark spot anomaly magnitude (σ)"
        }
    }
