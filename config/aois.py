"""
PipelineWatch-NG AOI configuration loader.

Single source of truth for area-of-interest definitions used across:
- Notebooks (Modules 1-3, ingestion / processing / ML)
- Module 5 NRT update script (modules/m5_nrt_update.py)
- Streamlit dashboard (app.py)

Usage from a notebook or script:

    from config.aois import get_active_aoi
    name, aoi = get_active_aoi()  # uses PIPELINEWATCH_AOI env var
    bounds = aoi["bounds"]
    print(f"Active AOI: {name} - {aoi['full_name']}")

    # Or pass an explicit name:
    name, aoi = get_active_aoi("tnp_corridor")

Bounds convention: [west, south, east, north] in decimal degrees,
matching Google Earth Engine's ee.Geometry.Rectangle() argument order.
"""

from pathlib import Path
import os

import yaml

CONFIG_PATH = Path(__file__).parent / "aois.yaml"
DEFAULT_AOI = "niger_delta"
ENV_VAR = "PIPELINEWATCH_AOI"


def load_aois() -> dict:
    """Load all AOI definitions from aois.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_aoi(name: str | None = None) -> tuple[str, dict]:
    """
    Return the (name, config) tuple for the active AOI.

    Resolution order:
      1. Explicit `name` argument if passed
      2. PIPELINEWATCH_AOI environment variable
      3. Default ("niger_delta")
    """
    name = name or os.environ.get(ENV_VAR) or DEFAULT_AOI
    aois = load_aois()
    if name not in aois:
        available = ", ".join(sorted(aois.keys()))
        raise ValueError(
            f"AOI '{name}' not found in config/aois.yaml. "
            f"Available AOIs: {available}"
        )
    return name, aois[name]


def list_available_aois() -> list[str]:
    """List all AOI names defined in aois.yaml."""
    return sorted(load_aois().keys())
