"""
PipelineWatch-NG Module 5: near-real-time alert update.

This script is the automation-friendly version of
notebooks/05_module5_nrt_update.ipynb. It reads the latest FIRMS/VIIRS and
TROPOMI SO2 observations from Google Earth Engine, writes the dashboard NRT
cache files, and leaves Git commits/pushes to GitHub Actions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import ee
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cached"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Make the project root importable so `from config.aois import ...` works
# regardless of where the script is invoked from.
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from config.aois import get_active_aoi  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update PipelineWatch-NG NRT alert files.")
    parser.add_argument("--days", type=int, default=7, help="Rolling NRT window in days.")
    parser.add_argument(
        "--project",
        default=os.environ.get("GEE_PROJECT_ID") or "pipelinewatch-ng",
        help="Google Earth Engine project ID.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional UTC end date as YYYY-MM-DD. Defaults to today.",
    )
    parser.add_argument(
        "--aoi",
        default=None,
        help=(
            "AOI name from config/aois.yaml. If omitted, uses the "
            "PIPELINEWATCH_AOI env var or falls back to 'niger_delta'."
        ),
    )
    return parser.parse_args()


def init_earth_engine(project_id: str) -> None:
    """Initialize Earth Engine for local use or GitHub Actions service-account use."""
    service_account = os.environ.get("EE_SERVICE_ACCOUNT")
    private_key = os.environ.get("EE_PRIVATE_KEY")
    credentials_json = os.environ.get("EE_CREDENTIALS_JSON")

    if service_account and private_key:
        key_data = private_key.replace("\\n", "\n")
        credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
        ee.Initialize(credentials, project=project_id)
        return

    if credentials_json:
        data = json.loads(credentials_json)
        credentials = ee.ServiceAccountCredentials(data["client_email"], key_data=credentials_json)
        ee.Initialize(credentials, project=project_id)
        return

    ee.Initialize(project=project_id)


def get_window(days: int, end_date: str | None) -> tuple[str, str]:
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_firms_metrics(roi: ee.Geometry, start: str, end: str, aoi: dict) -> dict:
    firms_brightness_k = float(aoi.get("firms_brightness_k", 330.0))
    baseline_thermal_hotspots = float(aoi.get("baseline_thermal_hotspots", 50))
    baseline_weeks = float(aoi.get("baseline_weeks", 26))
    baseline_weekly_avg = baseline_thermal_hotspots / baseline_weeks

    firms_nrt = (
        ee.ImageCollection("FIRMS")
        .filterDate(start, end)
        .filterBounds(roi)
        .select(["T21", "confidence", "line_number"])
    )
    image_count = int(firms_nrt.size().getInfo())

    if image_count == 0:
        return {
            "firms_images": 0,
            "fire_pixels": 0,
            "fire_anomaly": False,
            "baseline_weekly_avg": round(baseline_weekly_avg, 3),
        }

    firms_comp = (
        firms_nrt.select("T21")
        .max()
        .rename("T21_max")
        .clip(roi)
        .addBands(firms_nrt.select("T21").count().rename("fire_count").clip(roi))
    )
    hot_px = (
        firms_comp.select("T21_max")
        .gt(firms_brightness_k)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=375,
            maxPixels=1e9,
            bestEffort=True,
        )
        .getInfo()
    )
    fire_pixels = int(hot_px.get("T21_max", 0) or 0)
    fire_anomaly = fire_pixels > (baseline_weekly_avg * 1.5)

    return {
        "firms_images": image_count,
        "fire_pixels": fire_pixels,
        "fire_anomaly": bool(fire_anomaly),
        "baseline_weekly_avg": round(baseline_weekly_avg, 3),
    }


def fetch_tropomi_metrics(roi: ee.Geometry, start: str, end: str, aoi: dict) -> dict:
    so2_threshold_du = float(aoi.get("so2_threshold_du", 1.5))

    tropomi_nrt = (
        ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_SO2")
        .filterDate(start, end)
        .filterBounds(roi)
        .select(["SO2_column_number_density", "cloud_fraction"])
    )
    image_count = int(tropomi_nrt.size().getInfo())

    if image_count == 0:
        return {
            "tropomi_images": 0,
            "so2_mean_du": 0.0,
            "so2_max_du": 0.0,
            "so2_anomaly": False,
        }

    def mask_and_convert(image):
        cloud = image.select("cloud_fraction")
        so2_du = image.select("SO2_column_number_density").multiply(2241.5).rename("SO2_DU")
        return so2_du.updateMask(cloud.lt(0.4))

    so2_nrt = tropomi_nrt.map(mask_and_convert).mean().rename("SO2_mean_DU").clip(roi)
    so2_stats = so2_nrt.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
        geometry=roi,
        scale=5500,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    so2_mean = float(so2_stats.get("SO2_mean_DU_mean", 0) or 0)
    so2_max = float(so2_stats.get("SO2_mean_DU_max", 0) or 0)

    return {
        "tropomi_images": image_count,
        "so2_mean_du": round(so2_mean, 3),
        "so2_max_du": round(so2_max, 3),
        "so2_anomaly": bool(so2_mean > so2_threshold_du),
    }


def build_alert_report(start: str, end: str, fire: dict, so2: dict, aoi_name: str, aoi: dict) -> dict:
    combined_alert = fire["fire_anomaly"] or so2["so2_anomaly"]
    alert_level = "HIGH" if fire["fire_anomaly"] and so2["so2_anomaly"] else "MEDIUM" if combined_alert else "LOW"

    return {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "nrt_window": f"{start} to {end}",
        "aoi_name": aoi_name,
        "aoi_full_name": aoi.get("full_name", aoi_name),
        "aoi_country": aoi.get("country", ""),
        "aoi_bounds": aoi.get("bounds"),
        "alert_level": alert_level,
        "fire_pixels": fire["fire_pixels"],
        "fire_anomaly": fire["fire_anomaly"],
        "baseline_weekly_avg": fire["baseline_weekly_avg"],
        "so2_mean_du": so2["so2_mean_du"],
        "so2_max_du": so2["so2_max_du"],
        "so2_anomaly": so2["so2_anomaly"],
        "firms_images": fire["firms_images"],
        "tropomi_images": so2["tropomi_images"],
    }


def save_latest_report(report: dict) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "m5_nrt_latest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def update_history(report: dict) -> pd.DataFrame:
    history_path = CACHE_DIR / "m5_nrt_history.csv"
    new_row = pd.DataFrame(
        [
            {
                "date": report["nrt_window"].split(" to ")[1],
                "fire_pixels": report["fire_pixels"],
                "so2_mean_du": report["so2_mean_du"],
                "so2_max_du": report["so2_max_du"],
                "alert_level": report["alert_level"],
                "fire_anomaly": report["fire_anomaly"],
                "so2_anomaly": report["so2_anomaly"],
                "firms_images": report["firms_images"],
                "tropomi_images": report["tropomi_images"],
            }
        ]
    )

    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, new_row], ignore_index=True)
        history = history.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        history = new_row

    history.to_csv(history_path, index=False)
    return history


def write_trend_chart(history: pd.DataFrame, aoi: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "m5_nrt_trend.html"
    plot_df = history.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    so2_threshold_du = float(aoi.get("so2_threshold_du", 1.5))
    aoi_label = aoi.get("full_name", aoi.get("short_name", "AOI"))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Rolling VIIRS thermal hotspot count", "Rolling TROPOMI SO2 mean"),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["fire_pixels"],
            mode="lines+markers",
            name="Thermal hotspots",
            line=dict(color="#E24B4A", width=2),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["so2_mean_du"],
            mode="lines+markers",
            name="SO2 mean",
            line=dict(color="#1D9E75", width=2),
            marker=dict(size=6),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=so2_threshold_du,
        line_dash="dash",
        line_color="#854F0B",
        annotation_text="Threshold",
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"PipelineWatch-NG - Rolling NRT Trend<br>{aoi_label}",
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(title_text="Hot pixels", row=1, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="SO2 (DU)", row=2, col=1, gridcolor="#f0f0f0")
    fig.write_html(out)
    return out


def main() -> None:
    args = parse_args()
    aoi_name, aoi = get_active_aoi(args.aoi)
    bounds = aoi["bounds"]
    start, end = get_window(args.days, args.end_date)

    print(f"Active AOI : {aoi_name} ({aoi.get('full_name', '')})")
    print(f"Bounds     : {bounds}  [west, south, east, north]")
    print(f"NRT window : {start} to {end}")

    init_earth_engine(args.project)
    # ROI must be constructed AFTER ee.Initialize, otherwise EE client raises
    # "Earth Engine client library not initialized" on first ee.* call.
    roi = ee.Geometry.Rectangle(bounds)

    fire = fetch_firms_metrics(roi, start, end, aoi)
    so2 = fetch_tropomi_metrics(roi, start, end, aoi)
    report = build_alert_report(start, end, fire, so2, aoi_name, aoi)

    latest_path = save_latest_report(report)
    history = update_history(report)
    chart_path = write_trend_chart(history, aoi)

    print("PIPELINEWATCH-NG NRT ALERT REPORT")
    print(f"Alert level    : {report['alert_level']}")
    print(f"Thermal pixels : {report['fire_pixels']} | anomaly={report['fire_anomaly']}")
    print(f"SO2 mean DU    : {report['so2_mean_du']} | anomaly={report['so2_anomaly']}")
    print(f"Latest JSON    : {latest_path.relative_to(PROJECT_ROOT)}")
    print(f"History CSV    : {(CACHE_DIR / 'm5_nrt_history.csv').relative_to(PROJECT_ROOT)}")
    print(f"Trend chart    : {chart_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
