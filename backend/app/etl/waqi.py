"""
World Air Quality Index (WAQI/AQICN) ETL for backend startup.

WAQI provides free air quality data for 10,000+ cities worldwide.
API: https://aqicn.org/api/

Steps (per city):
- api.waqi.info/feed/{city}/ → get PM2.5
- robust mean (if multiple stations)
- write JSON summary under etl/processed_data/pm25_<city>.json
"""

import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

WAQI_BASE_URL = "https://api.waqi.info"
WAQI_FEED = f"{WAQI_BASE_URL}/feed"


def _get_token() -> str:
    """Get WAQI API token from environment."""
    return os.getenv("WAQI_API_TOKEN", "")


def _fetch_city_feed(city: str, token: str, log_f=None) -> Optional[Dict[str, Any]]:
    """Fetch air quality data for a city from WAQI."""
    if not token:
        return None

    timeout = float(os.getenv("ETL_REQUEST_TIMEOUT_SECS", "10"))

    # Try multiple city name formats
    city_formats = [
        city.lower().strip().replace(" ", "-"),  # lahore
        f"{city.lower().strip().replace(' ', '-')}-pakistan",  # lahore-pakistan
        city.lower().strip(),  # lahore (no dash)
    ]

    # Add city-specific alternatives
    alt_names = {
        "Lahore": ["lahore", "lahore-pakistan"],
        "Karachi": ["karachi", "karachi-pakistan"],
        "Islamabad": ["islamabad", "islamabad-pakistan"],
        "Peshawar": ["peshawar", "peshawar-pakistan"],
        "Quetta": ["quetta", "quetta-pakistan"],
    }
    if city in alt_names:
        city_formats = alt_names[city] + city_formats

    # Remove duplicates while preserving order
    seen = set()
    city_formats = [f for f in city_formats if not (f in seen or seen.add(f))]

    params = {"token": token}

    for city_format in city_formats:
        try:
            url = f"{WAQI_FEED}/{city_format}/"
            if log_f:
                log_f.write(f"  trying WAQI URL: {url}\n")

            resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok":
                    if log_f:
                        log_f.write(f"  WAQI success: {city_format}\n")
                    return data
                else:
                    if log_f:
                        log_f.write(f"  WAQI status not ok: {data.get('status')}\n")
            elif resp.status_code == 404:
                if log_f:
                    log_f.write(f"  WAQI 404: {city_format}\n")
                continue
            else:
                resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if log_f:
                log_f.write(f"  WAQI HTTP error {e.response.status_code}: {e}\n")
            continue
        except Exception as e:
            if log_f:
                log_f.write(f"  WAQI error: {e}\n")
            continue

    return None


def _extract_pm25_values(
    data: Dict[str, Any],
) -> Tuple[List[float], Dict[str, Any], Optional[float]]:
    """
    Extract all PM2.5 values from WAQI response.

    WAQI response structure:
    {
        "status": "ok",
        "data": {
            "iaqi": {"pm25": {"v": 34}},  # Real-time PM2.5
            "forecast": {
                "daily": {
                    "pm25": [
                        {"avg": 150, "max": 159, "min": 103, "day": "2025-02-16"},
                        ...
                    ]
                }
            }
        }
    }

    Returns:
        (values_list, forecast_data): All PM2.5 values and structured forecast data
    """
    values: List[float] = []
    forecast_data: Dict[str, Any] = {}

    if data.get("status") != "ok":
        return values, forecast_data, None

    # Main station data
    aqi_data = data.get("data", {})

    # 1. PRIMARY: Extract real-time PM2.5 from iaqi (individual air quality index)
    real_time_pm25: Optional[float] = None
    iaqi = aqi_data.get("iaqi", {})
    pm25_obj = iaqi.get("pm25", {})
    pm25_value = pm25_obj.get("v")
    if pm25_value is not None:
        try:
            real_time_pm25 = float(pm25_value)
            values.append(real_time_pm25)
        except (ValueError, TypeError):
            pass

    # Store real-time separately in forecast_data
    if real_time_pm25 is not None:
        forecast_data["real_time"] = real_time_pm25

    # 2. Extract ALL forecast PM2.5 data (avg, max, min for each day)
    forecast = aqi_data.get("forecast", {})
    daily_forecast = forecast.get("daily", {})
    pm25_forecast = daily_forecast.get("pm25", [])

    if pm25_forecast:
        forecast_data["daily"] = []
        for day_data in pm25_forecast:
            day_vals = {}
            # Extract avg, max, min for each day
            for key in ["avg", "max", "min"]:
                val = day_data.get(key)
                if val is not None:
                    try:
                        day_vals[key] = float(val)
                        values.append(float(val))  # Add all values for mean calculation
                    except (ValueError, TypeError):
                        pass
            if day_vals:
                day_vals["day"] = day_data.get("day", "")
                forecast_data["daily"].append(day_vals)

    # 3. Fallback: Use main AQI if PM2.5 not directly available
    if len(values) == 0:
        aqi = aqi_data.get("aqi")
        if aqi and isinstance(aqi, (int, float)):
            if aqi > 50:
                values.append(float(aqi) * 0.5)

    return values, forecast_data, real_time_pm25


def write_recent_pm25_mean(
    city: str,
    hours: int = 24,
    out_dir: Optional[str] = None,
    log_path: Optional[str] = None,
) -> str:
    """
    Fetch PM2.5 from WAQI and write summary JSON.

    Args:
        city: City name (e.g., "Lahore")
        hours: Hours of lookback (for compatibility, but WAQI is real-time)
        out_dir: Output directory
        log_path: Optional log file to write progress

    Returns:
        Path to written JSON file
    """
    if not out_dir:
        out_dir = os.getenv("PROCESSED_DATA_DIR", "/etl/processed_data")

    log_f = None
    if log_path:
        try:
            Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
            log_f = open(log_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            log_f = None

    token = _get_token()
    values = []
    mean_pm25 = None
    count = 0
    using_waqi_data = False  # Track if we got real WAQI data
    forecast_info: Dict[str, Any] = {}  # Initialize forecast data

    if not token:
        if log_f:
            log_f.write(f"WAQI: no API token configured, using fallback baseline\n")
    else:
        if log_f:
            log_f.write(f"WAQI: fetching city={city}\n")

        data = _fetch_city_feed(city, token, log_f=log_f)

        forecast_info = {}
        real_time_pm25 = None
        measurement_timestamp = None  # Timestamp of real-time measurement
        if data:
            values, forecast_info, real_time_pm25 = _extract_pm25_values(data)

            # Extract measurement timestamp from WAQI response
            aqi_data = data.get("data", {})
            time_data = aqi_data.get("time", {})
            if time_data:
                # Try ISO format first, then Unix timestamp
                iso_time = time_data.get("iso")
                if iso_time:
                    measurement_timestamp = iso_time
                elif time_data.get("v"):  # Unix timestamp
                    try:
                        ts = int(time_data["v"])
                        measurement_timestamp = (
                            datetime.fromtimestamp(ts).isoformat() + "Z"
                        )
                    except (ValueError, TypeError):
                        pass

            if log_f:
                log_f.write(f"WAQI: extracted {len(values)} PM2.5 values\n")
                if real_time_pm25 is not None:
                    log_f.write(f"WAQI: real-time PM2.5 = {real_time_pm25}\n")
                if measurement_timestamp:
                    log_f.write(
                        f"WAQI: measurement timestamp = {measurement_timestamp}\n"
                    )
                if forecast_info.get("daily"):
                    log_f.write(
                        f"WAQI: found {len(forecast_info['daily'])} days of forecast data\n"
                    )

            if values:
                mean_pm25 = float(statistics.mean(values))
                count = len(values)
                using_waqi_data = True  # We got real WAQI data
            else:
                mean_pm25 = None
                count = 0
        else:
            values = []
            mean_pm25 = None
            count = 0
            if log_f:
                log_f.write(f"WAQI: failed to fetch data for {city}\n")

    # Fallback to hardcoded baselines if no data
    if mean_pm25 is None or count == 0:
        baseline_map = {
            "Lahore": 150.0,
            "Karachi": 120.0,
            "Islamabad": 80.0,
            "Peshawar": 140.0,
            "Quetta": 90.0,
        }
        fallback = baseline_map.get(city, 100.0)
        if log_f:
            log_f.write(f"using fallback baseline={fallback} for {city}\n")
        mean_pm25 = fallback
        count = 1
        values = [fallback]
        using_waqi_data = False  # Using baseline, not WAQI

    payload = {
        "city": city,
        "hours": hours,
        "count": count,
        "mean_pm25": mean_pm25,
        "source": "waqi" if using_waqi_data else "baseline",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    # Add real-time PM2.5 separately (current measurement)
    if real_time_pm25 is not None:
        payload["real_time_pm25"] = real_time_pm25

    # Add measurement timestamp if available
    if measurement_timestamp:
        payload["measurement_timestamp"] = measurement_timestamp

    # Add all PM2.5 values if we have them (real-time + forecast)
    if values:
        payload["pm25_values"] = values
        payload["pm25_min"] = float(min(values))
        payload["pm25_max"] = float(max(values))
        # Historical mean (excluding real-time if forecast data exists)
        if len(values) > 1 and real_time_pm25 is not None:
            forecast_values = [v for v in values if v != real_time_pm25]
            if forecast_values:
                payload["historical_mean_pm25"] = float(
                    statistics.mean(forecast_values)
                )

    # Add forecast data if available
    if forecast_info.get("daily"):
        payload["forecast"] = forecast_info

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"pm25_{city.lower().replace(' ', '_')}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if log_f:
        log_f.write(
            f"wrote {out_path} (mean_pm25={mean_pm25}, source={payload['source']})\n"
        )
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass

    return out_path
