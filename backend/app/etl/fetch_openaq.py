"""
OpenAQ Data Fetcher (backend-local copy)
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAQ_BASE_URL = "https://api.openaq.org/v2"
MEASUREMENTS_ENDPOINT = f"{OPENAQ_BASE_URL}/measurements"


def fetch_city_air_quality(
    city: str,
    parameter: str = "pm25",
    limit: int = 100,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    params = {
        "city": city,
        "parameter": parameter,
        "limit": limit,
        "order_by": "datetime",
        "sort": "desc",
    }
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    api_key = os.getenv("OPENAQ_API_KEY", "")
    if api_key:
        params["api_key"] = api_key
    response = requests.get(MEASUREMENTS_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def process_and_store(raw_data: Dict[str, Any], output_path: str) -> str:
    results = raw_data.get("results", [])
    if not results:
        raise ValueError("No results to process")
    df = pd.json_normalize(results)
    if "date.utc" not in df.columns or "value" not in df.columns:
        raise ValueError("Unexpected OpenAQ schema")
    df = df[["date.utc", "parameter", "value", "unit", "location"]].dropna()
    df["timestamp"] = pd.to_datetime(df["date.utc"], utc=True)
    df["hour"] = df["timestamp"].dt.floor("H")
    agg = (
        df.groupby(["hour", "parameter"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_value"})
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.lower().endswith(".csv"):
        agg.to_csv(output_path, index=False)
    else:
        agg.to_json(output_path, orient="records", date_format="iso")
    return output_path


def write_city_hourly_pm(
    city: str, hours: int = 72, out_dir: str = "./app/etl/processed_data"
) -> str:
    now = datetime.utcnow()
    resp = fetch_city_air_quality(
        city,
        parameter="pm25",
        limit=1000,
        date_from=(now - timedelta(hours=hours)).isoformat(),
        date_to=now.isoformat(),
    )
    out_path = os.path.join(out_dir, f"openaq_{city.lower().replace(' ', '_')}.csv")
    return process_and_store(resp, out_path)


def write_recent_pm25_mean(
    city: str, hours: int = 24, out_dir: str = "./app/etl/processed_data"
) -> str:
    now = datetime.utcnow()
    resp = fetch_city_air_quality(
        city,
        parameter="pm25",
        limit=1000,
        date_from=(now - timedelta(hours=hours)).isoformat(),
        date_to=now.isoformat(),
    )
    values = [
        float(r.get("value"))
        for r in resp.get("results", [])
        if r.get("value") is not None
    ]
    mean_pm25 = float(sum(values) / len(values)) if values else None
    payload = {
        "city": city,
        "hours": hours,
        "count": len(values),
        "mean_pm25": mean_pm25,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"pm25_{city.lower().replace(' ', '_')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


"""
OpenAQ Data Fetcher

Fetches air quality data from the OpenAQ API.
OpenAQ provides real-time and historical air quality data from monitoring stations worldwide.

API Documentation: https://docs.openaq.org/
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAQ API endpoints
OPENAQ_BASE_URL = "https://api.openaq.org/v2"
MEASUREMENTS_ENDPOINT = f"{OPENAQ_BASE_URL}/measurements"
LOCATIONS_ENDPOINT = f"{OPENAQ_BASE_URL}/locations"


def fetch_city_air_quality(
    city: str,
    parameter: str = "pm25",
    limit: int = 100,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch air quality measurements for a specific city

    Args:
        city: City name (e.g., "Lahore", "Mumbai", "Beijing")
        parameter: Air quality parameter to fetch
            Options: pm25, pm10, no2, so2, o3, co, bc
        limit: Maximum number of measurements to return
        date_from: Start date in ISO format (optional)
        date_to: End date in ISO format (optional)

    Returns:
        Dictionary containing API response with measurements

    Example response:
    {
        "meta": {"found": 100, "limit": 100},
        "results": [
            {
                "location": "US Diplomatic Post: Lahore",
                "parameter": "pm25",
                "value": 85.3,
                "unit": "µg/m³",
                "date": {"utc": "2025-01-01T00:00:00Z"}
            },
            ...
        ]
    }
    """
    # Build query parameters
    params = {
        "city": city,
        "parameter": parameter,
        "limit": limit,
        "order_by": "datetime",
        "sort": "desc",
    }

    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    # Add API key if available
    api_key = os.getenv("OPENAQ_API_KEY", "")
    if api_key:
        params["api_key"] = api_key

    try:
        logger.info(f"Fetching {parameter} data for {city} (limit={limit})")
        response = requests.get(MEASUREMENTS_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Fetched {data['meta']['found']} measurements")

        return data

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from OpenAQ: {e}")
        raise


def fetch_multiple_parameters(
    city: str, parameters: List[str] = ["pm25", "pm10", "no2"], limit: int = 100
) -> Dict[str, Any]:
    """
    Fetch multiple air quality parameters for a city

    Args:
        city: City name
        parameters: List of parameters to fetch
        limit: Limit per parameter

    Returns:
        Dictionary with parameter names as keys
    """
    results = {}

    for param in parameters:
        try:
            data = fetch_city_air_quality(city, parameter=param, limit=limit)
            results[param] = data.get("results", [])
        except Exception as e:
            logger.warning(f"Failed to fetch {param} for {city}: {e}")
            results[param] = []

    return results


def fetch_locations_in_city(city: str) -> List[Dict[str, Any]]:
    """
    Get all monitoring locations in a city

    Args:
        city: City name

    Returns:
        List of location dictionaries with metadata
    """
    params = {"city": city, "limit": 100}

    api_key = os.getenv("OPENAQ_API_KEY", "")
    if api_key:
        params["api_key"] = api_key

    try:
        logger.info(f"Fetching monitoring locations for {city}")
        response = requests.get(LOCATIONS_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        locations = data.get("results", [])
        logger.info(f"Found {len(locations)} monitoring locations")

        return locations

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching locations: {e}")
        raise


def fetch_recent_data(
    city: str, hours: int = 24, parameter: str = "pm25"
) -> List[Dict[str, Any]]:
    """
    Fetch recent air quality data for the last N hours

    Args:
        city: City name
        hours: Number of hours to look back
        parameter: Air quality parameter

    Returns:
        List of measurements
    """
    now = datetime.utcnow()
    date_from = (now - timedelta(hours=hours)).isoformat()
    date_to = now.isoformat()

    data = fetch_city_air_quality(
        city, parameter=parameter, limit=1000, date_from=date_from, date_to=date_to
    )

    return data.get("results", [])


def process_and_store(raw_data: Dict[str, Any], output_path: str) -> str:
    """Convert OpenAQ response to hourly aggregates and write to disk.

    - Flattens results to DataFrame
    - Drops missing values
    - Aggregates to hourly mean per parameter
    - Saves to CSV/JSON based on extension
    """
    results = raw_data.get("results", [])
    if not results:
        raise ValueError("No results to process")

    df = pd.json_normalize(results)
    # Expected columns: 'date.utc', 'parameter', 'value', 'unit', 'location'
    if "date.utc" not in df.columns or "value" not in df.columns:
        raise ValueError("Unexpected OpenAQ schema")

    df = df[["date.utc", "parameter", "value", "unit", "location"]].dropna()
    df["timestamp"] = pd.to_datetime(df["date.utc"], utc=True)
    df["hour"] = df["timestamp"].dt.floor("H")

    agg = (
        df.groupby(["hour", "parameter"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_value"})
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.lower().endswith(".csv"):
        agg.to_csv(output_path, index=False)
    else:
        agg.to_json(output_path, orient="records", date_format="iso")

    logger.info(f"Wrote processed OpenAQ data: {output_path} ({len(agg)} rows)")
    return output_path


def write_city_hourly_pm(
    city: str, hours: int = 72, out_dir: str = "./app/etl/processed_data"
) -> str:
    """Fetch recent PM2.5 for a city and write hourly means CSV.

    Output file: etl/processed_data/openaq_{city}.csv
    Columns: hour (ISO), parameter, mean_value
    """
    now = datetime.utcnow()
    resp = fetch_city_air_quality(
        city,
        parameter="pm25",
        limit=1000,
        date_from=(now - timedelta(hours=hours)).isoformat(),
        date_to=now.isoformat(),
    )
    out_path = os.path.join(out_dir, f"openaq_{city.lower().replace(' ', '_')}.csv")
    return process_and_store(resp, out_path)


def write_recent_pm25_mean(
    city: str, hours: int = 24, out_dir: str = "./ app/etl/processed_data"
) -> str:
    """
    Fetch recent PM2.5 measurements and write a small JSON summary to disk.

    Output schema (JSON):
    {
      "city": "Lahore",
      "hours": 24,
      "count": 120,
      "mean_pm25": 63.4,
      "generated_at": "2025-01-01T00:00:00Z"
    }
    """
    results = fetch_recent_data(city=city, hours=hours, parameter="pm25")
    values = [float(r.get("value")) for r in results if r.get("value") is not None]
    mean_pm25 = float(sum(values) / len(values)) if values else None

    payload = {
        "city": city,
        "hours": hours,
        "count": len(values),
        "mean_pm25": mean_pm25,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"pm25_{city.lower().replace(' ', '_')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Wrote PM2.5 summary for {city} (hours={hours}, count={len(values)}, mean={mean_pm25}) to {out_path}"
    )
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAQ PM2.5 ETL")
    parser.add_argument("--city", type=str, default=os.getenv("CITY", "Lahore"))
    parser.add_argument("--hours", type=int, default=int(os.getenv("HOURS", 24)))
    parser.add_argument(
        "--out_dir", type=str, default=os.getenv("OUT_DIR", "./app/etl/processed_data")
    )
    args = parser.parse_args()

    try:
        path = write_recent_pm25_mean(args.city, args.hours, args.out_dir)
        print(path)
    except Exception as e:
        logger.error(f"ETL failed: {e}")
        raise
