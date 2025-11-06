"""
Lightweight loaders to read processed ETL outputs for use in inference.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def load_recent_pm25_mean(city: str, base_dir: Optional[str] = None) -> Optional[float]:
    """
    Load PM2.5 value for model inference from ETL output.
    
    Priority:
    1. real_time_pm25 (current measurement) - BEST for model inference
    2. mean_pm25 (overall mean) - fallback if no real-time
    3. historical_mean_pm25 (forecast mean) - alternative fallback
    4. Hardcoded baseline - final fallback
    
    Note: Checks data freshness if measurement_timestamp is available.
    Data older than 24 hours is considered stale but still used.
    
    Returns None if the file does not exist or data is invalid.
    """
    if not base_dir:
        base_dir = os.getenv("PROCESSED_DATA_DIR", "/etl/processed_data")
    filename = f"pm25_{city.lower().replace(' ', '_')}.json"
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check data freshness if timestamp available
        measurement_ts = data.get("measurement_timestamp")
        if measurement_ts:
            try:
                # Parse ISO timestamp (handle both with and without timezone)
                ts_str = str(measurement_ts)
                if "T" in ts_str:
                    # Replace Z with +00:00 for proper timezone handling
                    if ts_str.endswith("Z"):
                        ts_str = ts_str[:-1] + "+00:00"
                    measurement_dt = datetime.fromisoformat(ts_str)
                    # Convert to UTC naive for comparison
                    if measurement_dt.tzinfo:
                        measurement_dt = measurement_dt.replace(tzinfo=None)
                    # Calculate age in hours
                    age_hours = (datetime.utcnow() - measurement_dt).total_seconds() / 3600
                    if age_hours > 24:
                        # Log warning but still use the data
                        import logging
                        logging.getLogger(__name__).warning(
                            f"PM2.5 data for {city} is {age_hours:.1f} hours old (timestamp: {measurement_ts})"
                        )
            except Exception:
                pass  # Ignore timestamp parsing errors
        
        # PRIORITY 1: Use real-time PM2.5 (current measurement)
        # This is the most accurate for model inference as it represents current conditions
        real_time = data.get("real_time_pm25")
        if real_time is not None:
            return float(real_time)
        
        # PRIORITY 2: Use overall mean (includes real-time + forecast)
        mean = data.get("mean_pm25")
        if mean is not None:
            return float(mean)
        
        # PRIORITY 3: Use historical mean (forecast only, if available)
        historical_mean = data.get("historical_mean_pm25")
        if historical_mean is not None:
            return float(historical_mean)
        
        # PRIORITY 4: Fallback city baselines when ETL wrote a placeholder due to API failure
        city_key = city.strip().lower()
        baselines = {
            "lahore": 100.0,
            "karachi": 80.0,
            "islamabad": 60.0,
            "peshawar": 95.0,
            "quetta": 70.0,
        }
        return baselines.get(city_key)
    except Exception:
        return None


def load_mobility_bundle(city: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load synthetic mobility, traffic, and transit datasets if present."""
    if not base_dir:
        base_dir = os.getenv("PROCESSED_DATA_DIR", "/etl/processed_data")
    slug = city.lower().replace(" ", "_")
    out: Dict[str, Any] = {}
    for name in ["mobility", "traffic", "transit"]:
        path = os.path.join(base_dir, f"{name}_{slug}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out[name] = json.load(f)
            except Exception:
                out[name] = []
        else:
            out[name] = []
    return out


def load_energy_bundle(city: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load synthetic energy, renewable, and carbon datasets if present."""
    if not base_dir:
        base_dir = os.getenv("PROCESSED_DATA_DIR", "/etl/processed_data")
    slug = city.lower().replace(" ", "_")
    out: Dict[str, Any] = {}
    for name in ["energy", "renewable", "carbon"]:
        path = os.path.join(base_dir, f"{name}_{slug}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out[name] = json.load(f)
            except Exception:
                out[name] = []
        else:
            out[name] = []
    return out
