"""
Mobility ETL (backend-local copy)
Generates synthetic bundles per city.
"""

import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_mobility_stub(
    city: str, hours: int = 24, start_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    if start_time is None:
        start_time = datetime.utcnow()
    data: List[Dict[str, Any]] = []
    for h in range(hours):
        ts = start_time + timedelta(hours=h)
        hod = ts.hour
        if 8 <= hod <= 10 or 17 <= hod <= 19:
            base = 0.8 + random.uniform(-0.1, 0.1)
        elif 0 <= hod <= 5:
            base = 0.2 + random.uniform(-0.05, 0.05)
        else:
            base = 0.6 + random.uniform(-0.1, 0.1)
        data.append(
            {
                "timestamp": ts.isoformat() + "Z",
                "city": city,
                "mobility_index": round(max(0.0, min(1.0, base)), 3),
                "retail_and_recreation": round(random.gauss(-15, 10), 1),
                "grocery_and_pharmacy": round(random.gauss(-5, 8), 1),
                "parks": round(random.gauss(-20, 15), 1),
                "transit_stations": round(random.gauss(-18, 12), 1),
                "workplaces": round(random.gauss(-25, 15), 1),
                "residential": round(random.gauss(10, 8), 1),
            }
        )
    return data


def fetch_traffic_congestion(city: str, hours: int = 24) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now = datetime.utcnow()
    for h in range(hours):
        ts = now + timedelta(hours=h)
        hod = ts.hour
        if 8 <= hod <= 10 or 17 <= hod <= 19:
            congestion = 0.7 + random.uniform(-0.1, 0.2)
        elif 0 <= hod <= 5:
            congestion = 0.1 + random.uniform(0, 0.1)
        else:
            congestion = 0.4 + random.uniform(-0.1, 0.1)
        congestion = max(0.0, min(1.0, congestion))
        out.append(
            {
                "timestamp": ts.isoformat() + "Z",
                "city": city,
                "congestion_level": round(congestion, 3),
                "avg_speed_kmh": round(60 * (1 - congestion), 1),
                "jam_length_km": round(congestion * 50, 1),
            }
        )
    return out


def fetch_public_transit_usage(city: str, hours: int = 24) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now = datetime.utcnow()
    for h in range(hours):
        ts = now + timedelta(hours=h)
        hod = ts.hour
        if 7 <= hod <= 9 or 17 <= hod <= 19:
            ridership = random.randint(5000, 8000)
        elif 0 <= hod <= 5:
            ridership = random.randint(100, 500)
        else:
            ridership = random.randint(2000, 4000)
        out.append(
            {
                "timestamp": ts.isoformat() + "Z",
                "city": city,
                "ridership": ridership,
                "occupancy_rate": round(random.uniform(0.4, 0.9), 2),
                "avg_wait_time_min": round(random.uniform(3, 15), 1),
            }
        )
    return out


def write_mobility_bundle(
    city: str, hours: int = 48, out_dir: str = "./app/etl/processed_data"
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    files: Dict[str, str] = {}
    mobility = fetch_mobility_stub(city, hours)
    traffic = fetch_traffic_congestion(city, hours)
    transit = fetch_public_transit_usage(city, hours)

    def _write(name: str, data: List[Dict[str, Any]]):
        path = os.path.join(out_dir, f"{name}_{city.lower().replace(' ', '_')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        files[name] = path

    _write("mobility", mobility)
    _write("traffic", traffic)
    _write("transit", transit)
    return files


"""
Mobility Data Fetcher

Fetches mobility and traffic data from various sources:
- Google Mobility Reports (Community Mobility)
- City traffic APIs
- Synthetic data generation for testing

TODO: Integrate with actual mobility data sources
- Google Mobility Data: https://www.google.com/covid19/mobility/
- City-specific traffic APIs (e.g., TomTom, HERE)
- Public transit APIs
"""

import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Peshawar",
    "Quetta",
]


def fetch_mobility_stub(
    city: str, hours: int = 24, start_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic mobility data (stub implementation)

    In production, replace with actual API calls to:
    - Google Mobility Reports
    - City traffic management systems
    - Mobile network data (anonymized)
    - Public transit usage data

    Args:
        city: City name
        hours: Number of hours of data to generate
        start_time: Starting timestamp (default: now)

    Returns:
        List of hourly mobility records

    Example output:
    [
        {
            "timestamp": "2025-01-01T00:00:00Z",
            "city": "Lahore",
            "mobility_index": 0.45,  # 0-1 scale, 1 = normal traffic
            "retail_and_recreation": -20,  # % change from baseline
            "grocery_and_pharmacy": -10,
            "parks": -30,
            "transit_stations": -25,
            "workplaces": -40,
            "residential": +15
        },
        ...
    ]
    """
    if start_time is None:
        start_time = datetime.utcnow()

    logger.info(f"Generating synthetic mobility data for {city} ({hours} hours)")

    mobility_data = []

    for h in range(hours):
        timestamp = start_time + timedelta(hours=h)
        hour_of_day = timestamp.hour

        # Simulate daily traffic patterns
        # Peak hours: 8-10 AM and 5-7 PM
        if 8 <= hour_of_day <= 10 or 17 <= hour_of_day <= 19:
            base_mobility = 0.8 + random.uniform(-0.1, 0.1)
        elif 0 <= hour_of_day <= 5:
            base_mobility = 0.2 + random.uniform(-0.05, 0.05)
        else:
            base_mobility = 0.6 + random.uniform(-0.1, 0.1)

        # Clamp to [0, 1]
        mobility_index = max(0.0, min(1.0, base_mobility))

        # Generate Google Mobility-style metrics (% change from baseline)
        record = {
            "timestamp": timestamp.isoformat() + "Z",
            "city": city,
            "mobility_index": round(mobility_index, 3),
            "retail_and_recreation": round(random.gauss(-15, 10), 1),
            "grocery_and_pharmacy": round(random.gauss(-5, 8), 1),
            "parks": round(random.gauss(-20, 15), 1),
            "transit_stations": round(random.gauss(-18, 12), 1),
            "workplaces": round(random.gauss(-25, 15), 1),
            "residential": round(random.gauss(10, 8), 1),
        }

        mobility_data.append(record)

    logger.info(f"Generated {len(mobility_data)} mobility records")
    return mobility_data


def fetch_traffic_congestion(city: str, hours: int = 24) -> List[Dict[str, Any]]:
    """
    Fetch traffic congestion data

    TODO: Integrate with traffic APIs
    - TomTom Traffic API
    - HERE Traffic API
    - Google Maps Traffic API
    - City-specific traffic management systems

    Args:
        city: City name
        hours: Number of hours of data

    Returns:
        List of hourly traffic congestion records
    """
    logger.info(f"Generating traffic congestion data for {city}")

    traffic_data = []
    now = datetime.utcnow()

    for h in range(hours):
        timestamp = now + timedelta(hours=h)
        hour_of_day = timestamp.hour

        # Simulate congestion levels (0 = free flow, 1 = heavy congestion)
        if 8 <= hour_of_day <= 10 or 17 <= hour_of_day <= 19:
            congestion = 0.7 + random.uniform(-0.1, 0.2)
        elif 0 <= hour_of_day <= 5:
            congestion = 0.1 + random.uniform(0, 0.1)
        else:
            congestion = 0.4 + random.uniform(-0.1, 0.1)

        congestion = max(0.0, min(1.0, congestion))

        record = {
            "timestamp": timestamp.isoformat() + "Z",
            "city": city,
            "congestion_level": round(congestion, 3),
            "avg_speed_kmh": round(60 * (1 - congestion), 1),
            "jam_length_km": round(congestion * 50, 1),
        }

        traffic_data.append(record)

    return traffic_data


def fetch_public_transit_usage(city: str, hours: int = 24) -> List[Dict[str, Any]]:
    """
    Fetch public transit usage data

    TODO: Integrate with transit APIs
    - City public transit APIs
    - Smart card data (anonymized)
    - Real-time transit occupancy

    Args:
        city: City name
        hours: Number of hours

    Returns:
        List of hourly transit usage records
    """
    logger.info(f"Generating public transit data for {city}")

    transit_data = []
    now = datetime.utcnow()

    for h in range(hours):
        timestamp = now + timedelta(hours=h)
        hour_of_day = timestamp.hour

        # Simulate ridership (passengers per hour)
        if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
            ridership = random.randint(5000, 8000)
        elif 0 <= hour_of_day <= 5:
            ridership = random.randint(100, 500)
        else:
            ridership = random.randint(2000, 4000)

        record = {
            "timestamp": timestamp.isoformat() + "Z",
            "city": city,
            "ridership": ridership,
            "occupancy_rate": round(random.uniform(0.4, 0.9), 2),
            "avg_wait_time_min": round(random.uniform(3, 15), 1),
        }

        transit_data.append(record)

    return transit_data


# TODO: Implement actual API integrations
def fetch_google_mobility_report(city: str, country: str = "PK"):
    """
    Fetch Google Community Mobility Reports

    TODO: Download and parse Google Mobility CSV files
    https://www.google.com/covid19/mobility/
    """
    pass


def fetch_tomtom_traffic(city: str, api_key: str):
    """
    Fetch TomTom Traffic Flow data

    TODO: Implement TomTom API integration
    https://developer.tomtom.com/traffic-api/documentation
    """
    pass


def write_mobility_bundle(
    city: str, hours: int = 48, out_dir: str = "./app/etl/processed_data"
) -> Dict[str, str]:
    """Generate and write mobility-related datasets for a city.

    Writes three files:
      - mobility_<city>.json (synthetic mobility indices)
      - traffic_<city>.json (synthetic congestion metrics)
      - transit_<city>.json (synthetic public transit usage)
    Returns a map of dataset name to file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    files: Dict[str, str] = {}
    
    mobility = fetch_mobility_stub(city, hours)
    traffic = fetch_traffic_congestion(city, hours)
    transit = fetch_public_transit_usage(city, hours)

    def _write(name: str, data: List[Dict[str, Any]]):
        path = os.path.join(out_dir, f"{name}_{city.lower().replace(' ', '_')}.json")
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        files[name] = path

    _write("mobility", mobility)
    _write("traffic", traffic)
    _write("transit", transit)

    logger.info(
        f"Wrote mobility bundle for {city}: "
        + ", ".join([f"{k}={v}" for k, v in files.items()])
    )
    return files


if __name__ == "__main__":
    print("=" * 60)
    print("Mobility Data Fetcher - Example Usage")
    print("=" * 60)

    # Generate synthetic mobility data
    mobility = fetch_mobility_stub("Lahore", hours=24)
    print(f"\nGenerated {len(mobility)} mobility records")
    print("\nSample record:")
    print(mobility[0])

    # Generate traffic congestion data
    traffic = fetch_traffic_congestion("Lahore", hours=24)
    print(f"\nGenerated {len(traffic)} traffic records")
    print("\nSample traffic record:")
    print(traffic[0])

    print("\n" + "=" * 60)
    print("Note: This is synthetic data for testing.")
    print("For production, integrate with actual mobility data sources.")
