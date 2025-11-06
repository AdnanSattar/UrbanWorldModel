"""
Energy ETL (backend-local copy)
Generates synthetic energy bundles per city.
"""

import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_energy_stub(
    city: str, hours: int = 24, start_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    if start_time is None:
        start_time = datetime.utcnow()
    base_consumption = 1200.0
    out: List[Dict[str, Any]] = []
    for h in range(hours):
        ts = start_time + timedelta(hours=h)
        hod = ts.hour
        if 18 <= hod <= 22:
            demand = 1.3 + random.uniform(-0.05, 0.05)
        elif 0 <= hod <= 5:
            demand = 0.7 + random.uniform(-0.05, 0.05)
        else:
            demand = 1.0 + random.uniform(-0.1, 0.1)
        total = base_consumption * demand
        if 8 <= hod <= 16:
            solar_factor = 1.0 - abs(hod - 12) / 8
            solar = 100 * solar_factor + random.uniform(-10, 10)
        else:
            solar = random.uniform(0, 5)
        wind = 50 + random.uniform(-20, 30)
        renewable = max(0.0, solar + wind)
        fossil = total - renewable
        renewable_pct = (renewable / total) * 100 if total > 0 else 0.0
        carbon_intensity = 700 * (fossil / total) if total > 0 else 0.0
        out.append(
            {
                "timestamp": ts.isoformat() + "Z",
                "city": city,
                "total_consumption_mwh": round(total, 2),
                "renewable_generation_mwh": round(renewable, 2),
                "fossil_generation_mwh": round(fossil, 2),
                "renewable_percentage": round(renewable_pct, 2),
                "grid_frequency_hz": round(50 + random.uniform(-0.05, 0.05), 2),
                "carbon_intensity_gco2_kwh": round(carbon_intensity, 1),
            }
        )
    return out


def fetch_renewable_generation(city: str, hours: int = 24) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now = datetime.utcnow()
    for h in range(hours):
        ts = now + timedelta(hours=h)
        hod = ts.hour
        solar = 200 * max(0.0, 1.0 - abs(hod - 12) / 6) if 6 <= hod <= 18 else 0.0
        wind = 150 + random.uniform(-50, 50)
        hydro = 80 + random.uniform(-10, 10)
        out.append(
            {
                "timestamp": ts.isoformat() + "Z",
                "city": city,
                "solar_mw": round(max(0.0, solar), 2),
                "wind_mw": round(max(0.0, wind), 2),
                "hydro_mw": round(max(0.0, hydro), 2),
                "total_renewable_mw": round(max(0.0, solar + wind + hydro), 2),
            }
        )
    return out


def fetch_carbon_intensity(city: str, hours: int = 24) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now = datetime.utcnow()
    for h in range(hours):
        ts = now + timedelta(hours=h)
        hod = ts.hour
        if 10 <= hod <= 16:
            intensity = random.uniform(400, 550)
        else:
            intensity = random.uniform(600, 750)
        out.append(
            {
                "timestamp": ts.isoformat() + "Z",
                "city": city,
                "carbon_intensity_gco2_kwh": round(intensity, 1),
                "marginal_intensity": round(intensity * 1.1, 1),
            }
        )
    return out


def write_energy_bundle(
    city: str, hours: int = 48, out_dir: str = "./app/etl/processed_data"
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    files: Dict[str, str] = {}
    energy = fetch_energy_stub(city, hours)
    renewable = fetch_renewable_generation(city, hours)
    carbon = fetch_carbon_intensity(city, hours)

    def _write(name: str, data: List[Dict[str, Any]]):
        path = os.path.join(out_dir, f"{name}_{city.lower().replace(' ', '_')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        files[name] = path

    _write("energy", energy)
    _write("renewable", renewable)
    _write("carbon", carbon)
    return files


"""
Energy Data Fetcher

Fetches energy consumption and generation data from various sources:
- Grid operators / utilities
- Open energy datasets
- Renewable energy generation data

TODO: Integrate with actual energy data sources
- National/regional grid operators
- Smart meter data (aggregated)
- Renewable energy tracking systems
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


def fetch_energy_stub(
    city: str, hours: int = 24, start_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic energy consumption data (stub implementation)

    In production, replace with actual API calls to:
    - Grid operators (e.g., LESCO for Lahore)
    - Energy Information Administration (EIA)
    - European Network of Transmission System Operators (ENTSO-E)
    - Smart grid data providers

    Args:
        city: City name
        hours: Number of hours of data to generate
        start_time: Starting timestamp (default: now)

    Returns:
        List of hourly energy consumption records

    Example output:
    [
        {
            "timestamp": "2025-01-01T00:00:00Z",
            "city": "Lahore",
            "total_consumption_mwh": 1200.5,
            "renewable_generation_mwh": 150.3,
            "fossil_generation_mwh": 1050.2,
            "renewable_percentage": 12.5,
            "grid_frequency_hz": 50.02,
            "carbon_intensity_gco2_kwh": 650
        },
        ...
    ]
    """
    if start_time is None:
        start_time = datetime.utcnow()

    logger.info(f"Generating synthetic energy data for {city} ({hours} hours)")

    energy_data = []

    # Baseline consumption (MWh)
    base_consumption = 1200

    for h in range(hours):
        timestamp = start_time + timedelta(hours=h)
        hour_of_day = timestamp.hour

        # Simulate daily energy demand patterns
        # Peak hours: 6-10 PM (residential + commercial)
        if 18 <= hour_of_day <= 22:
            demand_factor = 1.3 + random.uniform(-0.05, 0.05)
        elif 0 <= hour_of_day <= 5:
            demand_factor = 0.7 + random.uniform(-0.05, 0.05)
        else:
            demand_factor = 1.0 + random.uniform(-0.1, 0.1)

        total_consumption = base_consumption * demand_factor

        # Simulate renewable generation (solar + wind)
        # Solar: peaks during day (8 AM - 4 PM)
        if 8 <= hour_of_day <= 16:
            solar_factor = 1.0 - abs(hour_of_day - 12) / 8  # Peak at noon
            solar_generation = 100 * solar_factor + random.uniform(-10, 10)
        else:
            solar_generation = random.uniform(0, 5)  # Minimal at night

        # Wind: variable throughout day
        wind_generation = 50 + random.uniform(-20, 30)

        renewable_generation = max(0, solar_generation + wind_generation)
        fossil_generation = total_consumption - renewable_generation

        renewable_percentage = (renewable_generation / total_consumption) * 100

        # Carbon intensity (g CO2/kWh) - higher when more fossil fuels used
        carbon_intensity = 700 * (fossil_generation / total_consumption)

        record = {
            "timestamp": timestamp.isoformat() + "Z",
            "city": city,
            "total_consumption_mwh": round(total_consumption, 2),
            "renewable_generation_mwh": round(renewable_generation, 2),
            "fossil_generation_mwh": round(fossil_generation, 2),
            "renewable_percentage": round(renewable_percentage, 2),
            "grid_frequency_hz": round(50 + random.uniform(-0.05, 0.05), 2),
            "carbon_intensity_gco2_kwh": round(carbon_intensity, 1),
        }

        energy_data.append(record)

    logger.info(f"Generated {len(energy_data)} energy records")
    return energy_data


def fetch_renewable_generation(
    city: str, hours: int = 24, source_type: str = "all"
) -> List[Dict[str, Any]]:
    """
    Fetch renewable energy generation data by source

    TODO: Integrate with renewable energy tracking APIs
    - Solar farm monitoring systems
    - Wind farm data
    - Hydroelectric generation
    - Battery storage systems

    Args:
        city: City name
        hours: Number of hours
        source_type: 'solar', 'wind', 'hydro', or 'all'

    Returns:
        List of hourly renewable generation records
    """
    logger.info(f"Generating renewable energy data for {city}")

    renewable_data = []
    now = datetime.utcnow()

    for h in range(hours):
        timestamp = now + timedelta(hours=h)
        hour_of_day = timestamp.hour

        # Solar generation pattern
        if 6 <= hour_of_day <= 18:
            solar_capacity_factor = max(0, 1.0 - abs(hour_of_day - 12) / 6)
            solar_mw = 200 * solar_capacity_factor + random.uniform(-10, 10)
        else:
            solar_mw = 0

        # Wind generation (more variable)
        wind_mw = 150 + random.uniform(-50, 50)

        # Hydro (relatively stable)
        hydro_mw = 80 + random.uniform(-10, 10)

        record = {
            "timestamp": timestamp.isoformat() + "Z",
            "city": city,
            "solar_mw": round(max(0, solar_mw), 2),
            "wind_mw": round(max(0, wind_mw), 2),
            "hydro_mw": round(max(0, hydro_mw), 2),
            "total_renewable_mw": round(max(0, solar_mw + wind_mw + hydro_mw), 2),
        }

        renewable_data.append(record)

    return renewable_data


def fetch_grid_status(city: str) -> Dict[str, Any]:
    """
    Fetch current grid status and metrics

    TODO: Integrate with grid operator APIs

    Args:
        city: City name

    Returns:
        Current grid status
    """
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "city": city,
        "frequency_hz": round(50 + random.uniform(-0.1, 0.1), 3),
        "voltage_kv": round(220 + random.uniform(-5, 5), 1),
        "load_mw": round(1200 + random.uniform(-100, 100), 1),
        "available_capacity_mw": round(1500 + random.uniform(-50, 50), 1),
        "reserve_margin_percent": round(random.uniform(15, 25), 1),
        "status": "normal",  # or "alert", "emergency"
    }


def fetch_carbon_intensity(city: str, hours: int = 24) -> List[Dict[str, Any]]:
    """
    Fetch grid carbon intensity over time

    Carbon intensity varies based on generation mix.
    Lower when more renewables are online.

    Args:
        city: City name
        hours: Number of hours

    Returns:
        List of hourly carbon intensity records
    """
    carbon_data = []
    now = datetime.utcnow()

    for h in range(hours):
        timestamp = now + timedelta(hours=h)
        hour_of_day = timestamp.hour

        # Lower intensity during day (more solar)
        if 10 <= hour_of_day <= 16:
            intensity = random.uniform(400, 550)
        else:
            intensity = random.uniform(600, 750)

        record = {
            "timestamp": timestamp.isoformat() + "Z",
            "city": city,
            "carbon_intensity_gco2_kwh": round(intensity, 1),
            "marginal_intensity": round(intensity * 1.1, 1),  # Marginal often higher
        }

        carbon_data.append(record)

    return carbon_data


# TODO: Implement actual API integrations
def fetch_eia_data(region: str):
    """
    Fetch US Energy Information Administration data

    TODO: Implement EIA API integration
    https://www.eia.gov/opendata/
    """
    pass


def fetch_entso_e_data(country: str):
    """
    Fetch European grid data

    TODO: Implement ENTSO-E API integration
    https://transparency.entsoe.eu/
    """
    pass


def write_energy_bundle(
    city: str, hours: int = 48, out_dir: str = "./app/etl/processed_data"
) -> Dict[str, str]:
    """Generate and write energy-related datasets for a city.

    Writes three files:
      - energy_<city>.json (synthetic consumption/gen)
      - renewable_<city>.json (synthetic split by source)
      - carbon_<city>.json (synthetic carbon intensity)
    Returns a map of dataset name to file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    files: Dict[str, str] = {}

    energy = fetch_energy_stub(city, hours)
    renewable = fetch_renewable_generation(city, hours)
    carbon = fetch_carbon_intensity(city, hours)

    import json

    def _write(name: str, data: List[Dict[str, Any]]):
        path = os.path.join(out_dir, f"{name}_{city.lower().replace(' ', '_')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        files[name] = path

    _write("energy", energy)
    _write("renewable", renewable)
    _write("carbon", carbon)

    logger.info(
        f"Wrote energy bundle for {city}: "
        + ", ".join([f"{k}={v}" for k, v in files.items()])
    )
    return files


if __name__ == "__main__":
    print("=" * 60)
    print("Energy Data Fetcher - Example Usage")
    print("=" * 60)

    # Generate synthetic energy data
    energy = fetch_energy_stub("Lahore", hours=24)
    print(f"\nGenerated {len(energy)} energy records")
    print("\nSample record:")
    print(energy[0])

    # Generate renewable generation data
    renewable = fetch_renewable_generation("Lahore", hours=24)
    print(f"\nGenerated {len(renewable)} renewable records")
    print("\nSample renewable record:")
    print(renewable[0])

    # Get current grid status
    status = fetch_grid_status("Lahore")
    print("\nCurrent grid status:")
    print(status)

    print("\n" + "=" * 60)
    print("Note: This is synthetic data for testing.")
    print("For production, integrate with actual energy data providers.")
