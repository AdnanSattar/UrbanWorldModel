# Energy Data Sources (Pakistan)

This document lists potential energy/electricity data sources relevant to Pakistan for integrating real energy consumption, generation mix, and grid metrics.

## National/Regional Providers (Pakistan)

### NEPRA (National Electric Power Regulatory Authority)
- Website: https://nepra.org.pk/
- Data: Annual reports (generation mix, tariffs), occasional dashboards
- API: None public; may require scraping or official request
- Notes: Good for historical aggregates; not real-time

### NTDC (National Transmission & Despatch Company)
- Website: https://www.ntdc.com.pk/
- Data: System demand, generation, transmission; daily load curves (PDF)
- API: None public; may require scraping or data sharing agreement
- Notes: Useful for daily demand curves and generation by source

### CPPA-G (Central Power Purchasing Agency)
- Website: https://www.cppa.gov.pk/
- Data: Market settlement stats, generation mix snapshots
- API: None public
- Notes: Monthly/periodic summaries

### DISCOs (Distribution Companies)
- LESCO, IESCO, KE, etc.
- Data: Service-level outages, demand updates (often via web/news)
- API: None public; may provide feeds via partnership

## International/Public APIs

### EIA (U.S. Energy Information Administration)
- API: https://www.eia.gov/opendata/ (JSON)
- Data: International energy statistics; limited Pakistan coverage
- Example: `series?id=INTL.44-1-PAK-MK.A` (example IDs vary)
- Pros: Free, documented API; global coverage
- Cons: Mostly historical/annual; limited granularity

### ENTSO-E Transparency (EU)
- API: https://transparency.entsoe.eu/
- Data: EU region only (not Pakistan)
- Notes: Use as reference for data model/ideas (not a source for PK)

### Open Power System Data (OPSD)
- Website: https://open-power-system-data.org/
- Data: Cleaned datasets (Europe-centric)
- Notes: Useful for methodology; not Pakistan

### OpenWeatherMap Energy (indirect)
- API: https://openweathermap.org/api (weather/solar/wind)
- Use: Derive generation proxies from weather (solar irradiance, wind speed)
- Pros: Global, free tiers
- Cons: Indirect; needs modeling

## Recommended Strategy

- Near-term: Use synthetic data (current), augment with publicly available reports (NTDC/NEPRA) for realistic ranges.
- Medium-term: Scrape NTDC daily load/generation PDFs to build daily curves.
- Long-term: Establish data-sharing with NTDC/CPPA-G or DISCOs for programmatic feeds.

## Example Integration Sketch (EIA)

```python
# backend/app/etl/energy_eia.py
import os, requests

EIA_API_KEY = os.getenv("EIA_API_KEY", "")
BASE = "https://api.eia.gov/series/"

def fetch_series(series_id: str):
    params = {"api_key": EIA_API_KEY, "series_id": series_id}
    r = requests.get(BASE, params=params, timeout=15)
    r.raise_for_status()
    return r.json()
```

## Notes
- Store energy outputs under `PROCESSED_DATA_DIR` alongside air quality.
- Include fields such as `energy_mwh`, `renewable_share`, `timestamp`.


