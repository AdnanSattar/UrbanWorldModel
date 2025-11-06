# Mobility Data Sources (Pakistan)

Potential sources for traffic/mobility indices, congestion, and transit usage for Pakistan.

## National/Local Sources

### Punjab Safe Cities Authority (PSCA)
- Website: https://psca.gop.pk/
- Data: Traffic cameras, congestion monitoring (Lahore)
- API: None public; partnership may be required
- Notes: Strong candidate for Lahore-specific congestion data

### City Traffic Police (Lahore/Karachi/Islamabad)
- Websites/Social media: Incident and congestion updates
- API: None public; may need scraping/feeds
- Notes: Event/incident integration for spikes

## Commercial APIs

### TomTom Traffic API
- Docs: https://developer.tomtom.com/traffic-api/documentation
- Data: Live incident flow, travel times, congestion indices
- Endpoints: `Traffic Flow`, `Traffic Incidents`
- Coverage: Global (includes Pakistan urban areas)
- Pricing: Free tier + paid
- Pros: Mature, detailed; map-matched

### HERE Traffic API
- Docs: https://developer.here.com/documentation/traffic/dev_guide/index.html
- Data: Live flow/speeds, incidents
- Coverage: Global; Pakistan urban corridors
- Pricing: Free tier + paid

### Google Maps Traffic (Indirect)
- Docs: https://developers.google.com/maps/documentation
- Data: Directions/travel time; traffic layer indirectly
- Notes: No direct traffic index API; can infer via travel times

### INRIX (Enterprise)
- Data: Advanced traffic analytics
- Pricing: Enterprise
- Notes: Likely overkill for MVP

## Open/Public Proxies

### Apple Mobility Trends (Deprecated)
- Historical CSV (no longer updated)

### Google COVID-19 Mobility Reports (Archived)
- Historical CSV by region; good for baseline behavior

### OpenStreetMap (OSM)
- Use network and metadata to simulate load by time
- Combine with events/weather for synthetic indices

## Recommended Strategy

- Near-term: Continue with realistic synthetic generation; calibrate ranges using Google Mobility Reports historical CSVs.
- Medium-term: Integrate TomTom or HERE for Lahore/Karachi corridors; sample flow tiles or key routes hourly.
- Long-term: Establish local partnerships (PSCA) for city-level indices.

## Example Integration Sketch (TomTom Flow)

```bash
# Example request (tile-based)
# https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/10/json?point=31.5497,74.3436&unit=KMPH&key=YOUR_TOMTOM_KEY
```

```python
# backend/app/etl/mobility_tomtom.py
import os, requests

TOMTOM_KEY = os.getenv("TOMTOM_KEY", "")
BASE = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/10/json"

def fetch_flow(lat: float, lon: float):
    params = {"point": f"{lat},{lon}", "unit": "KMPH", "key": TOMTOM_KEY}
    r = requests.get(BASE, params=params, timeout=10)
    r.raise_for_status()
    return r.json()
```

## Notes
- Store mobility outputs under `PROCESSED_DATA_DIR`.
- Include fields such as `traffic_index` (0–2), `avg_speed_kmph`, `timestamp`.


