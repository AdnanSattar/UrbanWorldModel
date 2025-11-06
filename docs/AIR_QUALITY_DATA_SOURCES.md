# Air Quality Data Sources

## Overview

Primary and alternative air quality data sources used or considered for UrbanSim WM. WAQI is the primary live source; OpenAQ is a fallback. Additional providers are listed for future upgrades.

## Current Integration

### WAQI / AQICN (Primary)
- Used in production via `backend/app/etl/waqi.py`
- Extracts real-time PM2.5 (`iaqi.pm25.v`), forecast, min/max; writes to `PROCESSED_DATA_DIR`
- Requires `WAQI_API_TOKEN`

### OpenAQ v3 (Fallback)
- Implemented in `backend/app/etl/openaq.py`
- Robust mean across sensors with backoff; used when WAQI fails

## Recommended Free/Low-Cost APIs

### 1. **World Air Quality Index (WAQI/AQICN)** ⭐ RECOMMENDED

- **URL**: `https://api.waqi.info/`
- **Free**: Yes (with rate limits)
- **Coverage**: 10,000+ stations worldwide, including Pakistan
- **Endpoint**: `GET /feed/{city}/`
- **Example**: `https://api.waqi.info/feed/lahore/?token=YOUR_TOKEN`
- **Data**: Real-time AQI, PM2.5, PM10, NO2, O3, etc.
- **Token**: Free token available at <https://aqicn.org/api/>
- **Rate Limit**: ~1000 requests/day
- **Pros**: Free, easy to use, good Pakistan coverage
- **Cons**: Requires free token registration

### 2. **IQAir AirVisual API**

- **URL**: `https://api.airvisual.com/v2/`
- **Free**: Community tier (limited calls)
- **Coverage**: 10,000+ cities worldwide
- **Endpoint**: `GET /city?city={city}&state={state}&country={country}`
- **Example**: `https://api.airvisual.com/v2/city?city=Lahore&state=Punjab&country=Pakistan&key=YOUR_KEY`
- **Data**: Real-time AQI, PM2.5, PM10, temperature, humidity
- **Key**: Free tier available at <https://www.iqair.com/us/air-quality-monitors/api>
- **Rate Limit**: 10,000 calls/month (free tier)
- **Pros**: Good documentation, reliable
- **Cons**: Requires API key, state/country needed

### 3. **Google Air Quality API**

- **URL**: `https://airquality.googleapis.com/v1/`
- **Free**: Trial (requires Google Cloud account)
- **Coverage**: 100+ countries, 500m resolution
- **Endpoint**: REST API with coordinates or place IDs
- **Data**: Real-time, forecast, historical, multiple pollutants
- **Key**: Requires Google Cloud API key
- **Rate Limit**: Based on billing/quota
- **Pros**: High resolution, comprehensive
- **Cons**: Requires Google Cloud setup, may have costs

### 4. **Airly API**

- **URL**: `https://airapi.airly.eu/v2/`
- **Free**: Trial available
- **Coverage**: 13,000+ sensors worldwide
- **Endpoint**: `GET /measurements/point?lat={lat}&lng={lng}`
- **Data**: PM1, PM2.5, PM10, NO2, O3, CO, SO2, weather
- **Key**: Trial key available
- **Pros**: Detailed data, weather included
- **Cons**: Requires coordinates, may need paid plan

### 5. **PurpleAir API**

### 6. **OpenWeatherMap Air Pollution API**
- **URL**: `https://openweathermap.org/api/air-pollution`
- **Free**: Limited free tier; paid plans
- **Coverage**: Global; includes weather context
- **Endpoint**: `GET /air_pollution?lat={}&lon={}&appid={}`
- **Pros/Cons**: Established API; key required; rate limits

### 7. **Local Pakistan EPA Stations**
- Government sources; coverage varies; may need scraping/custom integration

## Implementation Recommendation

- **URL**: `https://api.purpleair.com/v1/`
- **Free**: Yes (public data)
- **Coverage**: 10,000+ sensors globally (community-driven)
- **Endpoint**: `GET /sensors?fields=pm2.5`
- **Data**: Real-time PM2.5, temperature, humidity
- **Key**: Free API key available
- **Pros**: Community-driven, real-time
- **Cons**: Sensor-based (may not have all cities)

## Implementation Recommendation

**Primary Choice: WAQI/AQICN API**

- ✅ Free token (easy registration)
- ✅ Good Pakistan city coverage
- ✅ Simple REST API
- ✅ No coordinates needed (city name works)
- ✅ Real-time PM2.5 data

**Backup Choice: IQAir AirVisual**

- ✅ Reliable service
- ✅ Good documentation
- ✅ Requires state/country (but we know these)

## Next Steps

1. Register for WAQI free token at <https://aqicn.org/api/>
2. Keep OpenAQ as fallback; add IQAir as optional primary when keys available
3. Maintain baseline fallback for resilience

## Code Structure

```python
# backend/app/etl/waqi.py
WAQI_BASE_URL = "https://api.waqi.info"
WAQI_TOKEN = os.getenv("WAQI_API_TOKEN", "")

def fetch_city_pm25(city: str) -> Optional[float]:
    """Fetch PM2.5 from WAQI API"""
    url = f"{WAQI_BASE_URL}/feed/{city.lower()}/"
    params = {"token": WAQI_TOKEN}
    # ... implementation
```
