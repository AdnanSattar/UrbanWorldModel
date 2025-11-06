# Complete Model Inference Flow: ETL → Model

## Quick Answer to Your Question

**What does the model use?**

- ✅ **NOW**: Uses `real_time_pm25` (current measurement) - **BEST**
- ❌ **BEFORE**: Used `mean_pm25` (included old forecast data) - **WRONG**

**Why this matters:**

- Model needs current conditions to predict future accurately
- `mean_pm25` (136.89) mixed real-time (34) with old forecast averages (150+)
- Now using `real_time_pm25` (34) gives accurate starting point

## Complete Flow Diagram

Notes:

- WAQI is the primary air quality source; OpenAQ is used as fallback.
- Paths are configurable via env: `LOGS_DIR` (default `./logs`), `PROCESSED_DATA_DIR` (default `./etl/processed_data`).

```bash
┌─────────────────────────────────────────────────────────────┐
│ 1. BACKEND STARTUP (bootstrap_*.log)                        │
│    backend/app/main.py                                       │
│    └─> ETL runs for each city                               │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ETL PROCESS (backend/app/etl/waqi.py)                   │
│    write_recent_pm25_mean(city)                             │
│    ├─> Fetch from WAQI API                                  │
│    │   └─> data.iaqi.pm25.v = 34 (real-time)               │
│    │   └─> data.time.iso = "2025-02-18T18:00:00"           │
│    │   └─> data.forecast.daily.pm25 = [...] (old)          │
│    ├─> Extract: real_time_pm25 = 34                        │
│    ├─> Extract: forecast values = [150, 159, ...]          │
│    ├─> Calculate: mean_pm25 = 136.89 (all values)          │
│    └─> Write JSON to etl/processed_data/pm25_lahore.json   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. JSON OUTPUT (pm25_lahore.json)                           │
│    {                                                         │
│      "real_time_pm25": 34.0,        ← CURRENT (used now!)  │
│      "mean_pm25": 136.89,           ← OLD (was used)       │
│      "historical_mean_pm25": 140.70,                       │
│      "pm25_values": [34, 150, ...],                        │
│      "measurement_timestamp": "2025-02-18T18:00:00+05:00", │
│      "forecast": {...}                                      │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. API REQUEST (POST /api/simulate)                        │
│    {                                                         │
│      "city": "Lahore",                                       │
│      "horizon_hours": 48,                                    │
│      "policy": {...}                                         │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DATA LOADER (backend/app/data/loader.py)                 │
│    load_recent_pm25_mean("Lahore")                          │
│    ├─> Read pm25_lahore.json                                │
│    ├─> Check freshness (measurement_timestamp)              │
│    ├─> PRIORITY 1: real_time_pm25 = 34.0 ✅                │
│    └─> Return: 34.0                                         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. MODEL INITIALIZATION (backend/app/api/simulate.py)       │
│    observed_pm25 = load_recent_pm25_mean(city) or 85.0     │
│    initial_state = {                                         │
│      "pm25": 34.0 * (0.98 + 0.02 * gdp_idx),  ← Uses 34!   │
│      "energy_mwh": 1200.0,                                   │
│      "traffic_index": 1.0                                    │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. MODEL INFERENCE (backend/app/models/world_model.py)      │
│    model.predict(initial_state, policy, horizon)           │
│    ├─> Encode: [pm25=34, energy=1200, traffic=1] → latent  │
│    ├─> RSSM Rollout: Predict future latent states          │
│    ├─> Predictor: Decode latent → [pm25, energy, traffic]  │
│    └─> Return: (horizon, 3) predictions                    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. RESPONSE (SimulationResponse)                           │
│    {                                                         │
│      "simulated": [                                          │
│        {"hour": 0, "pm25": 35, "energy": 1180, ...},       │
│        {"hour": 1, "pm25": 36, "energy": 1160, ...},       │
│        ...                                                   │
│      ]                                                       │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
```

## What Changed

### Before (WRONG)

```python
# loader.py
mean = data.get("mean_pm25")  # 136.89 (includes old forecast)
return float(mean)

# Result: Model starts with 136.89 instead of current 34
```

### After (CORRECT)

```python
# loader.py
real_time = data.get("real_time_pm25")  # 34.0 (current)
if real_time is not None:
    return float(real_time)  # ✅ Use current value
    
# Result: Model starts with 34 (accurate current conditions)
```

## Data Freshness Issue

**Problem:** WAQI forecast dates show "2025-02-16" (February)

- If current date is November 2025, this is **9 months old**
- Forecast data is historical averages, not real-time

**Solution:**

1. ✅ Use `real_time_pm25` (34) - this is current
2. ✅ Store `measurement_timestamp` from WAQI response
3. ✅ Check freshness in loader (warn if > 24 hours old)
4. ✅ Document alternative data sources (IQAir, Google, etc.)

## Values Available (But Not All Used)

```json
{
  "real_time_pm25": 34.0,        ← ✅ USED (current measurement)
  "mean_pm25": 136.89,           ← ❌ NOT USED (mixed old/new)
  "historical_mean_pm25": 140.70, ← ❌ NOT USED (old forecast avg)
  "pm25_values": [34, 150, ...], ← ❌ NOT USED (all values)
  "pm25_min": 34.0,              ← ❌ NOT USED (min of all)
  "pm25_max": 174.0,             ← ❌ NOT USED (max of all)
  "forecast": {...}              ← ❌ NOT USED (structured forecast)
}
```

**Why only `real_time_pm25`?**

- Model needs a single starting value
- Current conditions are most important for future prediction
- Historical averages would bias the model

## Alternative Data Sources

See `docs/AIR_QUALITY_DATA_SOURCES.md` for:

- IQAir AirVisual API (recommended)
- Google Air Quality API
- Ambee API
- OpenWeatherMap API

**Recommendation:** Use IQAir as primary, keep WAQI as backup.

## Testing the Flow

1. **Check ETL output:**

   ```bash
   cat etl/processed_data/pm25_lahore.json | jq '.real_time_pm25'
   # Should show: 34.0
   ```

2. **Check loader:**

   ```python
   from app.data.loader import load_recent_pm25_mean
   value = load_recent_pm25_mean("Lahore")
   print(value)  # Should show: 34.0
   ```

3. **Check API:**

   ```bash
   curl -X POST http://localhost:8000/api/simulate \
     -H "Content-Type: application/json" \
     -d '{"city": "Lahore", "horizon_hours": 48, "policy": {...}}'
   # Check logs for "observed_pm25 = 34.0"
   ```

## Summary

✅ **Model now uses `real_time_pm25` (34)** instead of `mean_pm25` (136.89)
✅ **Data freshness check** added (warns if > 24 hours old)
✅ **Timestamp extraction** from WAQI response
✅ **Alternative sources documented** for future migration

**Next steps:**

1. Monitor data freshness in logs
2. Consider IQAir API if WAQI data stays stale
3. Optionally use forecast data for trend analysis (future enhancement)
