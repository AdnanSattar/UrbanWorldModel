# UrbanSim WM - Quick Start Guide

Get up and running in 5 minutes!

---

## Prerequisites

Ensure you have installed:

- **Docker** (v20.10+)
- **Docker Compose** (v2.0+)
- **Make** (optional, but recommended)

Check versions:

```bash
docker --version
docker-compose --version
make --version
```

---

## Step 1: Clone and Setup

```bash
# Clone the repository
cd urbansim-wm

# Copy environment template
cp .env.example .env

# (Recommended) Edit .env to add API keys and paths
# Key variables:
# - WAQI_API_TOKEN=                 # optional for real air quality
# - LOGS_DIR=./logs                 # mounted in Docker for logs
# - PROCESSED_DATA_DIR=./etl/processed_data  # mounted for ETL outputs
# - REDIS_URL=redis://redis:6379/0  # optional caching
# nano .env
```

---

## Step 2: Build and Start

### Using Make (Recommended)

```bash
# Build all containers
make build

# Start services
make up

# View logs
make logs
```

### Using Docker Compose Directly

```bash
# Build
docker-compose build

# Start
docker-compose up -d

# View logs (backend)
docker-compose logs -f backend
```

---

## Step 3: Access the Application

Once the services are running:

- **🎨 Frontend Dashboard**: <http://localhost:3000>
- **🔌 Backend API**: <http://localhost:8000>
- **📚 API Documentation**: <http://localhost:8000/docs>
- **❤️ Health Check**: <http://localhost:8000/>

Notes:

- The frontend uses relative API paths (`/api/...`) via Next.js rewrites for Docker. No manual base URL needed.
- On first startup, the backend ETL runs in the background. Check logs under `LOGS_DIR` (default `./logs`).
- ETL outputs are written to `PROCESSED_DATA_DIR` (default `./etl/processed_data`). Ensure volumes are mounted in `docker-compose.yml`.

---

## Step 4: Run Your First Simulation

### Via Frontend UI

1. Open <http://localhost:3000>
2. Use the sliders to adjust:
   - **Car-Free Ratio** (0-100%)
   - **Renewable Energy Mix** (0-100%)
3. Click **"Run Simulation"**
4. View the results in interactive charts!

### Via API (curl)

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Lahore",
    "start_time": "2025-01-01T00:00:00Z",
    "horizon_hours": 48,
    "policy": {
      "car_free_ratio": 0.2,
      "renewable_mix": 0.35
    }
  }'
```

### Via API (Python)

```python
import requests

response = requests.post(
    "http://localhost:8000/api/simulate",
    json={
        "city": "Lahore",
        "start_time": "2025-01-01T00:00:00Z",
        "horizon_hours": 48,
        "policy": {
            "car_free_ratio": 0.2,
            "renewable_mix": 0.35
        }
    }
)

print(response.json())
```

---

## Optional: Enable Real Air Quality (WAQI)

1. Get a WAQI token and set `WAQI_API_TOKEN` in `.env`.
2. Ensure volumes are mounted:

```yaml
# docker-compose.yml (snippet)
  backend:
    environment:
      - WAQI_API_TOKEN=${WAQI_API_TOKEN}
      - LOGS_DIR=/logs
      - PROCESSED_DATA_DIR=/etl/processed_data
    volumes:
      - ./logs:/logs
      - ./etl/processed_data:/etl/processed_data
```

Outputs include `real_time_pm25`, `pm25_values`, min/max, and forecast. If WAQI fails, the system falls back gracefully (source becomes `baseline`).

---

## Troubleshooting

- Backend stuck on startup: it offloads ETL to a background thread. Check `/logs/bootstrap_*.log` under `LOGS_DIR`.
- No ETL outputs: verify `PROCESSED_DATA_DIR` volume is mounted and writable.
- Frontend API errors in Docker: ensure relative API paths are used and the backend service is named `backend`.
- WAQI DNS errors inside Docker: retry or check Docker DNS/network configuration.

---

## Step 5: Train the Model (Optional)

```bash
# Using Make
make train

# Or with docker-compose
docker-compose --profile training run --rm training
```

This will run the training harness (currently a stub).

---

## Common Commands

```bash
# View all available commands
make help

# Stop all services
make down

# Restart services
make restart

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Open shell in backend container
make shell-be

# Clean up everything
make clean
```

---

## Troubleshooting

### Port Already in Use

If ports 3000 or 8000 are already in use:

```bash
# Find and kill processes using the ports
# On Linux/Mac:
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9

# On Windows (PowerShell):
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

### Docker Build Fails

```bash
# Clean rebuild
make clean
make build
```

### Frontend Can't Connect to Backend

1. Check if backend is running:

   ```bash
   curl http://localhost:8000/
   ```

2. Check logs:

   ```bash
   docker-compose logs backend
   ```

3. Verify environment variables in `.env`:

   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

### Container Health Check Failing

```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs backend

# Restart services
make restart
```

---

## Next Steps

1. **Explore the API**: Visit <http://localhost:8000/docs> for interactive API documentation
2. **Customize Policies**: Edit `frontend/components/PolicyControls.tsx` to add new policy parameters
3. **Add Real Data**: Integrate actual data sources in `etl/` directory
4. **Train the Model**: Implement real training logic in `training/urban_world_model.py`
5. **Add New Charts**: Create custom visualizations in `frontend/components/`

---

## Example Scenarios to Try

### Baseline Scenario

- Car-Free Ratio: 0%
- Renewable Mix: 0%

### Moderate Intervention

- Car-Free Ratio: 20%
- Renewable Mix: 35%

### Aggressive Green Policy

- Car-Free Ratio: 40%
- Renewable Mix: 70%

### Maximum Impact

- Car-Free Ratio: 60%
- Renewable Mix: 90%

Compare the results across different scenarios!

---

## Development Mode

For local development without Docker:

```bash
# Install dependencies
make install

# Backend (Terminal 1)
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (Terminal 2)
cd frontend
npm run dev

# The frontend will be at http://localhost:3000
# The backend will be at http://localhost:8000
```

---

## Getting Help

- **Documentation**: See [README.md](README.md) for full documentation
- **API Reference**: <http://localhost:8000/docs>

---
