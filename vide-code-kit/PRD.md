# UrbanSim WM — Product Requirements Document (PRD)

## Overview

UrbanSim WM is a simulation platform that models city-scale air quality, energy, and mobility dynamics using world models (DreamerV3-style latent dynamics). It enables urban policy makers and researchers to simulate the impact of interventions such as car-free days or renewable energy expansion.

---

## Objectives

- Predict air pollution (PM2.5), energy usage, and traffic patterns.
- Enable policy simulation through world models.
- Support retraining from real-time data streams (OpenAQ, energy grids, mobility feeds).

---

## Key Components

| Layer | Tech | Purpose |
|-------|------|----------|
| Frontend | Next.js 14 (App Router) + Tailwind + Recharts | Interactive simulator UI |
| Backend | FastAPI | Simulation API and retraining trigger |
| Model | PyTorch (Dreamer-style) | Latent world model for prediction |
| ETL | Python scripts | Fetch and preprocess city data |
| Infra | Docker Compose + Makefile | Multi-service orchestration |

## 1) Problem Statement

Urban stakeholders need to quickly test the impact of policies (car‑free days, renewable adoption) on air quality, energy demand, and mobility. Current tools are slow, siloed, or require expert knowledge.

## 2) Objectives & KPIs

- Deliver an interactive simulator for “what‑if” policy analyses.
- KPIs
  - P95 API latency (/api/simulate, 48h horizon): ≤ 800 ms (stub), ≤ 2 s (model)
  - Time‑to‑first‑simulation on a fresh boot: ≤ 10 s (ETL boot runs in background)
  - ETL freshness: PM2.5 real‑time timestamp not older than 24h
  - Dashboard uptime: ≥ 99% (prod)
  - Model accuracy (phase 2): MAE vs baselines improves ≥ 10%

## 3) Personas

- Policy Analyst: explores scenarios, exports results.
- Energy Planner: tests renewable mixes and carbon intensity.
- Transport Planner: evaluates car‑free and congestion policies.
- Data Scientist/ML: trains/improves world model, integrates data.

## 4) Scope

### In‑Scope (MVP)

- Frontend (Next.js 14 App Router) at `/` with two sliders: car‑free ratio, renewable mix.
- Mobile‑responsive Tailwind UI; relative API paths (`/api/...`) via Next.js rewrites.
- Charts (Recharts): PM2.5, energy (MWh), traffic index.
- Backend (FastAPI): POST `/api/simulate`, POST `/api/retrain`, GET `/api/health`, GET `/`.
- Training service (PyTorch) with DreamerV3‑style loop (encoder, RSSM, predictor, reward head, checkpoints, TensorBoard).
- ETL for air quality: WAQI (primary) with OpenAQ fallback; logs and processed outputs persisted via Docker volumes.
- Dockerized stack + Makefile automation.

### Out‑of‑Scope (MVP)

- AuthN/Z, multi‑tenant roles, RBAC.
- Real‑time streaming or websockets.
- GIS maps and spatial breakdowns.
- RL policy optimization.

## 5) Functional Requirements

### Frontend

- Sliders for car‑free ratio (0–1) and renewable mix (0–1) with presets.
- Run simulation button; loading and error states; raw JSON toggle.
- Three charts with summary stats and tooltips.

### Backend

- POST `/api/simulate`: returns hourly array of `{hour, pm25, energy_mwh, traffic_index}`.
- POST `/api/retrain`: queues background retrain (stub).
- GET `/api/health` and GET `/` for status.

### Training

- CLI entry: `python urban_world_model.py`; writes checkpoints under `training/checkpoints/`.
- Config via `training/configs/base.yaml`; normalization stats auto‑updated from ETL outputs.
- TensorBoard logging, KL annealing, gradient clipping, LR scheduling.

### ETL

- WAQI feed (primary) via `backend/app/etl/waqi.py` using `WAQI_API_TOKEN`.
  - Extracts `real_time_pm25`, `pm25_values`, min/max, forecast; includes `measurement_timestamp`.
  - Marks `source` as `waqi` or `baseline` on fallback.
- OpenAQ v3 fallback via `backend/app/etl/openaq.py` with robust mean and backoff.
- Paths configurable via env: `LOGS_DIR` (default `./logs`), `PROCESSED_DATA_DIR` (default `./etl/processed_data`).

## 6) Non‑Functional Requirements

- Reliability: dockerized services; restart unless‑stopped; healthcheck on backend.
- Performance: targets in §2; chart render < 300 ms for 48 points.
- Security: CORS locked in prod; secrets via env; no secrets in git.
- Observability: structured logs; add metrics later; ETL logs under `LOGS_DIR`.
- Portability: `docker-compose up` runs all; `make` targets for DX.

## 7) Data Contracts (API)

TimeSeriesDataPoint

- hour: int
- pm25: float (µg/m³)
- energy_mwh: float (MWh)
- traffic_index: float (0–2)

PolicyConfig

- car_free_ratio: float [0,1]
- renewable_mix: float [0,1]

## 8) UX

- Responsive layout; accessible labels; keyboard‑operable sliders.
- Dark mode aware.

## 9) Risks / Mitigations

- Data gaps & rate limits → caching, sampling, retries; local synthetic fallbacks.
- Model complexity creep → stage: stub → baseline → DreamerV3.
- Latency under load → later: caching, async inference, autoscaling.

## 10) Acceptance Criteria (MVP)

- `make up` exposes FE (3000) and API (8000); `/docs` reachable.
- Changing sliders + run updates charts with 48 hourly points; mobile layout usable.
- `/api/simulate` matches schema with `meta.generated_at`.
- `make train` completes and writes checkpoints; TensorBoard logs present.
- ETL outputs written to `PROCESSED_DATA_DIR` with `real_time_pm25` when available.

## 11) Later Phases

- Spatial maps (tiles/hex), neighborhood stats.
- Real mobility/energy integrations (TomTom/HERE, EIA/ENTSO‑E).
- TorchServe/gRPC serving; Celery/Redis retraining.
- CI/CD, tests, model monitoring.
