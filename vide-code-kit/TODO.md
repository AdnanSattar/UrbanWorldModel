# UrbanSim WM — TODO Backlog

Owner: UrbanSim WM Team  
Status: Live

## Legend

- [ ] Todo
- [~] In Progress
- [x] Done
- (P1) High | (P2) Medium | (P3) Low

---

## MVP (v0.1.0)

- [x] (P1) Scaffold FastAPI backend with `/api/simulate`, `/api/health`, `/retrain`
- [x] (P1) Next.js 14 App Router UI at `/` with two sliders
- [x] (P1) Recharts: PM2.5, Energy, Traffic charts with summary stats
- [x] (P1) Synthetic simulation logic and example response
- [x] (P1) Training skeleton: encoder, rssm, predictor, config, checkpoint write
- [x] (P1) ETL stubs: OpenAQ, mobility, energy
- [x] (P1) Docker Compose + Makefile + `.env.example`
- [x] (P2) Docs: README, QUICKSTART, CONTRIBUTING, IMPLEMENTATION_SUMMARY
- [x] (P2) Frontend Dockerfile & production build

## Beta (v0.2.0)

- [x] (P1) Integrate OpenAQ ETL into a repeatable pipeline (cache + sampling)
  - ✅ WAQI ETL integrated with real-time data extraction
  - ✅ OpenAQ ETL available as fallback
  - ✅ ETL bootstrap pipeline runs on backend startup
  - ✅ Data freshness validation and timestamp extraction
- [x] (P1) Add baseline inference into `ModelWrapper` (e.g., MLP/ARIMA)
  - ✅ `_baseline_predict()` method implemented with deterministic dynamics
  - ✅ Fallback mechanism when model unavailable
- [ ] (P1) Input validation + error handling polish for `/api/simulate`
- [ ] (P2) Add comparison view of multiple scenarios (UI)
- [ ] (P2) Export results to CSV/JSON from UI
- [ ] (P2) Add simple caching for repeat simulations
- [ ] (P3) Add linting/test scaffolding (backend & frontend)

## Model (v0.3.0)

- [ ] (P1) Implement DreamerV3 training loop: losses, optimizer, logging
- [ ] (P1) Add dataset loaders for ETL outputs (train/val/test split)
- [ ] (P1) Save/load real checkpoints; integrate into `ModelWrapper`
- [ ] (P2) Add metrics dashboard (TensorBoard)
- [ ] (P2) Add uncertainty estimates to predictions

## Data Integrations (v0.4.0)

- [ ] (P1) Mobility data: integrate real source (TomTom/HERE or city feeds)
- [ ] (P1) Energy data: integrate EIA/ENTSO‑E or city provider
- [ ] (P2) Weather data integration for improved AQ prediction

## Production Hardening (v0.5.0)

- [ ] (P1) Add CI/CD (GitHub Actions): lint, build, smoke tests
- [ ] (P1) Add observability (metrics, traces) and structured JSON logs
- [ ] (P1) Add caching layer (Redis) + rate limiting
- [ ] (P2) AuthN/Z if multi‑tenant requirements emerge
- [ ] (P2) Swap stub serving with TorchServe/gRPC for scalable inference

## UX & Visualization

- [x] (P2) Enhance explainability endpoint with real model latent states
  - ✅ Extract actual latent states (z, h) from RSSM during inference
  - ✅ PCA projection to 2D for visualization
  - ✅ Policy-based labeling for diverse latent representations
- [ ] (P2) Add map-based spatial visualization (phase 2)
- [ ] (P2) Mobile layout improvements and accessibility audit

## Docs & Ops

- [x] (P2) PRD & Plan docs at repo root
- [x] (P2) Comprehensive documentation suite
  - ✅ Data flow documentation (ETL → Model)
  - ✅ Model inference flow guide
  - ✅ Alternative data sources guide
  - ✅ Codebase status and stubs inventory
  - ✅ WAQI setup guide
- [ ] (P3) Architecture diagram (C4-like) under `docs/`
- [ ] (P3) API JSON schema docs and examples under `docs/`

---

## TODO — Work Breakdown

### ⏳ Week 1–2

- [x] Finalize Dockerfiles for backend, frontend, training.
  - ✅ All Dockerfiles implemented
  - ✅ Docker Compose configuration complete
- [x] Add `.env.example` and initial config files.
  - ✅ `.env.example` with all required variables
  - ✅ Config management via pydantic-settings
- [x] Spin up dev environment via docker-compose.
  - ✅ `make up` command working
  - ✅ All services running in Docker

### 🧠 Week 3–6

- [x] Implement ETL for OpenAQ and mobility datasets.
  - ✅ WAQI ETL integration (real-time air quality data)
  - ✅ OpenAQ ETL as fallback option
  - ✅ ETL bootstrap pipeline on backend startup
  - ⚠️ Mobility datasets still using synthetic data
- [x] Create DreamerV3-like RSSM training stub.
  - ✅ Encoder, RSSM, Predictor modules implemented
  - ✅ Training loop structure complete
- [ ] Log training metrics to TensorBoard.

### ⚙️ Week 7–9

- [x] Add `/simulate` endpoint logic (model inference).
  - ✅ ModelWrapper with baseline prediction
  - ✅ Integration with ETL data (real-time PM2.5)
  - ✅ Policy-based simulation working
- [x] Add `/retrain` background task.
  - ✅ Background task implementation
  - ✅ Log streaming to training/logs/
  - ✅ Checkpoint management
- [x] Test API response structure.
  - ✅ All endpoints operational
  - ✅ Response models validated

### 💻 Week 10–12

- [x] Add App Router pages `/` and `/metrics`.
  - ✅ Main simulator page at `/` with policy controls
  - ✅ Charts and visualization components
- [x] Integrate Recharts visualizations.
  - ✅ PM2.5, Energy, Traffic charts
  - ✅ Real-time simulation visualization
- [x] Add Tailwind styling and responsive layout.
  - ✅ Basic Tailwind styling implemented
  - ✅ Mobile layout improvements completed
    - Responsive typography (text sizes scale with screen size)
    - Mobile-optimized padding and spacing
    - Touch-friendly buttons (min 44px height on mobile)
    - Responsive grid layouts (1 col mobile → 2-4 cols desktop)
    - Horizontal scrolling for charts on mobile
    - Optimized city selector and form elements
    - Better breakpoint usage (sm, md, lg)
    - Improved accessibility with touch-manipulation

### 🚀 Week 13–16

- [ ] Setup GitHub Actions CI/CD.
- [ ] Add unit tests (pytest + frontend smoke tests).
- [ ] Push final container images to GHCR.
