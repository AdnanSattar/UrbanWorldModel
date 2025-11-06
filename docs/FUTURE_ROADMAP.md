# Future Roadmap

This document outlines the planned phases beyond the MVP, with focus areas and suggested milestones.

## Phase 1: Current (MVP) ✅

- [x] WAQI ETL integration (primary) with OpenAQ fallback
- [x] Real-time PM2.5 prioritized in model
- [x] Data freshness checking and timestamps
- [x] Baseline inference and mobile-responsive UI

## Phase 2: Next (Beta) 📋

- [ ] Real mobility data integration
  - Start: Google Mobility Reports (CSV)
  - Then: TomTom/HERE APIs (real-time)
- [ ] Real energy data integration
  - Research Pakistan grid providers
  - Or: EIA/ENTSO‑E as reference
- [ ] ETL caching and optimization (Redis)
- [ ] Input validation enhancements for `/api/simulate`
- [ ] UI scenario comparison and export (CSV/JSON)
- [ ] Initial test scaffolding (backend + frontend)

## Phase 3: Production 🚀

- [ ] Multi-source data aggregation (AQ + Mobility + Energy)
- [ ] Advanced explainability from model internals
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Observability (metrics, traces, structured logs)
- [ ] Broader test suite (unit, integration, e2e)
- [ ] Performance optimization and caching for inference

## Nice-to-Haves

- [ ] TorchServe/gRPC for scalable inference
- [ ] AuthN/Z for multi-tenant deployments
- [ ] Spatial visualizations (map overlays, districts)
- [ ] RL-based policy recommendation (longer-term)
