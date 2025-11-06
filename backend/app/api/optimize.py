"""
Simple policy optimization stub

Performs a coarse grid search over policy space using the current world model
to suggest an intervention given KPI weights.
"""

from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.data.loader import load_recent_pm25_mean
from app.models.model_wrapper import get_model
from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    city: str = Field(default="Lahore")
    horizon_hours: int = Field(default=48, ge=1, le=168)
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"pm25": 1.0, "energy": 0.3, "traffic": 0.3}
    )
    pm25_target: Optional[float] = Field(
        default=None, description="Upper bound target for mean PM2.5"
    )
    eval_budget: int = Field(default=32, ge=4, le=200)


class OptimizeResponse(BaseModel):
    policy: Dict[str, float]
    score: float
    candidates_tested: int


def optimize_policy(body: OptimizeRequest) -> OptimizeResponse:
    model = get_model(settings.MODEL_CHECKPOINT_PATH)
    observed_pm25 = load_recent_pm25_mean(body.city) or 85.0
    initial_state = {"pm25": observed_pm25, "energy_mwh": 1200.0, "traffic_index": 1.0}

    grid_car = [0.0, 0.1, 0.2, 0.3]
    grid_ren = [0.0, 0.2, 0.4, 0.6]
    best = {"car_free_ratio": 0.0, "renewable_mix": 0.0}
    best_score = float("inf")
    tested = 0
    remaining = max(4, body.eval_budget)

    def evaluate(car: float, ren: float) -> float:
        nonlocal best_score, best, tested, remaining
        if remaining <= 0:
            return float("inf")
        preds = model.predict(
            initial_state,
            {"car_free_ratio": car, "renewable_mix": ren},
            body.horizon_hours,
        )
        pm25 = float(preds[:, 0].mean())
        energy = float(preds[:, 1].mean())
        traffic = float(preds[:, 2].mean())
        score = (
            body.weights.get("pm25", 1.0) * pm25
            + body.weights.get("energy", 0.0) * energy
            + body.weights.get("traffic", 0.0) * traffic
        )
        tested += 1
        remaining -= 1
        if score < best_score:
            best_score = score
            best = {"car_free_ratio": car, "renewable_mix": ren}
        return pm25

    # Grid search
    for c in grid_car:
        for r in grid_ren:
            pm = evaluate(c, r)
            if remaining <= 0 or (body.pm25_target is not None and pm <= body.pm25_target):
                return OptimizeResponse(policy=best, score=best_score, candidates_tested=tested)

    # Random jittered samples (simple Bayesian-like exploration)
    import random

    while remaining > 0:
        c = min(0.6, max(0.0, best["car_free_ratio"] + random.uniform(-0.15, 0.15)))
        r = min(0.9, max(0.0, best["renewable_mix"] + random.uniform(-0.25, 0.25)))
        pm = evaluate(c, r)
        if remaining <= 0 or (body.pm25_target is not None and pm <= body.pm25_target):
            break

    return OptimizeResponse(policy=best, score=best_score, candidates_tested=tested)
