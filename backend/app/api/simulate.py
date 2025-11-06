"""
Simulation endpoint implementation
Generates forecasted urban dynamics based on policy interventions
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.data.loader import (
    load_energy_bundle,
    load_mobility_bundle,
    load_recent_pm25_mean,
)
from app.models.model_wrapper import get_model, get_model_meta
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Request/Response Models
class PolicyConfig(BaseModel):
    """Policy intervention parameters"""

    car_free_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of vehicles removed from roads (0.0-1.0)",
    )
    renewable_mix: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of energy from renewable sources (0.0-1.0)",
    )
    dynamic_pricing: Optional[bool] = Field(
        default=False, description="Enable dynamic demand pricing"
    )
    stricter_car_bans: Optional[bool] = Field(
        default=False, description="Apply stricter car bans"
    )
    gdp_activity_index: Optional[float] = Field(
        default=1.0, ge=0.5, le=1.5, description="Economic activity multiplier"
    )
    commercial_load_factor: Optional[float] = Field(
        default=1.0, ge=0.5, le=1.5, description="Commercial demand multiplier"
    )


class SimulationRequest(BaseModel):
    """Request body for simulation endpoint"""

    city: str = Field(default="Lahore", description="Target city name")
    start_time: Optional[str] = Field(
        default=None, description="Simulation start time (ISO format)"
    )
    horizon_hours: int = Field(
        default=48, ge=1, le=168, description="Forecast horizon in hours (max 1 week)"
    )
    policy: PolicyConfig = Field(
        default_factory=PolicyConfig, description="Policy intervention parameters"
    )
    uncertainty_samples: Optional[int] = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of MC-dropout samples for uncertainty",
    )


class TimeSeriesDataPoint(BaseModel):
    """Single time point in the simulation"""

    hour: int
    pm25: float = Field(description="PM2.5 concentration in µg/m³")
    energy_mwh: float = Field(description="Energy consumption in MWh")
    traffic_index: float = Field(description="Traffic congestion index (0-2)")
    pm25_std: Optional[float] = Field(default=None, description="Std dev of PM2.5")
    energy_std: Optional[float] = Field(default=None, description="Std dev of energy")
    traffic_std: Optional[float] = Field(default=None, description="Std dev of traffic")


class SimulationResponse(BaseModel):
    """Response containing simulation results"""

    city: str
    start: str
    horizon: int
    policy: PolicyConfig
    simulated: List[TimeSeriesDataPoint]
    meta: Dict[str, Any]


def simulate_policy(request: SimulationRequest) -> SimulationResponse:
    """
    Generate simulated urban dynamics based on policy parameters using world model inference.

    Uses the trained DreamerV3-style world model to predict future observations
    given initial state and policy interventions.

    Args:
        request: SimulationRequest with city, time, and policy params

    Returns:
        SimulationResponse with hourly forecasts
    """
    # Use current time if not specified
    start_time = request.start_time or datetime.utcnow().isoformat()

    # Extract policy parameters
    car_free_ratio = request.policy.car_free_ratio
    renewable_mix = request.policy.renewable_mix

    logger.info(
        f"Simulating {request.horizon_hours}h for {request.city} "
        f"with car_free={car_free_ratio:.2f}, renewable={renewable_mix:.2f}"
    )

    # Policy coupling and exogenous factors
    effective_car_free = car_free_ratio * (
        1.2 if (request.policy.stricter_car_bans or False) else 1.0
    )
    effective_renew = renewable_mix
    pricing_reduction = 0.1 if (request.policy.dynamic_pricing or False) else 0.0
    gdp_idx = request.policy.gdp_activity_index or 1.0
    commercial_load = request.policy.commercial_load_factor or 1.0

    # Use model inference via ModelWrapper
    model = get_model(settings.MODEL_CHECKPOINT_PATH)

    # Load initial PM2.5 from ETL if available
    observed_pm25 = load_recent_pm25_mean(request.city) or 85.0

    # Optional: adjust initial energy/traffic using ETL bundles if present
    mobility = load_mobility_bundle(request.city)
    energy_bundle = load_energy_bundle(request.city)

    # Use latest mobility/energy snapshots if available
    try:
        latest_mobility = (mobility.get("mobility") or [])[-1] if mobility else None
        latest_traffic = (mobility.get("traffic") or [])[-1] if mobility else None
    except Exception:
        latest_mobility, latest_traffic = None, None

    try:
        latest_energy = (energy_bundle.get("energy") or [])[-1] if energy_bundle else None
    except Exception:
        latest_energy = None
    base_energy = 1200.0 * gdp_idx * commercial_load * (1.0 - pricing_reduction)
    if latest_energy and isinstance(latest_energy.get("total_consumption_mwh"), (int, float)):
        base_energy = float(latest_energy["total_consumption_mwh"]) * (0.9 + 0.1 * gdp_idx)

    base_traffic = (
        float(latest_traffic["congestion_level"]) * 2.0
        if latest_traffic and isinstance(latest_traffic.get("congestion_level"), (int, float))
        else 1.0
    )
    base_traffic = max(0.0, min(2.0, base_traffic))

    initial_state = {
        "pm25": observed_pm25 * (0.98 + 0.02 * gdp_idx),
        "energy_mwh": base_energy,
        "traffic_index": max(
            0.0,
            base_traffic
            * gdp_idx
            * (1.0 - min(0.3, effective_car_free))
            * (1.0 - pricing_reduction),
        ),
    }

    # Run model prediction
    preds = model.predict(
        initial_state,
        {
            "car_free_ratio": max(0.0, min(1.0, effective_car_free)),
            "renewable_mix": max(0.0, min(1.0, effective_renew)),
        },
        request.horizon_hours,
    )
    # Optional uncertainty from model wrapper
    pm_std = getattr(model, "last_std", None)

    simulated = []
    for h in range(request.horizon_hours):
        # Model outputs are already in the correct format [pm25, energy_mwh, traffic_index]
        pm25 = float(preds[h, 0])
        energy = float(preds[h, 1])
        traffic = float(preds[h, 2])

        simulated.append(
            TimeSeriesDataPoint(
                hour=h,
                pm25=round(max(5.0, pm25), 2),
                energy_mwh=round(max(100.0, energy), 2),
                traffic_index=round(max(0.0, min(2.0, traffic)), 3),
                **(
                    {
                        "pm25_std": float(pm_std[h, 0]) if pm_std is not None else None,
                        "energy_std": (
                            float(pm_std[h, 1]) if pm_std is not None else None
                        ),
                        "traffic_std": (
                            float(pm_std[h, 2]) if pm_std is not None else None
                        ),
                    }
                ),
            )
        )

    response = SimulationResponse(
        city=request.city,
        start=start_time,
        horizon=request.horizon_hours,
        policy=request.policy,
        simulated=simulated,
        meta={
            "generated_at": time.time(),
            "model_version": "world_model_v1.0",
            "note": "Real model inference using DreamerV3-style world model",
            **(
                {
                    "model_step": get_model_meta().get("model_step"),
                    "model_hash": get_model_meta().get("model_hash"),
                    "checkpoint_file": get_model_meta().get("checkpoint_pth"),
                }
            ),
            "policy_effective": {
                "car_free_ratio": round(
                    float(max(0.0, min(1.0, effective_car_free))), 3
                ),
                "renewable_mix": round(float(max(0.0, min(1.0, effective_renew))), 3),
                "dynamic_pricing": bool(request.policy.dynamic_pricing or False),
                "stricter_car_bans": bool(request.policy.stricter_car_bans or False),
                "gdp_activity_index": round(float(gdp_idx), 2),
                "commercial_load_factor": round(float(commercial_load), 2),
            },
        },
    )
    logger.info(f"Simulation completed using world model inference")
    return response
