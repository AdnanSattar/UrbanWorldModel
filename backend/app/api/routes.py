"""
Main API router aggregating all endpoints
"""

import logging

from fastapi import APIRouter, BackgroundTasks

from .explain import LatentSampleResponse, get_latent_sample
from .optimize import OptimizeRequest, OptimizeResponse, optimize_policy
from .retrain import RetrainRequest, trigger_retrain
from .simulate import SimulationRequest, SimulationResponse, simulate_policy

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "UrbanSim WM API"}


@router.post("/simulate", response_model=SimulationResponse)
def simulate(request: SimulationRequest):
    """
    Simulate urban dynamics based on policy parameters

    Returns forecasted PM2.5, energy consumption, and traffic indices
    over a specified time horizon.
    """
    logger.info(f"Simulation request received for city: {request.city}")
    return simulate_policy(request)


@router.post(
    "\/retrain",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {"config_path": "training/configs/base.yaml"}
                }
            }
        }
    },
)
def retrain(background_tasks: BackgroundTasks, request: RetrainRequest):
    """
    Trigger model retraining in the background

    This endpoint starts a background task to retrain the world model
    with updated data. The actual training happens asynchronously.
    """
    logger.info(f"Retrain request received with config: {request.config_path}")
    return trigger_retrain(background_tasks, request.config_path)


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    """Suggest a policy via coarse search over action space."""
    logger.info(
        f"Optimize request for city={request.city}, horizon={request.horizon_hours}"
    )
    return optimize_policy(request)


@router.get("/explain/latent_sample", response_model=LatentSampleResponse)
def latent_sample(city: str = "Lahore"):
    """Return a synthetic latent projection for the requested city."""
    return get_latent_sample(city)
