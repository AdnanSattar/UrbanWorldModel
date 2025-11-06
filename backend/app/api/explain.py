"""Explainability-related endpoints.

Extracts real latent states from the world model for visualization.
"""

import logging
from typing import List

import numpy as np
from app.core.config import settings
from app.data.loader import load_recent_pm25_mean
from app.models.model_wrapper import get_model
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LatentPoint(BaseModel):
    x: float
    y: float
    label: str
    weight: float = Field(default=1.0)


class LatentSampleResponse(BaseModel):
    points: List[LatentPoint]
    meta: dict


def _pca_reduce(latent_states: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Simple PCA dimensionality reduction using numpy.

    Args:
        latent_states: Array of shape (n_samples, n_features)
        n_components: Number of dimensions to reduce to (default: 2)

    Returns:
        Reduced array of shape (n_samples, n_components)
    """
    if latent_states.shape[0] == 0 or latent_states.shape[1] == 0:
        return np.array([])

    # Center the data
    mean = np.mean(latent_states, axis=0)
    centered = latent_states - mean

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue magnitude (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project to n_components dimensions
    projection = centered @ eigenvectors[:, :n_components]

    return projection


def get_latent_sample(city: str) -> LatentSampleResponse:
    """
    Extract real latent states from the model for explainability visualization.

    Uses actual model inference to extract latent representations (z, h) from RSSM,
    then projects them to 2D using PCA for visualization.

    Args:
        city: City name to use for initial state

    Returns:
        LatentSampleResponse with 2D projected points from actual model states
    """
    try:
        # Load model and get initial state
        model = get_model(settings.MODEL_CHECKPOINT_PATH)
        observed_pm25 = load_recent_pm25_mean(city) or 85.0

        initial_state = {
            "pm25": observed_pm25,
            "energy_mwh": 1200.0,
            "traffic_index": 1.0,
        }

        # Try different policy configurations to get diverse latent states
        policies = [
            {"car_free_ratio": 0.0, "renewable_mix": 0.0},
            {"car_free_ratio": 0.3, "renewable_mix": 0.0},
            {"car_free_ratio": 0.0, "renewable_mix": 0.5},
            {"car_free_ratio": 0.3, "renewable_mix": 0.5},
        ]

        all_z_states = []
        labels = []

        # Extract latent states for each policy
        for policy in policies:
            latent_data = model.extract_latent_states(initial_state, policy, horizon=10)
            z_states = latent_data.get("z", np.array([]))

            if z_states.shape[0] > 0:
                # Use middle states (avoid initial transient)
                mid_start = z_states.shape[0] // 3
                mid_end = (2 * z_states.shape[0]) // 3
                z_subset = z_states[mid_start:mid_end]
                all_z_states.append(z_subset)

                # Label based on policy
                if policy["car_free_ratio"] > 0 and policy["renewable_mix"] > 0:
                    label = "Energy + Mobility"
                elif policy["car_free_ratio"] > 0:
                    label = "Mobility"
                elif policy["renewable_mix"] > 0:
                    label = "Air Quality"
                else:
                    label = "Baseline"
                labels.extend([label] * z_subset.shape[0])

        if len(all_z_states) == 0:
            # Fallback to synthetic if model not available
            logger.warning("Could not extract latent states, using synthetic data")
            return _get_synthetic_latent_sample(city)

        # Concatenate all states
        z_combined = np.vstack(all_z_states)

        # Project to 2D using PCA
        z_2d = _pca_reduce(z_combined, n_components=2)

        if z_2d.shape[0] == 0:
            return _get_synthetic_latent_sample(city)

        # Normalize to [-1, 1] range for better visualization
        if z_2d.shape[0] > 1:
            z_2d = (z_2d - z_2d.mean(axis=0)) / (z_2d.std(axis=0) + 1e-8)
            z_2d = np.clip(z_2d, -2, 2)  # Clip outliers

        # Create points
        points: List[LatentPoint] = []
        for i, (label, (x, y)) in enumerate(zip(labels, z_2d)):
            points.append(
                LatentPoint(
                    x=float(x),
                    y=float(y),
                    label=label,
                    weight=1.0,
                )
            )

        meta = {
            "city": city,
            "description": "Real latent space projection from model RSSM states",
            "source": "model_inference",
            "n_states": len(points),
        }

        return LatentSampleResponse(points=points, meta=meta)

    except Exception as e:
        logger.error(f"Error extracting latent states: {e}", exc_info=True)
        # Fallback to synthetic data
        return _get_synthetic_latent_sample(city)


def _get_synthetic_latent_sample(city: str) -> LatentSampleResponse:
    """
    Fallback function that returns synthetic data when model is not available.
    """
    import random

    seed = hash(city.lower()) & 0xFFFFFFFF
    rng = random.Random(seed)
    clusters = [
        ("Energy", (0.6, 0.2)),
        ("Air Quality", (-0.4, 0.5)),
        ("Traffic", (-0.2, -0.6)),
    ]
    points: List[LatentPoint] = []
    for label, (cx, cy) in clusters:
        for _ in range(20):
            points.append(
                LatentPoint(
                    x=cx + rng.gauss(0, 0.2),
                    y=cy + rng.gauss(0, 0.2),
                    label=label,
                    weight=0.8 + rng.random() * 0.4,
                )
            )
    meta = {
        "city": city,
        "description": "Synthetic latent projection (fallback - model not available)",
        "source": "synthetic",
    }
    return LatentSampleResponse(points=points, meta=meta)
