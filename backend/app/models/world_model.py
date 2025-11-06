"""
World Model for UrbanSim WM Backend

This module provides the UrbanWorldModel class that can be imported
and used by the ModelWrapper for inference.
"""

import logging

# Import model modules from training directory
# Note: These need to be accessible from the backend
# In production, you might want to package these as a shared library
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

# Add training directory to path so we can import modules
# Try multiple possible paths for Docker and local development
possible_paths = [
    Path(__file__).parent.parent.parent.parent
    / "training",  # Local: backend/app/models/../../training
    Path("/app/../training"),  # Docker: if training is sibling to backend
    Path("/train"),  # Docker: if training is mounted at /train
    Path("/training"),  # Docker: if training is mounted at /training
    Path("/app/training"),  # Docker: if training is copied to backend
]

for training_path in possible_paths:
    if training_path.exists() and (training_path / "modules").exists():
        sys.path.insert(0, str(training_path))
        break
else:
    # Fallback: assume training is in parent directory
    training_path = Path(__file__).parent.parent.parent.parent / "training"
    sys.path.insert(0, str(training_path))

from modules.encoder import Encoder
from modules.predictor import Predictor
from modules.rssm import RSSM

logger = logging.getLogger(__name__)


class UrbanWorldModel(nn.Module):
    """
    Complete world model combining encoder, RSSM, and predictor.

    This is the inference-ready version of the model used in training.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the world model

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model components
        encoder_config = config.get("encoder", {})
        rssm_config = config.get("rssm", {})
        predictor_config = config.get("predictor", {})

        self.encoder = Encoder(encoder_config)
        self.rssm = RSSM(rssm_config)
        self.predictor = Predictor(predictor_config)

        # Move to device
        self.to(self.device)
        self.eval()

        logger.info(f"UrbanWorldModel initialized on device: {self.device}")

    def forward(
        self, initial_obs: torch.Tensor, actions: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        """
        Forward pass: predict future observations given initial state and actions

        Args:
            initial_obs: Initial observation tensor (batch_size, obs_dim)
            actions: Action sequence tensor (batch_size, horizon, action_dim)
            horizon: Number of steps to predict

        Returns:
            Predicted observations tensor (batch_size, horizon, output_dim)
        """
        batch_size = initial_obs.shape[0]

        # Encode initial observation
        initial_z = self.encoder(initial_obs)

        # Initialize RSSM state
        state = self.rssm.init_state(batch_size)
        state["z"] = initial_z.to(self.device)
        state["h"] = state["h"].to(self.device)

        # Prepare output tensor
        predictions = []

        # Rollout for horizon steps
        for t in range(horizon):
            action_t = actions[:, t] if actions.dim() == 3 else actions
            action_t = action_t.to(self.device)

            # Update RSSM state (no observation for future prediction)
            state = self.rssm.forward(state, action_t, observation=None)

            # Predict observations from latent state
            pred = self.predictor(state)

            # Combine predictions into single tensor
            pred_tensor = torch.cat(
                [pred["pm25"], pred["energy_mwh"], pred["traffic_index"]], dim=-1
            )

            predictions.append(pred_tensor)

        # Stack predictions: (batch_size, horizon, 3)
        return torch.stack(predictions, dim=1)

    def predict_sequence(
        self, initial_state: Dict[str, float], policy: Dict[str, float], horizon: int
    ) -> torch.Tensor:
        """
        Predict sequence from initial state and policy

        Args:
            initial_state: Dictionary with 'pm25', 'energy_mwh', 'traffic_index'
            policy: Dictionary with 'car_free_ratio', 'renewable_mix'
            horizon: Number of hours to predict

        Returns:
            Predictions tensor (1, horizon, 3) with [pm25, energy_mwh, traffic_index]
        """
        # Normalize initial state to create observation vector
        # Format: [pm25, energy, traffic, time_features, policy_features]
        obs_dim = self.encoder.input_dim
        obs = torch.zeros(1, obs_dim, device=self.device)

        # Normalize initial state values
        obs[0, 0] = initial_state.get("pm25", 85.0) / 200.0
        obs[0, 1] = initial_state.get("energy_mwh", 1200.0) / 2000.0
        obs[0, 2] = initial_state.get("traffic_index", 1.0) / 2.0

        # Add policy features
        obs[0, 3] = policy.get("car_free_ratio", 0.0)
        obs[0, 4] = policy.get("renewable_mix", 0.0)

        # Create action sequence (same policy for all steps)
        action_dim = self.rssm.action_dim
        actions = torch.zeros(1, horizon, action_dim, device=self.device)
        actions[0, :, 0] = policy.get("car_free_ratio", 0.0)
        actions[0, :, 1] = policy.get("renewable_mix", 0.0)

        # Predict
        with torch.no_grad():
            predictions = self.forward(obs, actions, horizon)

        return predictions
