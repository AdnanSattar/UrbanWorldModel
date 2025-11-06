"""
Predictor Module for UrbanSim WM

Decodes latent states back to observable predictions:
- Air quality (PM2.5, NO2)
- Energy consumption
- Traffic indices
- Other urban metrics

Also includes reward predictor for reinforcement learning (future extension).
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """
    Observation predictor network

    Decodes latent states to predicted observations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize predictor

        Args:
            config: Predictor configuration
                - latent_dim: dimension of input latent state
                - hidden_dim: dimension of RSSM hidden state
                - output_dim: number of output features
        """
        super().__init__()
        self.config = config
        self.latent_dim = config.get("latent_dim", 128)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.output_dim = config.get("output_dim", 3)  # PM2.5, energy, traffic

        # Implement actual predictor architecture
        # Input: concatenated [h, z] from RSSM
        # Output: predicted observations
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict observations from latent state

        Args:
            state: State dictionary with 'h' and 'z'
                - h: deterministic state (batch, hidden_dim)
                - z: stochastic state (batch, latent_dim)

        Returns:
            Dictionary of predictions:
                - pm25: Predicted PM2.5 levels
                - energy: Predicted energy consumption
                - traffic: Predicted traffic index
        """
        # Implement actual prediction
        h = state["h"]
        z = state["z"]
        latent = torch.cat([h, z], dim=-1)
        output = self.net(latent)

        # Separate outputs and apply appropriate activations
        # PM2.5: positive values, typically 5-200 µg/m³
        pm25 = F.softplus(output[:, 0:1]) * 200.0 + 5.0
        # Energy: positive values, typically 100-2000 MWh
        energy = F.softplus(output[:, 1:2]) * 2000.0 + 100.0
        # Traffic: sigmoid to [0, 2] range
        traffic = torch.sigmoid(output[:, 2:3]) * 2.0

        return {"pm25": pm25, "energy_mwh": energy, "traffic_index": traffic}

    def predict_sequence(
        self, state_sequence: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict observations for a sequence of states

        Args:
            state_sequence: Dictionary with sequences
                - h: (batch, seq_len, hidden_dim)
                - z: (batch, seq_len, latent_dim)

        Returns:
            Dictionary of predicted sequences
        """
        # Vectorized sequence prediction: flatten time into batch, run one forward, then reshape
        h = state_sequence["h"]  # (batch, seq, hidden)
        z = state_sequence["z"]  # (batch, seq, latent)
        batch_size, seq_len, _ = h.shape

        h_flat = h.reshape(batch_size * seq_len, self.hidden_dim)
        z_flat = z.reshape(batch_size * seq_len, self.latent_dim)

        out = self.forward({"h": h_flat, "z": z_flat})

        preds = {
            "pm25": out["pm25"].reshape(batch_size, seq_len, 1),
            "energy_mwh": out["energy_mwh"].reshape(batch_size, seq_len, 1),
            "traffic_index": out["traffic_index"].reshape(batch_size, seq_len, 1),
        }
        return preds


class RewardPredictor(nn.Module):
    """
    Reward predictor for RL-based policy optimization

    Future extension: predict reward signals for different policies
    Example rewards:
    - Lower PM2.5 levels (health benefit)
    - Lower energy costs
    - Reduced congestion
    - Weighted combination based on city priorities
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.latent_dim = config.get("latent_dim", 128)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.out_dim = 1

        # Simple MLP over concatenated [h, z] to predict a scalar reward
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim),
        )

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict reward from state

        Args:
            state: dict with keys 'h' and 'z'

        Returns:
            Tensor of shape (batch, 1)
        """
        h = state["h"]
        z = state["z"]
        x = torch.cat([h, z], dim=-1)
        return self.net(x)


# Helper functions
def create_predictor(config: Dict[str, Any]) -> Predictor:
    """Factory function to create predictor from config"""
    return Predictor(config)


def create_reward_predictor(config: Dict[str, Any]) -> RewardPredictor:
    """Factory function to create reward predictor from config"""
    return RewardPredictor(config)
