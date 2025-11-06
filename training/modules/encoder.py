"""
Encoder Module for UrbanSim WM

Encodes multi-modal observations (air quality, energy, traffic, time/economic
features) into a compact latent representation. Implemented architecture:
- Tabular encoder: MLP with LayerNorm/Dropout for time-series features
- Optional spatial path (commented scaffold) for grid inputs (e.g., heatmaps)
- Simple fusion via concatenation/project (current implementation uses tabular)
"""

from typing import Any, Dict

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network for urban observations

    Maps raw observations to latent embeddings for the RSSM.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize encoder

        Args:
            config: Encoder configuration
                - latent_dim: dimension of latent representation
                - input_dim: dimension of input observations
        """
        super().__init__()
        self.config = config
        self.latent_dim = config.get("latent_dim", 128)
        self.input_dim = config.get("input_dim", 10)  # PM2.5, energy, traffic, etc.

        # Implement actual encoder architecture for tabular time-series data
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.latent_dim),
        )

        # Example for spatial data (grid-based):
        # self.conv_net = nn.Sequential(
        #     nn.Conv2d(in_channels, 32, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(flattened_size, self.latent_dim)
        # )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to latent space

        Args:
            observations: Raw observations (batch_size, obs_dim)
                Expected features:
                - PM2.5 concentration
                - NO2 concentration
                - Energy consumption
                - Traffic density
                - Weather features (temperature, humidity, wind)
                - Time features (hour, day_of_week)

        Returns:
            Latent embeddings (batch_size, latent_dim)
        """
        # Forward pass through encoder network
        return self.net(observations)

    def encode_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of multi-modal observations

        Args:
            batch: Dictionary with different observation modalities
                - 'tabular': time-series features
                - 'spatial': grid-based features (optional)
                - 'metadata': city info, policy params

        Returns:
            Batch of latent embeddings
        """
        # Implement multi-modal encoding
        tabular_features = batch.get("tabular")
        if tabular_features is None:
            # Fallback: create features from initial state if available
            if "initial_state" in batch:
                state = batch["initial_state"]
                # Create feature vector: [pm25, energy, traffic, time_features, policy_features]
                features = torch.zeros(1, self.input_dim)
                if isinstance(state, dict) and len(state) > 0:
                    features[0, 0] = state.get("pm25", 85.0) / 200.0  # Normalize
                    features[0, 1] = state.get("energy_mwh", 1200.0) / 2000.0
                    features[0, 2] = state.get("traffic_index", 1.0) / 2.0
                return self.forward(features)
            return self.forward(torch.zeros(1, self.input_dim))
        return self.forward(tabular_features)


# Helper function for creating encoder
def create_encoder(config: Dict[str, Any]) -> Encoder:
    """Factory function to create encoder from config"""
    return Encoder(config)
