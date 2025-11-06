"""
Recurrent State Space Model (RSSM) for UrbanSim WM

Implements the core temporal dynamics model inspired by DreamerV3.
The RSSM learns to predict future latent states given current state and actions.

Key components:
- Deterministic state: h_t (GRU/LSTM cell)
- Stochastic state: z_t (posterior and prior distributions)
- Transition model: p(z_t | h_t, z_{t-1})
- Representation model: q(z_t | h_t, o_t)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    """
    Recurrent State Space Model

    Maintains a latent state that combines:
    - Deterministic recurrent state (captures temporal patterns)
    - Stochastic state (captures uncertainty)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RSSM

        Args:
            config: RSSM configuration
                - latent_dim: dimension of stochastic state
                - hidden_dim: dimension of deterministic state
                - action_dim: dimension of action/policy space
        """
        super().__init__()
        self.config = config
        self.latent_dim = config.get("latent_dim", 128)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.action_dim = config.get("action_dim", 2)  # car_free_ratio, renewable_mix

        # Implement actual RSSM architecture
        # Components:
        # 1. Recurrent cell (GRU) for deterministic state
        self.recurrent = nn.GRUCell(
            input_size=self.latent_dim + self.action_dim, hidden_size=self.hidden_dim
        )

        # 2. Prior network: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim * 2),  # mean and logvar
        )

        # 3. Posterior network: q(z_t | h_t, obs_t)
        # For inference, we'll use latent_dim from encoder as obs_dim
        obs_dim = self.latent_dim  # Encoder output dimension
        self.posterior_net = nn.Sequential(
            nn.Linear(self.hidden_dim + obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim * 2),
        )

        self.initial_state = None

    def forward(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single step forward pass

        Args:
            prev_state: Previous state dict with 'h' and 'z'
            action: Action/policy tensor (batch_size, action_dim)
            observation: Current observation (for posterior, optional)

        Returns:
            Dictionary with new state:
                - 'h': deterministic state
                - 'z': stochastic state
                - 'prior_mean': mean of prior p(z_t | h_t)
                - 'prior_logvar': log variance of prior
                - 'posterior_mean': mean of posterior (if obs provided)
                - 'posterior_logvar': log variance of posterior
        """
        # Implement actual RSSM step
        # Extract previous states
        h_prev = prev_state["h"]
        z_prev = prev_state["z"]

        # 1. Update deterministic state
        h = self.recurrent(torch.cat([z_prev, action], dim=-1), h_prev)

        # 2. Compute prior distribution
        prior_params = self.prior_net(h)
        prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
        # Clamp logvar for numerical stability
        prior_logvar = torch.clamp(prior_logvar, min=-10, max=2)

        # 3. If observation available, compute posterior
        if observation is not None:
            posterior_params = self.posterior_net(torch.cat([h, observation], dim=-1))
            post_mean, post_logvar = torch.chunk(posterior_params, 2, dim=-1)
            post_logvar = torch.clamp(post_logvar, min=-10, max=2)
            z = self.reparameterize(post_mean, post_logvar)
            return {
                "h": h,
                "z": z,
                "prior_mean": prior_mean,
                "prior_logvar": prior_logvar,
                "posterior_mean": post_mean,
                "posterior_logvar": post_logvar,
            }
        else:
            # Use prior for prediction (no observation)
            z = self.reparameterize(prior_mean, prior_logvar)
            return {
                "h": h,
                "z": z,
                "prior_mean": prior_mean,
                "prior_logvar": prior_logvar,
            }

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from N(mean, var)

        Args:
            mean: Mean of distribution
            logvar: Log variance of distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Initialize state for a new sequence

        Args:
            batch_size: Batch size

        Returns:
            Initial state dictionary
        """
        return {
            "h": torch.zeros(batch_size, self.hidden_dim),
            "z": torch.zeros(batch_size, self.latent_dim),
        }

    def rollout(
        self,
        initial_state: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout the model for multiple steps (imagination)

        This is used for policy simulation without observations.

        Args:
            initial_state: Starting state
            actions: Sequence of actions (batch_size, horizon, action_dim)
            horizon: Number of steps to rollout

        Returns:
            Tuple of (hidden_states, latent_states) over time
        """
        h_sequence = []
        z_sequence = []

        state = initial_state

        for t in range(horizon):
            state = self.forward(state, actions[:, t], observation=None)
            h_sequence.append(state["h"])
            z_sequence.append(state["z"])

        # Stack sequences
        h_all = torch.stack(h_sequence, dim=1)  # (batch, horizon, hidden_dim)
        z_all = torch.stack(z_sequence, dim=1)  # (batch, horizon, latent_dim)

        return h_all, z_all


# Helper function
def create_rssm(config: Dict[str, Any]) -> RSSM:
    """Factory function to create RSSM from config"""
    return RSSM(config)
