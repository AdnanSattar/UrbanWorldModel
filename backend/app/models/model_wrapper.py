"""
Model Wrapper for World Model Inference

This module provides a wrapper around the trained DreamerV3-style world model
for inference in the FastAPI backend.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from app.models.world_model import UrbanWorldModel

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wrapper class for the Urban World Model

    Handles model loading, preprocessing, and inference for policy simulations.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the model wrapper

        Args:
            checkpoint_path: Path to model checkpoint file (.ckpt or .pth)
        """
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelWrapper initialized with device: {self.device}")
        self.last_std: Optional[np.ndarray] = None

    def load(self):
        """
        Load the model checkpoint

        Loads encoder, RSSM, and predictor modules from checkpoint.
        Sets model to eval mode and moves to appropriate device (CPU/GPU).
        """
        logger.info(f"Loading model checkpoint: {self.checkpoint_path}")

        if self.checkpoint_path and self.checkpoint_path != "":
            try:
                checkpoint_path = Path(self.checkpoint_path)

                # Check if checkpoint exists
                if not checkpoint_path.exists():
                    logger.warning(
                        f"Checkpoint not found at {checkpoint_path}. "
                        "Using untrained model with random weights."
                    )
                    # Use default config from training
                    config = {
                        "encoder": {"latent_dim": 128, "input_dim": 10},
                        "rssm": {"latent_dim": 128, "hidden_dim": 256, "action_dim": 2},
                        "predictor": {
                            "latent_dim": 128,
                            "hidden_dim": 256,
                            "output_dim": 3,
                        },
                    }
                    self.model = UrbanWorldModel(config)
                    logger.info("Initialized model with default config (untrained)")
                    return

                # Load checkpoint
                if checkpoint_path.suffix == ".json":
                    # JSON checkpoint (training metadata)
                    with open(checkpoint_path, "r") as f:
                        checkpoint_data = json.load(f)
                    config = checkpoint_data.get("config", {})

                    # Try to load PyTorch weights if available
                    pt_path = checkpoint_path.with_suffix(".pth")
                    if pt_path.exists():
                        logger.info(f"Loading weights from {pt_path}")
                        state_dict = self._safe_torch_load(pt_path)
                        self.model = UrbanWorldModel(config)
                        self.model.load_state_dict(state_dict, strict=False)
                        self.model.to(self.device)
                        self.model.eval()
                        logger.info("Model loaded successfully with weights")
                        # Record model meta
                        self.model_meta = {
                            "checkpoint_json": str(checkpoint_path),
                            "checkpoint_pth": str(pt_path),
                            "model_step": checkpoint_data.get("step"),
                            "model_hash": self._sha256_file(pt_path),
                        }
                    else:
                        # No weights file, use default model
                        logger.warning(
                            f"No weights file found at {pt_path}, using untrained model"
                        )
                        self.model = UrbanWorldModel(config)
                        logger.info("Initialized model from config (no weights)")
                else:
                    # Direct PyTorch checkpoint
                    checkpoint = self._safe_torch_load(checkpoint_path)
                    if isinstance(checkpoint, dict) and "config" in checkpoint:
                        config = checkpoint.get("config", {})
                        self.model = UrbanWorldModel(config)
                        if "state_dict" in checkpoint:
                            self.model.load_state_dict(
                                checkpoint["state_dict"], strict=False
                            )
                        self.model.to(self.device)
                        self.model.eval()
                        logger.info("Model loaded successfully from PyTorch checkpoint")
                        self.model_meta = {
                            "checkpoint_json": None,
                            "checkpoint_pth": str(checkpoint_path),
                            "model_step": checkpoint.get("step"),
                            "model_hash": self._sha256_file(checkpoint_path),
                        }
                    else:
                        # Assume it's a state dict directly
                        config = {
                            "encoder": {"latent_dim": 128, "input_dim": 10},
                            "rssm": {
                                "latent_dim": 128,
                                "hidden_dim": 256,
                                "action_dim": 2,
                            },
                            "predictor": {
                                "latent_dim": 128,
                                "hidden_dim": 256,
                                "output_dim": 3,
                            },
                        }
                        self.model = UrbanWorldModel(config)
                        self.model.load_state_dict(checkpoint, strict=False)
                        self.model.to(self.device)
                        self.model.eval()
                        logger.info("Model loaded successfully from state dict")
                        self.model_meta = {
                            "checkpoint_json": None,
                            "checkpoint_pth": str(checkpoint_path),
                            "model_step": None,
                            "model_hash": self._sha256_file(checkpoint_path),
                        }

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.warning("Falling back to untrained model")
                # Fallback to default config
                config = {
                    "encoder": {"latent_dim": 128, "input_dim": 10},
                    "rssm": {"latent_dim": 128, "hidden_dim": 256, "action_dim": 2},
                    "predictor": {
                        "latent_dim": 128,
                        "hidden_dim": 256,
                        "output_dim": 3,
                    },
                }
                self.model = UrbanWorldModel(config)
        else:
            logger.warning("No checkpoint path provided, using untrained model")
            # Use default config
            config = {
                "encoder": {"latent_dim": 128, "input_dim": 10},
                "rssm": {"latent_dim": 128, "hidden_dim": 256, "action_dim": 2},
                "predictor": {"latent_dim": 128, "hidden_dim": 256, "output_dim": 3},
            }
            self.model = UrbanWorldModel(config)
            self.model_meta = None

    def _safe_torch_load(self, path: Path):
        """Safely load a PyTorch checkpoint, preferring weights_only when supported."""
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=True
            )  # PyTorch ≥ 2.4
            logging.info(f"Loaded checkpoint with weights_only=True: {path}")
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
            logging.warning(f"Loaded checkpoint without weights_only (legacy): {path}")
        return checkpoint

    def _sha256_file(self, path: Path) -> Optional[str]:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def predict(
        self, initial_state: Dict[str, Any], policy: Dict[str, float], horizon: int
    ) -> np.ndarray:
        """
        Predict future observations using the world model.

        Uses the trained model to perform inference given initial state and policy.
        Returns array with columns [pm25, energy_mwh, traffic_index].

        Args:
            initial_state: Dictionary with 'pm25', 'energy_mwh', 'traffic_index'
            policy: Dictionary with 'car_free_ratio', 'renewable_mix'
            horizon: Number of hours to predict

        Returns:
            numpy array of shape (horizon, 3) with predictions
        """
        if self.model is None:
            logger.warning("Model not loaded, falling back to baseline prediction")
            return self._baseline_predict(initial_state, policy, horizon)

        logger.info(f"[Model] Predicting {horizon}h with policy: {policy}")

        try:
            # Build normalized initial observation via preprocess
            merged = {
                **{k: v for k, v in initial_state.items()},
                **{k: v for k, v in policy.items()},
                "hour": 0,
                "day_of_week": 0,
            }
            initial_obs = self.preprocess(merged)  # (1, input_dim)

            # Create action sequence (policy repeated over horizon)
            action_dim = getattr(self.model.rssm, "action_dim", 2)
            actions = torch.zeros(1, horizon, action_dim, device=self.device)
            actions[0, :, 0] = float(policy.get("car_free_ratio", 0.0))
            actions[0, :, 1] = float(policy.get("renewable_mix", 0.0))

            # Use model forward pass directly with prepared tensors
            with torch.no_grad():
                predictions = self.model.forward(initial_obs, actions, horizon)

            # predictions shape: (1, horizon, 3)
            pred_array = predictions[0].detach().cpu().numpy()

            logger.info(f"Model prediction completed: shape {pred_array.shape}")
            return pred_array

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            logger.warning("Falling back to baseline prediction")
            return self._baseline_predict(initial_state, policy, horizon)

    def extract_latent_states(
        self, initial_state: Dict[str, Any], policy: Dict[str, float], horizon: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Extract latent states (z, h) from RSSM during inference for explainability.

        Args:
            initial_state: Dictionary with 'pm25', 'energy_mwh', 'traffic_index'
            policy: Dictionary with 'car_free_ratio', 'renewable_mix'
            horizon: Number of steps to extract (default: 10 for visualization)

        Returns:
            Dictionary with 'z' (stochastic latent) and 'h' (deterministic hidden) states
        """
        if self.model is None:
            logger.warning("Model not loaded, cannot extract latent states")
            return {"z": np.array([]), "h": np.array([])}

        try:
            # Build normalized initial observation
            merged = {
                **{k: v for k, v in initial_state.items()},
                **{k: v for k, v in policy.items()},
                "hour": 0,
                "day_of_week": 0,
            }
            initial_obs = self.preprocess(merged)

            # Create action sequence
            action_dim = getattr(self.model.rssm, "action_dim", 2)
            actions = torch.zeros(1, horizon, action_dim, device=self.device)
            actions[0, :, 0] = float(policy.get("car_free_ratio", 0.0))
            actions[0, :, 1] = float(policy.get("renewable_mix", 0.0))

            # Extract latent states during rollout
            batch_size = 1
            initial_z = self.model.encoder(initial_obs)
            state = self.model.rssm.init_state(batch_size)
            state["z"] = initial_z.to(self.device)
            state["h"] = state["h"].to(self.device)

            latent_states_z = [state["z"].detach().cpu().numpy()]
            latent_states_h = [state["h"].detach().cpu().numpy()]

            # Rollout and collect states
            with torch.no_grad():
                for t in range(horizon):
                    action_t = actions[:, t].to(self.device)
                    state = self.model.rssm.forward(state, action_t, observation=None)
                    latent_states_z.append(state["z"].detach().cpu().numpy())
                    latent_states_h.append(state["h"].detach().cpu().numpy())

            # Stack: (horizon+1, batch_size, latent_dim)
            z_stacked = np.vstack(latent_states_z)  # (horizon+1, latent_dim)
            h_stacked = np.vstack(latent_states_h)  # (horizon+1, hidden_dim)

            return {
                "z": z_stacked,
                "h": h_stacked,
            }
        except Exception as e:
            logger.error(f"Failed to extract latent states: {e}")
            return {"z": np.array([]), "h": np.array([])}

    def _baseline_predict(
        self, initial_state: Dict[str, Any], policy: Dict[str, float], horizon: int
    ) -> np.ndarray:
        """
        Baseline prediction using simple deterministic dynamics.

        Used as fallback when model is not available or fails.
        """
        car_free_ratio = float(policy.get("car_free_ratio", 0.0))
        renewable_mix = float(policy.get("renewable_mix", 0.0))

        base_pm25 = float(initial_state.get("pm25", 85.0))
        base_energy = float(initial_state.get("energy_mwh", 1200.0))
        base_traffic = float(initial_state.get("traffic_index", 1.0))

        pm25_renewable_reduction = renewable_mix * 0.25
        pm25_carfree_reduction = car_free_ratio * 0.30
        energy_renewable_reduction = renewable_mix * 0.15
        traffic_reduction = car_free_ratio

        out = []
        for h in range(horizon):
            hour_of_day = h % 24
            daily_factor = 0.8 + 0.4 * abs((hour_of_day - 12) / 12)

            pm25 = base_pm25 * (1 - pm25_renewable_reduction - pm25_carfree_reduction)
            pm25 = pm25 * daily_factor

            energy = base_energy * (1 - energy_renewable_reduction)
            energy = energy * daily_factor

            traffic = base_traffic * (1 - traffic_reduction)
            traffic = min(2.0, max(0.0, traffic * daily_factor))

            out.append([pm25, energy, traffic])

        return np.array(out, dtype=float)

    def preprocess(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess raw input data for the model
        """
        # Normalization stats aligned with training/configs/base.yaml
        pm25_mean, pm25_std = 50.0, 20.0
        energy_mean, energy_std = 1000.0, 200.0
        traffic_mean, traffic_std = 1.0, 0.3

        car_free = float(raw_data.get("car_free_ratio", 0.0))
        renewable = float(raw_data.get("renewable_mix", 0.0))
        gdp_idx = float(raw_data.get("gdp_activity_index", 1.0))
        commercial = float(raw_data.get("commercial_load_factor", 1.0))

        pm25 = float(raw_data.get("pm25", 85.0))
        energy = float(raw_data.get("energy_mwh", 1200.0))
        traffic = float(raw_data.get("traffic_index", 1.0))

        hour = int(raw_data.get("hour", 0)) % 24
        hour_sin = torch.sin(torch.tensor(hour * (2 * torch.pi / 24)))
        hour_cos = torch.cos(torch.tensor(hour * (2 * torch.pi / 24)))

        # Day of week optional; default to 0 (Monday)
        dow = int(raw_data.get("day_of_week", 0)) % 7
        dow_angle = dow * (2 * torch.pi / 7)
        dow_sin = torch.sin(torch.tensor(dow_angle))
        dow_cos = torch.cos(torch.tensor(dow_angle))

        # Normalize observed variables
        pm25_n = (pm25 - pm25_mean) / max(1e-6, pm25_std)
        energy_n = (energy - energy_mean) / max(1e-6, energy_std)
        traffic_n = (traffic - traffic_mean) / max(1e-6, traffic_std)

        # Assemble vector matching encoder.input_dim (default 12)
        features = [
            pm25_n,
            energy_n,
            traffic_n,
            car_free,
            renewable,
            gdp_idx,
            commercial,
            float(hour_sin),
            float(hour_cos),
            float(dow_sin),
            float(dow_cos),
        ]

        # Pad/truncate to encoder input size
        input_dim = getattr(self.model.encoder, "input_dim", 12) if self.model else 12
        if len(features) < input_dim:
            features.extend([0.0] * (input_dim - len(features)))
        else:
            features = features[:input_dim]

        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        return x

    def postprocess(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """
        Postprocess model outputs to human-readable format
        """
        if model_output is None:
            return {}

        out = model_output.detach().cpu()
        if out.dim() == 1 and out.numel() == 3:
            pm25, energy, traffic = [float(x) for x in out.tolist()]
            return {
                "pm25": max(5.0, pm25),
                "energy_mwh": max(100.0, energy),
                "traffic_index": max(0.0, min(2.0, traffic)),
            }

        if out.dim() == 2 and out.shape[-1] == 3:
            seq = []
            for row in out.tolist():
                pm25, energy, traffic = row
                seq.append(
                    {
                        "pm25": max(5.0, float(pm25)),
                        "energy_mwh": max(100.0, float(energy)),
                        "traffic_index": max(0.0, min(2.0, float(traffic))),
                    }
                )
            return {"sequence": seq}

        return {"raw": out.tolist()}


# Singleton pattern for model instance
_model_instance: Optional[ModelWrapper] = None


def get_model(checkpoint_path: Optional[str] = None) -> ModelWrapper:
    """
    Get or create the singleton model instance

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        ModelWrapper instance
    """
    global _model_instance

    if _model_instance is None:
        _model_instance = ModelWrapper(checkpoint_path)
        _model_instance.load()
        logger.info("Model instance created and loaded")

    return _model_instance


def get_model_meta() -> Dict[str, Any]:
    """Return metadata about the loaded model (hash, step, paths)."""
    global _model_instance
    if _model_instance is None:
        return {}
    return getattr(_model_instance, "model_meta", {}) or {}
