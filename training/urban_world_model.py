"""
UrbanSim WM - Training Harness for Urban World Model

This script trains a DreamerV3-style latent world model on urban time-series data
including air quality, energy consumption, and mobility metrics.

Implemented:
- End-to-end DreamerV3-style training loop (Encoder → RSSM → Predictor)
- Synthetic dataset generator (replace with real ETL later)
- Validation loop, periodic and best checkpoints, early stopping
- KL annealing, gradient clipping, LR warmup+cosine decay
- Optional TensorBoard logging
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.encoder import Encoder
from modules.predictor import Predictor
from modules.rssm import RSSM
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UrbanWorldModel(nn.Module):
    """
    DreamerV3-style world model for urban dynamics

    Architecture:
    1. Encoder: Maps observations (PM2.5, energy, traffic) to latent space
    2. RSSM: Recurrent State Space Model for temporal dynamics
    3. Predictor: Decodes latent states to predicted observations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the world model

        Args:
            config: Configuration dictionary with model hyperparameters
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

        logger.info(f"Model initialized on device: {self.device}")

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the world model

        Args:
            observations: (batch_size, seq_len, obs_dim) observation sequence
            actions: (batch_size, seq_len, action_dim) action sequence
            reconstruction_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence loss

        Returns:
            Tuple of (reconstruction_loss, kl_loss, state_dict)
        """
        batch_size, seq_len, obs_dim = observations.shape
        obs_dim_full = self.encoder.input_dim

        # Initialize state
        state = self.rssm.init_state(batch_size)
        state["h"] = state["h"].to(self.device)
        state["z"] = state["z"].to(self.device)

        # Prepare for reconstruction
        reconstructed_obs = []
        kl_losses = []

        # Process sequence
        for t in range(seq_len):
            obs_t = observations[:, t]  # (batch, obs_dim)
            action_t = actions[:, t]  # (batch, action_dim)

            # Encode observation
            if obs_t.shape[-1] < obs_dim_full:
                # Pad observation if needed
                obs_padded = torch.zeros(batch_size, obs_dim_full, device=self.device)
                obs_padded[:, : obs_t.shape[-1]] = obs_t
                obs_t = obs_padded
            else:
                obs_t = obs_t[:, :obs_dim_full]

            encoded_obs = self.encoder(obs_t)

            # RSSM forward (with observation for posterior)
            state = self.rssm.forward(state, action_t, observation=encoded_obs)

            # Predict observation
            pred = self.predictor(state)
            pred_tensor = torch.cat(
                [pred["pm25"], pred["energy_mwh"], pred["traffic_index"]], dim=-1
            )
            reconstructed_obs.append(pred_tensor)

            # Compute KL divergence if posterior available
            if "posterior_mean" in state and "posterior_logvar" in state:
                kl = self._kl_divergence(
                    state["posterior_mean"],
                    state["posterior_logvar"],
                    state["prior_mean"],
                    state["prior_logvar"],
                )
                kl_losses.append(kl)

        # Stack reconstructions
        reconstructed = torch.stack(reconstructed_obs, dim=1)  # (batch, seq_len, 3)

        # Compute reconstruction loss (MSE on observed dimensions)
        obs_target = observations[:, :, :3]  # Take first 3 dims (pm25, energy, traffic)
        reconstruction_loss = F.mse_loss(reconstructed, obs_target)

        # Compute KL loss
        kl_loss = (
            torch.stack(kl_losses).mean()
            if kl_losses
            else torch.tensor(0.0, device=self.device)
        )

        # Total loss
        total_loss = reconstruction_weight * reconstruction_loss + kl_weight * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def _kl_divergence(
        self,
        mean1: torch.Tensor,
        logvar1: torch.Tensor,
        mean2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between two normal distributions"""
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        kl = 0.5 * (logvar2 - logvar1 + (var1 + (mean1 - mean2) ** 2) / var2 - 1.0)
        return kl.sum(dim=-1).mean()

    def save_checkpoint(self, path: str, step: int, optimizer=None):
        """
        Save model checkpoint with full state dict

        Args:
            path: Path to save checkpoint (without extension)
            step: Training step number
            optimizer: Optional optimizer state
        """
        checkpoint_dir = Path(path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON metadata
        json_path = Path(path).with_suffix(".json")
        checkpoint_data = {
            "step": step,
            "config": self.config,
            "timestamp": time.time(),
        }
        with open(json_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        # Save PyTorch state dict
        pth_path = Path(path).with_suffix(".pth")
        state_dict = {
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "predictor": self.predictor.state_dict(),
            "step": step,
        }
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()

        torch.save(state_dict, pth_path)

        logger.info(f"Checkpoint saved: {json_path} and {pth_path}")


class SyntheticUrbanDataset(Dataset):
    """
    Synthetic dataset generator for training the world model.

    Generates realistic urban time-series sequences with:
    - PM2.5 (air quality)
    - Energy consumption
    - Traffic index
    - Policy actions (car_free_ratio, renewable_mix)
    """

    def __init__(self, num_samples: int = 1000, sequence_length: int = 50):
        """
        Initialize synthetic dataset

        Args:
            num_samples: Number of sequences to generate
            sequence_length: Length of each sequence (hours)
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a synthetic sequence"""
        # Generate realistic urban dynamics
        seq_len = self.sequence_length

        # Generate policy actions (car_free_ratio, renewable_mix)
        # Actions can change over time (policy interventions)
        actions = np.zeros((seq_len, 2))
        for t in range(seq_len):
            # Random policy values
            actions[t, 0] = np.random.uniform(0.0, 0.5)  # car_free_ratio
            actions[t, 1] = np.random.uniform(0.0, 0.6)  # renewable_mix

        # Generate observations with policy effects (+ exogenous features)
        # Columns: pm25, energy, traffic, gdp_activity_index, commercial_load_factor, hour_sin, hour_cos
        observations = np.zeros((seq_len, 7))

        # Initial values
        pm25 = 85.0
        energy = 1200.0
        traffic = 1.0

        # Exogenous drivers (slow drift around 1.0)
        gdp = 1.0 + np.random.normal(0, 0.02)
        commercial = 1.0 + np.random.normal(0, 0.03)

        for t in range(seq_len):
            hour_of_day = t % 24
            hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
            hour_cos = np.cos(2 * np.pi * hour_of_day / 24)

            # Policy effects
            car_free = actions[t, 0]
            renewable = actions[t, 1]

            # Daily variation
            daily_factor = 0.8 + 0.4 * abs((hour_of_day - 12) / 12)

            # Update values with policy impacts and daily patterns
            # Update exogenous with small random walk
            gdp = max(0.5, min(1.5, gdp + np.random.normal(0, 0.005)))
            commercial = max(0.5, min(1.5, commercial + np.random.normal(0, 0.007)))

            pm25 = (
                pm25
                * (1 - renewable * 0.25 - car_free * 0.30)
                * daily_factor
                * (0.98 + 0.02 * gdp)
            )
            pm25 = max(5.0, pm25 + np.random.normal(0, 3.0))

            energy = energy * (1 - renewable * 0.15) * daily_factor * gdp * commercial
            energy = max(100.0, energy + np.random.normal(0, 50.0))

            traffic = traffic * (1 - car_free) * daily_factor * gdp
            traffic = max(0.0, min(2.0, traffic + np.random.normal(0, 0.05)))

            # Add some temporal correlation
            pm25 = 0.7 * pm25 + 0.3 * (85.0 if t == 0 else observations[t - 1, 0])
            energy = 0.8 * energy + 0.2 * (1200.0 if t == 0 else observations[t - 1, 1])
            traffic = 0.8 * traffic + 0.2 * (1.0 if t == 0 else observations[t - 1, 2])

            observations[t, 0] = pm25
            observations[t, 1] = energy
            observations[t, 2] = traffic
            observations[t, 3] = gdp
            observations[t, 4] = commercial
            observations[t, 5] = hour_sin
            observations[t, 6] = hour_cos

        return {
            "observations": torch.FloatTensor(observations),
            "actions": torch.FloatTensor(actions),
        }


def load_config(config_path: str = "configs/base.yaml") -> Dict[str, Any]:
    """Load training configuration from YAML file"""
    try:
        config = OmegaConf.load(config_path)
        cfg = OmegaConf.to_container(config, resolve=True)
        # Optionally auto-update normalization stats from ETL cache
        try:
            cfg = _maybe_update_norm_from_etl(cfg)
        except Exception as _e:
            logger.warning(f"Auto norm update skipped: {_e}")
        return cfg
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return {
            "encoder": {"latent_dim": 128},
            "rssm": {"latent_dim": 128, "hidden_dim": 256},
            "predictor": {"output_dim": 3},
            "training": {"steps": 20, "batch_size": 32, "learning_rate": 1e-4},
        }


def _maybe_update_norm_from_etl(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute normalization stats from cached ETL files and inject into config.

    Looks under backend/app/etl/processed_data or etl/processed_data.
    """
    data_cfg = cfg.get("data", {})
    if not data_cfg:
        return cfg
    # Feature flags (set data.auto_update_norm: true to enable)
    auto = data_cfg.get("auto_update_norm", True)
    if not auto:
        return cfg
    import json
    import os
    from pathlib import Path

    import numpy as np

    candidates = [
        Path("./app/etl/processed_data"),
        Path("./app/etl/processed_data"),
    ]
    base = None
    for p in candidates:
        if p.exists():
            base = p
            break
    if base is None:
        return cfg

    pm_vals = []
    for f in base.glob("pm25_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("mean_pm25") is not None:
                pm_vals.append(float(d["mean_pm25"]))
        except Exception:
            continue

    # Energy: average across city files
    energy_vals = []
    for f in base.glob("energy_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                arr = json.load(fh)
            for row in arr:
                v = row.get("total_consumption_mwh")
                if v is not None:
                    energy_vals.append(float(v))
        except Exception:
            continue

    # Traffic: convert congestion_level (0..1) to [0..2]
    traffic_vals = []
    for f in base.glob("traffic_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                arr = json.load(fh)
            for row in arr:
                c = row.get("congestion_level")
                if c is not None:
                    traffic_vals.append(float(c) * 2.0)
        except Exception:
            continue

    def _ms(arr, default_mean, default_std):
        if not arr:
            return default_mean, default_std
        a = np.asarray(arr, dtype=float)
        return float(np.mean(a)), float(np.std(a) + 1e-6)

    pm_mean, pm_std = _ms(
        pm_vals, data_cfg.get("pm25_mean", 50.0), data_cfg.get("pm25_std", 20.0)
    )
    en_mean, en_std = _ms(
        energy_vals,
        data_cfg.get("energy_mean", 1000.0),
        data_cfg.get("energy_std", 200.0),
    )
    tr_mean, tr_std = _ms(
        traffic_vals,
        data_cfg.get("traffic_mean", 1.0),
        data_cfg.get("traffic_std", 0.3),
    )

    data_cfg.update(
        {
            "pm25_mean": pm_mean,
            "pm25_std": pm_std,
            "energy_mean": en_mean,
            "energy_std": en_std,
            "traffic_mean": tr_mean,
            "traffic_std": tr_std,
        }
    )
    cfg["data"] = data_cfg
    logger.info(
        f"Normalization updated from ETL: pm25=({pm_mean:.2f},{pm_std:.2f}) energy=({en_mean:.1f},{en_std:.1f}) traffic=({tr_mean:.2f},{tr_std:.2f})"
    )
    return cfg


def train_loop(config: Dict[str, Any]):
    """
    Main training loop

    Implements full training pipeline:
    - Load synthetic dataset
    - Create data loaders
    - Initialize optimizer
    - Training loop with checkpointing

    Args:
        config: Training configuration
    """
    logger.info("=" * 60)
    logger.info("UrbanSim WM - Training Started")
    logger.info("=" * 60)

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize model
    model = UrbanWorldModel(config)
    model.train()

    # Training parameters
    training_config = config.get("training", {})
    total_steps = training_config.get("steps", 1000)
    batch_size = training_config.get("batch_size", 32)
    sequence_length = training_config.get("sequence_length", 50)
    # Debug tiny batch override
    dbg = config.get("debug", {}).get("tiny_batch", {})
    if dbg.get("enabled", False):
        batch_size = dbg.get("batch_size", batch_size)
        sequence_length = dbg.get("sequence_length", sequence_length)
    learning_rate = training_config.get("learning_rate", 0.0003)
    reconstruction_weight = training_config.get("reconstruction_weight", 1.0)
    kl_weight = training_config.get("kl_weight", 0.1)
    # KL annealing config
    anneal_cfg = training_config.get("kl_annealing", None)
    # Grad clip config
    grad_clip_cfg = training_config.get("grad_clip", {"type": "norm", "value": 1.0})
    grad_clip_val = grad_clip_cfg.get("value", 1.0)
    checkpoint_interval = training_config.get("checkpoint_interval", 1000)
    checkpoint_dir = Path(training_config.get("checkpoint_dir", "./checkpoints"))
    log_interval = config.get("logging", {}).get("log_interval", 100)
    # Validation / early stopping
    val_interval = training_config.get("val_interval", 500)
    val_samples = training_config.get("val_samples", 100)
    save_best = training_config.get("save_best", False)
    early_cfg = training_config.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", False)
    early_patience = early_cfg.get("patience", 5)

    # Create dataset and dataloader
    num_samples = max(total_steps * batch_size, 1000)  # Generate enough samples
    dataset = SyntheticUrbanDataset(
        num_samples=num_samples, sequence_length=sequence_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize optimizer
    # Optimizer
    opt_cfg = training_config.get("optimizer", {"name": "adamw"})
    if opt_cfg.get("name", "adamw").lower() == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = opt_cfg.get("eps", 1e-8)
        weight_decay = opt_cfg.get("weight_decay", 1e-4)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler (linear warmup + cosine)
    sch_cfg = training_config.get("lr_scheduler", {"name": "linear_warmup_cosine"})
    scheduler = None
    if sch_cfg.get("name", "").lower() == "linear_warmup_cosine":
        warmup_steps = sch_cfg.get("warmup_steps", 0)
        min_lr = sch_cfg.get("min_lr", 1e-6)
        import math

        def lr_lambda(step):
            if step < warmup_steps:
                return max(1e-8, (step + 1) / max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            return max(min_lr / learning_rate, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info(f"Training for {total_steps} steps")
    logger.info(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {model.device}")

    # TensorBoard (optional)
    tb_dir = config.get("logging", {}).get("tensorboard_dir")
    writer = SummaryWriter(tb_dir) if tb_dir else None

    # Training loop
    step = 0
    dataloader_iter = iter(dataloader)

    best_val = float("inf")
    bad_counts = 0
    while step < total_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Move batch to device
        observations = batch["observations"].to(model.device)
        actions = batch["actions"].to(model.device)

        # Forward pass
        optimizer.zero_grad()
        # KL annealing factor
        kl_weight_eff = kl_weight
        if anneal_cfg:
            strat = anneal_cfg.get("strategy", "linear")
            start = anneal_cfg.get("start_step", 0)
            end = anneal_cfg.get("end_step", int(0.1 * total_steps))
            sv = float(anneal_cfg.get("start_value", 0.0))
            ev = float(anneal_cfg.get("end_value", kl_weight))
            cur = max(0, min(step, end))
            if cur <= start:
                kl_weight_eff = sv
            elif cur >= end:
                kl_weight_eff = ev
            else:
                frac = (cur - start) / max(1, (end - start))
                kl_weight_eff = sv + frac * (ev - sv)

        total_loss, recon_loss, kl_loss = model.forward(
            observations,
            actions,
            reconstruction_weight=reconstruction_weight,
            kl_weight=kl_weight_eff,
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if grad_clip_cfg.get("type", "norm") == "norm":
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        else:
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_val)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Logging
        if (step + 1) % log_interval == 0:
            tl = float(total_loss.item())
            rl = float(recon_loss.item())
            kl = float(kl_loss.item())
            logger.info(
                f"Step {step+1}/{total_steps} - Total {tl:.4f} | Recon {rl:.4f} | KL {kl:.4f} | KLw {kl_weight_eff:.4f}"
            )
            if writer:
                writer.add_scalar("train/total_loss", tl, step + 1)
                writer.add_scalar("train/recon_loss", rl, step + 1)
                writer.add_scalar("train/kl_loss", kl, step + 1)

        # Validation & checkpoints
        if (step + 1) % val_interval == 0 or (step + 1) == total_steps:
            model.eval()
            with torch.no_grad():
                vds = SyntheticUrbanDataset(
                    num_samples=val_samples, sequence_length=sequence_length
                )
                vloader = DataLoader(
                    vds, batch_size=batch_size, shuffle=False, num_workers=0
                )
                vals = []
                for vb in vloader:
                    vobs = vb["observations"].to(model.device)
                    vact = vb["actions"].to(model.device)
                    vtot, vrecon, vkl = model.forward(
                        vobs,
                        vact,
                        reconstruction_weight=reconstruction_weight,
                        kl_weight=kl_weight_eff,
                    )
                    vals.append(vtot.item())
                val_loss = float(np.mean(vals)) if vals else float("inf")
            model.train()
            if writer:
                writer.add_scalar("val/total_loss", val_loss, step + 1)

            # Periodic checkpoint
            if (step + 1) % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"model_step_{step+1}"
                model.save_checkpoint(str(checkpoint_path), step + 1, optimizer)
                logger.info(f"Checkpoint saved at step {step+1}")

            # Best checkpoint & early stopping
            if save_best:
                if val_loss < best_val:
                    best_val = val_loss
                    bad_counts = 0
                    best_path = checkpoint_dir / "model_best"
                    model.save_checkpoint(str(best_path), step + 1, optimizer)
                    logger.info(
                        f"Best checkpoint updated at step {step+1} (val_loss={best_val:.4f})"
                    )
                else:
                    bad_counts += 1
                    if early_enabled and bad_counts >= early_patience:
                        logger.info(f"Early stopping at step {step+1}")
                        break

        step += 1

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "model_final"
    model.save_checkpoint(str(final_checkpoint_path), total_steps, optimizer)

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")
    logger.info("=" * 60)

    if writer:
        writer.close()


if __name__ == "__main__":
    # Load configuration
    config = load_config("configs/base.yaml")

    # Run training
    train_loop(config)
