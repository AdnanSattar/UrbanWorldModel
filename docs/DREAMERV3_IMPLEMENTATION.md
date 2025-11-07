# DreamerV3 Implementation in UrbanSim WM

## Table of Contents

1. [What is DreamerV3?](#what-is-dreamerv3)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Training Process](#training-process)
5. [Inference Process](#inference-process)
6. [Code Locations](#code-locations)
7. [How It All Works Together](#how-it-all-works-together)

---

## What is DreamerV3?

**DreamerV3** is a state-of-the-art world model architecture developed by Google DeepMind. It's a **latent world model**

**Latent** refers to hidden, unobserved, or underlying variables that are not directly measured but are inferred from observable data. These variables form a compressed, abstract representation (known as the latent space) of the complex, high-dimensional sensory input, such as images or raw sensor data.

**A latent world model** is a compressed, simplified representation of **reality** that an AI creates to predict and plan for the future. Instead of working directly with high-dimensional, pixel-level data like images, it distills the world into a compact "latent" or unobserved state, which represents the essential features and dynamics. This makes it more efficient for the AI to process and learn, leading to more powerful decision-making in tasks like reinforcement learning and autonomous driving.

The key functions and characteristics of the latent aspect in a world model are:

**Hidden Representation**: The raw observations (e.g., every pixel in a video feed) are high-dimensional and contain redundant information. The model learns to encode this into a lower-dimensional "state" that captures only the essential and relevant information about the environment's dynamics.

**Inferred, Not Directly Observed**: You cannot directly "see" or explicitly measure the latent state in the same way you can a pixel value. Its value is inferred through the model's internal processing, typically using an encoder network in an autoencoder-like architecture.

**Causal Factors**: These latent variables are often hypothesized to be the underlying causes of the observed data patterns. For example, in an autonomous driving model, latent variables might represent meaningful physical quantities like the position and velocity of other cars, which are not directly observed as a single value but influence the visible pixels over time.

**Facilitates Prediction and Planning**: By working with a compact, meaningful latent representation, the model can more efficiently learn the world's dynamics and predict future states and outcomes of actions. This is much more tractable than trying to predict future outcomes in the raw, high-dimensional observation space.

**Enables Generalization**: By forcing the model to focus on the core, underlying factors and disregard noise or irrelevant details, the latent representation helps the model generalize better to new situations.

**A latent world model** learns to predict future states of an environment by:

1. **Encoding** observations into a compact latent representation
2. **Learning temporal dynamics** in latent space (much more efficient than raw observations)
3. **Decoding** latent states back to predictions

### Why Used Latent World Model (DreamerV3) for Urban Simulation?

- **Efficiency**: Works in compressed latent space (128-256 dims) vs. raw observations
- **Uncertainty Modeling**: Stochastic latent states capture uncertainty
- **Policy Simulation**: Can "imagine" future scenarios without real data
- **Interpretability**: Latent states can be visualized and analyzed

### Key Concepts

- **Latent State (z)**: Stochastic representation of the world (captures uncertainty)
- **Hidden State (h)**: Deterministic recurrent state (captures temporal patterns)
- **Prior**: `p(z_t | h_t)` - What the model predicts without observation
- **Posterior**: `q(z_t | h_t, o_t)` - What the model infers from observation

---

## Architecture Overview

```bash
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Observations                       │
│  [PM2.5, Energy, Traffic, Time, Policy, GDP, Commercial]   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  1. ENCODER (training/modules/encoder.py)                   │
│     MLP: input_dim (12) → 256 → 256 → latent_dim (128)      │
│     Output: z_encoded (latent embedding)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  2. RSSM (training/modules/rssm.py)                        │
│     ┌─────────────────────────────────────┐                 │
│     │ Deterministic State (h)              │                 │
│     │ GRU: (z_prev + action) → h_t        │                 │
│     └──────────────┬──────────────────────┘                 │
│                    │                                          │
│     ┌──────────────▼──────────────────────┐                 │
│     │ Prior: p(z_t | h_t)                 │                 │
│     │ MLP: h_t → [mean, logvar]           │                 │
│     └──────────────┬──────────────────────┘                 │
│                    │                                          │
│     ┌──────────────▼──────────────────────┐                 │
│     │ Posterior: q(z_t | h_t, obs_t)      │                 │
│     │ MLP: [h_t, obs_t] → [mean, logvar]  │                 │
│     └──────────────┬──────────────────────┘                 │
│                    │                                          │
│     ┌──────────────▼──────────────────────┐                 │
│     │ Reparameterization Trick             │                 │
│     │ z_t = mean + ε * sqrt(var)          │                 │
│     └──────────────┬──────────────────────┘                 │
│                    │                                          │
│     Output: {h_t, z_t, prior_mean, prior_logvar, ...}       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  3. PREDICTOR (training/modules/predictor.py)                │
│     MLP: [h_t, z_t] → 256 → 256 → output_dim (3)           │
│     Output: {pm25, energy_mwh, traffic_index}              │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Encoder (`training/modules/encoder.py`)

**Purpose**: Compress raw observations into latent space

**Architecture**:

```python
Encoder(
  input_dim: 12  # PM2.5, energy, traffic, gdp, commercial, hour_sin, hour_cos, ...
  ↓
  Linear(12 → 256) + LayerNorm + ReLU + Dropout(0.1)
  ↓
  Linear(256 → 256) + LayerNorm + ReLU + Dropout(0.1)
  ↓
  Linear(256 → 128)
  ↓
  latent_dim: 128
)
```

**Input Features** (12 dimensions):

- `pm25`: Air quality (normalized)
- `energy_mwh`: Energy consumption (normalized)
- `traffic_index`: Traffic density (normalized)
- `car_free_ratio`: Policy action (0-1)
- `renewable_mix`: Policy action (0-1)
- `gdp_activity_index`: Economic indicator
- `commercial_load_factor`: Commercial activity
- `hour_sin`, `hour_cos`: Time encoding (24h cycle)
- `dow_sin`, `dow_cos`: Day of week encoding (7d cycle)
- Padding/truncation to match `input_dim`

**Key Code**:

```python
def forward(self, observations: torch.Tensor) -> torch.Tensor:
    """Encode observations to latent space"""
    return self.net(observations)  # (batch, 12) → (batch, 128)
```

---

### 2. RSSM (`training/modules/rssm.py`)

**Purpose**: Learn temporal dynamics in latent space

**Components**:

#### a) Deterministic State (h_t)

- **GRU Cell**: `GRUCell(latent_dim + action_dim → hidden_dim)`
- **Input**: `[z_{t-1}, action_t]` (previous latent + current action)
- **Output**: `h_t` (256-dim deterministic state)

#### b) Prior Network

- **Purpose**: Predict latent state without observation
- **Architecture**: `MLP(h_t → 256 → [mean, logvar])`
- **Output**: `p(z_t | h_t)` = Normal(mean, var)

#### c) Posterior Network

- **Purpose**: Infer latent state from observation
- **Architecture**: `MLP([h_t, obs_t] → 256 → [mean, logvar])`
- **Output**: `q(z_t | h_t, obs_t)` = Normal(mean, var)

#### d) Reparameterization Trick

```python
z_t = mean + ε * sqrt(exp(logvar))
where ε ~ N(0, 1)
```

**Key Code**:

```python
def forward(self, prev_state, action, observation=None):
    # 1. Update deterministic state
    h = self.recurrent(torch.cat([z_prev, action], dim=-1), h_prev)
    
    # 2. Compute prior
    prior_params = self.prior_net(h)  # → [mean, logvar]
    
    # 3. If observation available, compute posterior
    if observation is not None:
        posterior_params = self.posterior_net(torch.cat([h, observation], dim=-1))
        z = self.reparameterize(posterior_mean, posterior_logvar)
    else:
        z = self.reparameterize(prior_mean, prior_logvar)
    
    return {"h": h, "z": z, ...}
```

**Rollout (Imagination)**:

```python
def rollout(self, initial_state, actions, horizon):
    """Rollout model for multiple steps without observations"""
    for t in range(horizon):
        state = self.forward(state, actions[:, t], observation=None)
        # Uses PRIOR (not posterior) for future prediction
```

---

### 3. Predictor (`training/modules/predictor.py`)

**Purpose**: Decode latent states back to observations

**Architecture**:

```python
Predictor(
  input: [h_t, z_t]  # Concatenated (256 + 128 = 384 dims)
  ↓
  Linear(384 → 256) + LayerNorm + ReLU + Dropout(0.1)
  ↓
  Linear(256 → 256) + LayerNorm + ReLU + Dropout(0.1)
  ↓
  Linear(256 → 3)
  ↓
  Output: {pm25, energy_mwh, traffic_index}
)
```

**Output Activation**:

- `pm25`: `softplus(output[0]) * 200 + 5` → [5, 205] µg/m³
- `energy_mwh`: `softplus(output[1]) * 2000 + 100` → [100, 2100] MWh
- `traffic_index`: `sigmoid(output[2]) * 2` → [0, 2]

**Key Code**:

```python
def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    h = state["h"]  # (batch, 256)
    z = state["z"]  # (batch, 128)
    latent = torch.cat([h, z], dim=-1)  # (batch, 384)
    output = self.net(latent)  # (batch, 3)
    
    pm25 = F.softplus(output[:, 0:1]) * 200.0 + 5.0
    energy = F.softplus(output[:, 1:2]) * 2000.0 + 100.0
    traffic = torch.sigmoid(output[:, 2:3]) * 2.0
    
    return {"pm25": pm25, "energy_mwh": energy, "traffic_index": traffic}
```

---

## Training Process

### Training Script: `training/urban_world_model.py`

#### 1. Dataset

**Synthetic Dataset** (`SyntheticUrbanDataset`):

- Generates realistic urban time-series sequences
- **Sequence Length**: 50 hours
- **Features**: PM2.5, energy, traffic with policy effects
- **Actions**: `car_free_ratio`, `renewable_mix` (vary over time)

**Data Generation**:

```python
# Policy effects
pm25 = pm25 * (1 - renewable * 0.25 - car_free * 0.30) * daily_factor
energy = energy * (1 - renewable * 0.15) * daily_factor * gdp
traffic = traffic * (1 - car_free) * daily_factor * gdp
```

#### 2. Loss Function

**Total Loss**:

```python
L_total = w_recon * L_reconstruction + w_kl * L_KL
```

**Reconstruction Loss**:

```python
L_reconstruction = MSE(predicted_obs, true_obs)
# Compares predicted [pm25, energy, traffic] to ground truth
```

**KL Divergence Loss**:

```python
L_KL = KL(q(z_t | h_t, obs_t) || p(z_t | h_t))
# Encourages posterior to match prior (regularization)
```

**KL Annealing**:

- **Start**: `kl_weight = 0.0` (focus on reconstruction)
- **End**: `kl_weight = 0.1` (gradually add regularization)
- **Strategy**: Linear from step 0 to step 100

#### 3. Training Loop

```python
for step in range(total_steps):
    # 1. Forward pass
    total_loss, recon_loss, kl_loss = model.forward(
        observations, actions,
        reconstruction_weight=1.0,
        kl_weight=kl_weight_eff  # Annealed
    )
    
    # 2. Backward pass
    total_loss.backward()
    
    # 3. Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 4. Optimizer step
    optimizer.step()
    scheduler.step()  # LR warmup + cosine decay
```

#### 4. Training Configuration (`training/configs/base.yaml`)

```yaml
encoder:
  latent_dim: 128
  input_dim: 12

rssm:
  latent_dim: 128
  hidden_dim: 256
  action_dim: 2

predictor:
  latent_dim: 128
  hidden_dim: 256
  output_dim: 3

training:
  steps: 500
  batch_size: 32
  sequence_length: 50
  learning_rate: 0.0003
  reconstruction_weight: 1.0
  kl_weight: 0.1
  kl_annealing:
    strategy: linear
    start_step: 0
    end_step: 100
    start_value: 0.0
    end_value: 0.1
  optimizer:
    name: adamw
    betas: [0.9, 0.999]
    weight_decay: 1e-4
  lr_scheduler:
    name: linear_warmup_cosine
    warmup_steps: 50
    min_lr: 1e-6
  grad_clip:
    type: norm
    value: 1.0
```

#### 5. Training Techniques

**a) KL Annealing**:

- Prevents posterior collapse (posterior = prior too early)
- Gradually increases KL weight from 0.0 to 0.1

**b) Gradient Clipping**:

- Prevents exploding gradients
- Clips gradient norm to 1.0

**c) Learning Rate Schedule**:

- **Warmup**: Linear from 0 to `lr` over 50 steps
- **Decay**: Cosine from `lr` to `min_lr` (1e-6)

**d) Early Stopping**:

- Monitors validation loss
- Stops if no improvement for 5 consecutive validations

**e) Checkpointing**:

- **Periodic**: Every 100 steps
- **Best**: Saves best validation checkpoint
- **Final**: Saves at end of training

---

## Inference Process

### Inference Flow

```bash
1. API Request (POST /api/simulate)
   ↓
2. Load Initial State (from ETL data)
   ↓
3. Preprocess (normalize observations)
   ↓
4. Encode Initial Observation
   ↓
5. RSSM Rollout (imagination)
   ↓
6. Predictor Decode
   ↓
7. Postprocess (denormalize)
   ↓
8. Return Predictions
```

### Code Path: `backend/app/models/model_wrapper.py`

#### 1. Preprocessing

```python
def preprocess(self, raw_data: Dict[str, Any]) -> torch.Tensor:
    """Normalize raw observations"""
    # Normalize PM2.5, energy, traffic
    pm25_n = (pm25 - 50.0) / 20.0
    energy_n = (energy - 1000.0) / 200.0
    traffic_n = (traffic - 1.0) / 0.3
    
    # Encode time features
    hour_sin = sin(2π * hour / 24)
    hour_cos = cos(2π * hour / 24)
    
    # Assemble feature vector (12 dims)
    features = [pm25_n, energy_n, traffic_n, car_free, renewable, ...]
    return torch.tensor(features).unsqueeze(0)  # (1, 12)
```

#### 2. Model Forward Pass

```python
def predict(self, initial_state, policy, horizon):
    # 1. Preprocess
    initial_obs = self.preprocess({**initial_state, **policy})
    
    # 2. Create action sequence
    actions = torch.zeros(1, horizon, 2)
    actions[0, :, 0] = policy["car_free_ratio"]
    actions[0, :, 1] = policy["renewable_mix"]
    
    # 3. Model forward
    with torch.no_grad():
        predictions = self.model.forward(initial_obs, actions, horizon)
    
    # predictions: (1, horizon, 3) → (horizon, 3)
    return predictions[0].detach().cpu().numpy()
```

#### 3. World Model Forward (`backend/app/models/world_model.py`)

```python
def forward(self, initial_obs, actions, horizon):
    # 1. Encode initial observation
    initial_z = self.encoder(initial_obs)  # (1, 128)
    
    # 2. Initialize RSSM state
    state = self.rssm.init_state(batch_size=1)
    state["z"] = initial_z
    state["h"] = state["h"].to(device)
    
    # 3. Rollout for horizon steps
    predictions = []
    for t in range(horizon):
        action_t = actions[:, t]  # (1, 2)
        
        # RSSM forward (NO observation = uses PRIOR)
        state = self.rssm.forward(state, action_t, observation=None)
        
        # Predictor decode
        pred = self.predictor(state)
        pred_tensor = torch.cat([pred["pm25"], pred["energy_mwh"], pred["traffic_index"]], dim=-1)
        predictions.append(pred_tensor)
    
    # Stack: (1, horizon, 3)
    return torch.stack(predictions, dim=1)
```

#### 4. Latent State Extraction (for Explainability)

```python
def extract_latent_states(self, initial_state, policy, horizon=10):
    """Extract z and h states for visualization"""
    # ... encode and rollout ...
    
    latent_states_z = [state["z"].detach().cpu().numpy()]
    latent_states_h = [state["h"].detach().cpu().numpy()]
    
    for t in range(horizon):
        state = self.rssm.forward(state, action_t, observation=None)
        latent_states_z.append(state["z"].detach().cpu().numpy())
        latent_states_h.append(state["h"].detach().cpu().numpy())
    
    return {
        "z": np.vstack(latent_states_z),  # (horizon+1, 128)
        "h": np.vstack(latent_states_h),  # (horizon+1, 256)
    }
```

---

## Code Locations

### Training Code

| Component | File | Key Class/Function |
|-----------|------|-------------------|
| **Main Model** | `training/urban_world_model.py` | `UrbanWorldModel` |
| **Encoder** | `training/modules/encoder.py` | `Encoder` |
| **RSSM** | `training/modules/rssm.py` | `RSSM` |
| **Predictor** | `training/modules/predictor.py` | `Predictor` |
| **Training Loop** | `training/urban_world_model.py` | `train_loop()` |
| **Dataset** | `training/urban_world_model.py` | `SyntheticUrbanDataset` |
| **Config** | `training/configs/base.yaml` | YAML config |

### Inference Code

| Component | File | Key Class/Function |
|-----------|------|-------------------|
| **Model Wrapper** | `backend/app/models/model_wrapper.py` | `ModelWrapper` |
| **World Model** | `backend/app/models/world_model.py` | `UrbanWorldModel` |
| **API Endpoint** | `backend/app/api/simulate.py` | `simulate()` |
| **Data Loader** | `backend/app/data/loader.py` | `load_recent_pm25_mean()` |
| **Explainability** | `backend/app/api/explain.py` | `get_latent_sample()` |

---

## How It All Works Together

### Training → Inference Pipeline

```bash
┌─────────────────────────────────────────────────────────────┐
│ TRAINING PHASE                                              │
│                                                             │
│ 1. Generate/load sequences (50 hours each)                  │
│ 2. Forward pass: Encode → RSSM → Predictor                │
│ 3. Compute loss: Reconstruction + KL divergence           │
│ 4. Backward pass: Update encoder, RSSM, predictor weights  │
│ 5. Save checkpoint (encoder.pth, rssm.pth, predictor.pth) │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ CHECKPOINT SAVED                                            │
│ - model_best.json (metadata)                                │
│ - model_best.pth (weights)                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ INFERENCE PHASE                                             │
│                                                             │
│ 1. Load checkpoint (ModelWrapper.load())                    │
│ 2. Get initial state from ETL (load_recent_pm25_mean())     │
│ 3. Preprocess: Normalize observations                      │
│ 4. Encode: initial_obs → z_0                               │
│ 5. RSSM Rollout: z_0, h_0 → z_1, h_1 → ... → z_H, h_H     │
│    (uses PRIOR, not posterior - no observations)           │
│ 6. Predictor: [h_t, z_t] → [pm25, energy, traffic]        │
│ 7. Postprocess: Denormalize predictions                     │
│ 8. Return: List of predictions for each hour               │
└─────────────────────────────────────────────────────────────┘
```

### Example: Predicting 48 Hours Ahead

```python
# 1. Initial state (from ETL)
initial_state = {
    "pm25": 34.0,        # Real-time from WAQI
    "energy_mwh": 1200.0,
    "traffic_index": 1.0
}

# 2. Policy
policy = {
    "car_free_ratio": 0.3,   # 30% car-free
    "renewable_mix": 0.5      # 50% renewable
}

# 3. Model inference
predictions = model.predict(initial_state, policy, horizon=48)
# predictions: (48, 3) array
# [
#   [35.2, 1180.5, 0.7],  # Hour 0
#   [36.1, 1165.3, 0.6],  # Hour 1
#   ...
#   [42.5, 1100.2, 0.5],  # Hour 47
# ]
```

### Key Differences: Training vs. Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Observations** | Full sequence (50 hours) | Only initial (1 hour) |
| **RSSM Mode** | Uses **POSTERIOR** (with obs) | Uses **PRIOR** (no obs) |
| **Purpose** | Learn dynamics | Predict future |
| **Gradients** | Computed (backward pass) | Disabled (`torch.no_grad()`) |
| **Mode** | `model.train()` | `model.eval()` |

---

## Summary

### What Makes This DreamerV3-Style?

1. ✅ **Latent World Model**: Works in compressed latent space (128 dims)
2. ✅ **RSSM Architecture**: Deterministic (h) + Stochastic (z) states
3. ✅ **Prior/Posterior**: Predicts latent state with/without observations
4. ✅ **Reconstruction Loss**: Learns to decode latent → observations
5. ✅ **KL Regularization**: Encourages posterior ≈ prior
6. ✅ **Imagination Rollout**: Can predict future without observations

### Training Strategy

- **KL Annealing**: Gradually increase KL weight (0.0 → 0.1)
- **Gradient Clipping**: Prevent exploding gradients (norm = 1.0)
- **LR Schedule**: Warmup (50 steps) + Cosine decay
- **Early Stopping**: Stop if validation loss doesn't improve
- **Checkpointing**: Save best + periodic checkpoints

### Inference Strategy

- **Real-Time Data**: Uses `real_time_pm25` from WAQI (not old forecasts)
- **Policy Simulation**: Applies policy actions (car_free, renewable) over horizon
- **Latent Extraction**: Can extract z/h states for explainability
- **Baseline Fallback**: Falls back to deterministic baseline if model unavailable

---

## References

- **DreamerV3 Paper**: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- **Codebase**: See `training/modules/` for implementation
- **Config**: `training/configs/base.yaml` for hyperparameters
- **Training**: `training/urban_world_model.py` for training loop
- **Inference**: `backend/app/models/model_wrapper.py` for inference wrapper
