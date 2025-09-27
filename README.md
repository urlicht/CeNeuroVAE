# CeNeuroVAE

ceneurovae is a compact Variational Autoencoder (VAE) package tailored for neural time-series conditioned upon behaviors. It provides modular components (encoder, decoder, positional encoding, losses, data loaders) and a small training loop function to quickly test it out.

The model was tested out with the labeled datasets from [Atanas & Kim et. al. 2023](https://doi.org/10.1016/j.cell.2023.07.035). These datasets contain simultaneously recorded whole-brain neural traces and behavioral info such as velocity, feeding, and head curvature. While the model was tested on these C. elegans datasets, you can easily use this model/package to fit neural and behavioral datasets from any model systems, as long as most neurons have known neural identities.
---

## Table of contents

- Overview
- Package layout
- Key concepts & model overview
- Public API (typical names)
- Important variables and key tensor dimensions
- Example usage
- Training / hyperparameters
- Data & IO
- Contributing & testing
- License

---

## Overview

ceneurovae implements a configurable VAE for sequential data. Core ideas:

- Encoder compresses input sequences to a low-dimensional latent vector z.
- A probabilistic latent prior (N(0, I)) and a learned posterior q(z|x) are used.
- Decoder reconstructs sequences from z (optionally conditioned with positional encodings).
- Loss is reconstruction term (MSE / Huber) + KL divergence; KL weight can be scheduled to prevent posterior collapse.

---

## Package layout

(Inside `src/ceneurovae/`)

- `model/` — VAE and supporting utilities
  - `vae.py` — top-level VAE wiring (encoder, reparam, decoder)
  - `utility.py` — config dataclasses, positional encodings, helpers
  - `loss.py` — reconstruction losses and KL schedules
- `import_data.py` (or `_import_data.py`) — loaders & dataset helpers
- `training.py` — training loop, checkpointing, metrics
- `utils.py` — small helpers, logging, metrics

---

## Key concepts & model overview

Model components:

- Encoder
  - Maps input sequences to parameters of a Gaussian posterior (mu, logvar).
  - Typical backbone: 1D conv / temporal CNN, or stacked RNN / Transformer blocks.
  - Output: posterior mean `mu` and log-variance `logvar` of shape `(batch, latent_dim)`.

- Reparameterization
  - Sample z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I).

- Decoder
  - Conditioned on z (optionally tiled across time or combined with positional encodings).
  - Produces reconstructed sequence of same shape as input `(batch, seq_len, channels)`.

- PositionalEncoding
  - Adds explicit time location information to decoder inputs.
  - Shape: `(seq_len, embed_dim)` or broadcastable to `(batch, seq_len, embed_dim)`.

- Losses
  - Reconstruction: MSE or Huber on raw or normalized signals.
  - KL divergence between q(z|x) and p(z) (usually standard normal).
  - Annealing / scheduling: linearly increase KL weight, or step schedule.

---

## Important variables and key dimensions

Below are the canonical names and example default values. Adapt to your code.

- Data / input
  - batch size: `B`
  - sequence length (timesteps): `T` (e.g., 100, 250)
  - channels / features per timestep: `C` (e.g., 1 for a single trace, or multi-channel)
  - input tensor shape: `(B, T, C)`

- Encoder
  - hidden dimension(s): `H` (per layer, e.g., 128, 256)
  - number of layers: `L_enc` (e.g., 1-4)
  - encoder output (posterior params): `mu`, `logvar` shape `(B, Z)`

- Latent space
  - latent dimension: `Z` (e.g., 8, 16, 32)
  - latent tensor shape: `(B, Z)`

- Decoder
  - may receive `z` expanded to `(B, T, Z)` (tile) or combined via conditioning layers
  - decoder hidden dims: `H_dec`
  - reconstructed output shape: `(B, T, C)`

- Loss & training
  - reconstruction loss per example: sum / mean over `(T, C)`
  - KL per example: `0.5 * sum(1 + logvar - mu^2 - exp(logvar))` (sum over latent dims)
  - kl_weight (beta): scalar schedule (0 -> 1 typical)

Typical defaults to try:
- T = 100, C = 3, Z = 32, H = 256, batch = 32, learning rate = 1e-3

Shape summary (single forward example):
- input: x (B, T, C)
- encoder outputs: mu, logvar (B, Z)
- sampled: z (B, Z)
- decoder recon: x_hat (B, T, C)

---

## Example usage
Example usage:
```python
import torch
import torch.nn as nn

from ceneurovae.model import VAEConfig, NeuroBehaviorVAE
from ceneurovae.import_data import import_h5
from ceneurovae.data import build_loaders
from ceneurovae.optimizer import build_optimizer, build_scheduler
from ceneurovae.train import fit_model

path_data = "..."
datasets, labels = import_h5(path_data)

device = torch.device("cuda")
cfg = VAEConfig(n_identities=155, behavior_dim=3)
model = NeuroBehaviorVAE(cfg).to(device)

n_epoch = 100

optim = build_optimizer(model, lr=1e-3)

scheduler_seq, scheduler_plat, cosine_done = build_scheduler(
  optim, total_epochs=n_epoch, warmup_epochs=10,
  min_lr_ratio=0.05, use_plateau=True,
  cosine_portion=0.85
)

loader_train, loader_val, train_uids, val_uids = build_loaders(datasets, window_T=100,
                                                               stride=50, batch_size=8, num_workers=0)

list_loss, list_lr = fit_model(model, loader_train, loader_val, n_epoch, optim,
                               scheduler_seq, scheduler_plat, cosine_done, device)
```
---

## Training details & hyperparameters

Recommended checkpoints to experiment with:
- Reconstruction loss: MSE vs Huber (Huber more robust to outliers)
- KL schedule: start KL weight small and gradually increase to 1 over warmup steps
- Latent dim: larger values allow more capacity, but increase risk of posterior collapse
- Regularization: dropout, weight decay

Metrics to monitor:
- Validation reconstruction error (MSE / Huber)
- KL magnitude (per latent dim)
- ELBO (reconstruction + KL)
- Sampled reconstructions and latent traversals (visual inspection)

---

## Data & IO

- Data loaders should yield tensors shaped `(B, T, C)` and optionally masks for missing data.
- Normalization: normalize per-channel (mean/STD) during preprocessing; store scalers with checkpoints.
- For reproducibility, log config (VAEConfig) and training hyperparameters with each run.

---