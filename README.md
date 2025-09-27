# CeNeuroVAE

`ceneurovae` is a compact Variational Autoencoder (VAE) package tailored for neural/behavioral time-series. It provides modular components (encoder, decoder, positional encoding, losses, data loaders) and a small training harness so you can prototype and evaluate VAE variants quickly.

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
- A probabilistic latent prior (usually N(0, I)) and a learned posterior q(z|x) are used.
- Decoder reconstructs sequences from z (optionally conditioned with positional encodings).
- Loss is reconstruction term (MSE / Huber) + KL divergence; KL weight can be scheduled.

---

## Package layout

(Inside `src/ceneurovae/`)

- `__init__.py` — package-level re-exports / public API surface
- `model/` — VAE and supporting utilities
  - `vae.py` — top-level VAE wiring (encoder, reparam, decoder)
  - `utility.py` — config dataclasses, positional encodings, helpers
  - `loss.py` — reconstruction losses and KL schedules
  - `__init__.py` — re-exports public model symbols
- `import_data.py` (or `_import_data.py`) — loaders & dataset helpers
- `training.py` — training loop, checkpointing, metrics
- `utils.py` — small helpers, logging, metrics
- `cli.py` / `__main__.py` — optional CLI entrypoint

Expose the stable API from the package root (e.g. `from ceneurovae import NeuroBehaviorVAE, train`).

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

- PositionalEncoding (optional)
  - Adds explicit time location information to decoder inputs.
  - Shape: `(seq_len, embed_dim)` or broadcastable to `(batch, seq_len, embed_dim)`.

- Losses
  - Reconstruction: MSE or Huber on raw or normalized signals.
  - KL divergence between q(z|x) and p(z) (usually standard normal).
  - Annealing / scheduling: linearly increase KL weight, or step schedule.

---

## Public API (typical symbols)

Examples of names commonly re-exported at package root (replace with your actual names):

- `VAEConfig` — dataclass with model hyperparameters
- `NeuroBehaviorVAE` (or `VAE`) — the model class
- `train` — training loop helper
- `import_h5`, `split_datasets` — data loader helpers
- loss helpers: `loss_mse`, `loss_huber`, `step_kl_schedule`

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

Replace symbol names with actual exported names from your package.

```python
from ceneurovae import VAEConfig, NeuroBehaviorVAE, train, import_h5

# load data (example)
train_ds, valid_ds = import_h5("data.h5")

# model config
cfg = VAEConfig(
    seq_len=100,
    channels=3,
    latent_dim=32,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1,
)

model = NeuroBehaviorVAE(cfg)

# single forward
x = next(iter(train_ds))  # tensor shape (B, T, C)
recon, mu, logvar = model(x)  # recon: (B, T, C), mu/logvar: (B, Z)

# training (high-level)
train(model, train_ds, valid_ds,
      epochs=100,
      lr=1e-3,
      batch_size=32,
      kl_schedule="linear",  # or step schedule
)
```

If your package re-exports lazily, it's also valid to import submodules directly:
```python
import ceneurovae.model as model_pkg
model = model_pkg.NeuroBehaviorVAE(cfg)
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

## Testing & development

- Unit tests should cover: forward shapes, reparameterization correctness, loss numerics (KL non-negative), and data loaders.
- Add small integration tests: train for 1-2 epochs on a tiny synthetic dataset and assert losses decrease.

---

## Contributing

- Follow existing code style
- Add tests for new features
- Open PRs with clear descriptions and reproducible examples

---

## License

Add your preferred license here (e.g., MIT). If none provided, specify project license file.

---

If you want, I can:
- generate a README filled with exact class/function names found in your codebase,
- or directly write the file into `/Users/jungsookim/dev/py/ceneurovae/README.md`.

Which do you prefer?# filepath: /Users/jungsookim/dev/py/ceneurovae/README.md
# ...existing code...

# ceneurovae

ceneurovae is a compact Variational Autoencoder (VAE) package tailored for neural/behavioral time-series. It provides modular components (encoder, decoder, positional encoding, losses, data loaders) and a small training harness so you can prototype and evaluate VAE variants quickly.

This README explains package layout, model components, important configuration variables, and the shapes/dimensions you need to be aware of.

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
- A probabilistic latent prior (usually N(0, I)) and a learned posterior q(z|x) are used.
- Decoder reconstructs sequences from z (optionally conditioned with positional encodings).
- Loss is reconstruction term (MSE / Huber) + KL divergence; KL weight can be scheduled.

---

## Package layout

(Inside `src/ceneurovae/`)

- `__init__.py` — package-level re-exports / public API surface
- `model/` — VAE and supporting utilities
  - `vae.py` — top-level VAE wiring (encoder, reparam, decoder)
  - `utility.py` — config dataclasses, positional encodings, helpers
  - `loss.py` — reconstruction losses and KL schedules
  - `__init__.py` — re-exports public model symbols
- `import_data.py` (or `_import_data.py`) — loaders & dataset helpers
- `training.py` — training loop, checkpointing, metrics
- `utils.py` — small helpers, logging, metrics
- `cli.py` / `__main__.py` — optional CLI entrypoint

Expose the stable API from the package root (e.g. `from ceneurovae import NeuroBehaviorVAE, train`).

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

- PositionalEncoding (optional)
  - Adds explicit time location information to decoder inputs.
  - Shape: `(seq_len, embed_dim)` or broadcastable to `(batch, seq_len, embed_dim)`.

- Losses
  - Reconstruction: MSE or Huber on raw or normalized signals.
  - KL divergence between q(z|x) and p(z) (usually standard normal).
  - Annealing / scheduling: linearly increase KL weight, or step schedule.

---

## Public API (typical symbols)

Examples of names commonly re-exported at package root (replace with your actual names):

- `VAEConfig` — dataclass with model hyperparameters
- `NeuroBehaviorVAE` (or `VAE`) — the model class
- `train` — training loop helper
- `import_h5`, `split_datasets` — data loader helpers
- loss helpers: `loss_mse`, `loss_huber`, `step_kl_schedule`

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

Replace symbol names with actual exported names from your package.

```python
from ceneurovae import VAEConfig, NeuroBehaviorVAE, train, import_h5

# load data (example)
train_ds, valid_ds = import_h5("data.h5")

# model config
cfg = VAEConfig(
    seq_len=100,
    channels=3,
    latent_dim=32,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1,
)

model = NeuroBehaviorVAE(cfg)

# single forward
x = next(iter(train_ds))  # tensor shape (B, T, C)
recon, mu, logvar = model(x)  # recon: (B, T, C), mu/logvar: (B, Z)

# training (high-level)
train(model, train_ds, valid_ds,
      epochs=100,
      lr=1e-3,
      batch_size=32,
      kl_schedule="linear",  # or step schedule
)
```

If your package re-exports lazily, it's also valid to import submodules directly:
```python
import ceneurovae.model as model_pkg
model = model_pkg.NeuroBehaviorVAE(cfg)
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

## Testing & development

- Unit tests should cover: forward shapes, reparameterization correctness, loss numerics (KL non-negative), and data loaders.
- Add small integration tests: train for 1-2 epochs on a tiny synthetic dataset and assert losses decrease.

---

## Contributing

- Follow existing code style
- Add tests for new features
- Open PRs with clear descriptions and reproducible examples

---

## License

Add your preferred license here (e.g., MIT). If none provided, specify project license file.

---

If you want, I can:
- generate a README filled with exact class/function names found in your codebase,
- or directly write the file into `/Users/jungsookim/dev/py/ceneurovae/README.md`.

Which do you prefer?