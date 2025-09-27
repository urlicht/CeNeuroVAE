import torch

def loss_mse(recon, X, M):
    """
    Mean Squared Error (MSE) loss with masking.

    recon: (B, N, T) reconstructed data
    X: (B, N, T) original data
    M: (B, N, T) mask (1 for valid, 0 for invalid)
    """
    mse = (recon - X) ** 2
    mse = mse * M # masking
    loss_rec = mse.sum() / (M.sum().clamp_min(1.0))

    return loss_rec

def loss_huber(recon, X, M):
    """
    Huber loss with masking.

    recon: (B, N, T) reconstructed data
    X: (B, N, T) original data
    M: (B, N, T) mask (1 for valid, 0 for invalid)
    """
    diff = (recon - X) * M
    delta = 1.0  # tune 0.5–2.0
    huber = torch.where(diff.abs() <= delta, 0.5 * diff * diff, delta*(diff.abs() - 0.5 * delta))
    loss_rec = huber.sum() / M.sum().clamp_min(1.0)

    return loss_rec

def step_kl_schedule(model, epoch, total_epochs):
    # free-bits: 0 for first 10% epochs, then 0.01
    model.tau_freebits = 0.0 if epoch < max(3, int(0.1*total_epochs)) else 0.01

    # KL warmup
    beta_max = model.cfg.beta_kl
    warm = int(0.3 * total_epochs)
    model.cfg.beta_kl = beta_max * min(1.0, epoch / max(1, warm))