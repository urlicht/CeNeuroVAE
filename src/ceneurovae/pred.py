import torch
import torch.nn as nn

@torch.no_grad()
def get_full_sequence_reconstruction(model, X, M, Bx, I, win=200, hop=100, device='cpu'):
    """
    Run model in sliding windows and stitch together reconstruction for the full sequence.
    
    Args:
        model: MaskedVAE
        X:  (1, N, T) neural traces
        M:  (1, N, T) mask
        Bx: (1, Tb, T) behavior
        I:  (1, N) neural identities
        win: window length
        hop: hop length
        device: 'cuda'/'cpu'/'mps'

    Returns:
        recon_full: (1, N, T) averaged reconstruction
    """
    model.eval()
    X = X.to(device); M = M.to(device); Bx = Bx.to(device); I = I.to(device)

    B, N, T = X.shape
    recon_full = torch.zeros_like(X)
    weight     = torch.zeros_like(M)  # to average overlaps

    for t0 in range(0, T, hop):
        t1 = min(t0 + win, T)
        # slice
        Xw  = X[:, :, t0:t1]
        Mw  = M[:, :, t0:t1]
        Bxw = Bx[:, :, t0:t1]
        # pad window tail to win if needed (optional)
        if t1 - t0 < win:
            pad_t = win - (t1 - t0)
            Xw  = nn.functional.pad(Xw,  (0,pad_t))
            Mw  = nn.functional.pad(Mw,  (0,pad_t))
            Bxw = nn.functional.pad(Bxw, (0,pad_t))

        out = model(Xw, Mw, Bxw, I)
        rw  = out["reconstruction"][:, :, :t1-t0]   # unpad back to real length

        recon_full[:, :, t0:t1] += rw
        weight[:, :, t0:t1]     += 1.0

    recon_full = recon_full / weight.clamp_min(1.0)

    return recon_full

@torch.no_grad()
def get_full_sequence_latent(model, X, M, Bx, I, win=200, hop=100, device='cuda'):
    """
    Run model in sliding windows and stitch together latent z for the full sequence.

    Args:
      model: MaskedVAE
      X:  (1, N, T) neural traces
      M:  (1, N, T) mask
      Bx: (1, Tb, T) behavior
      I:  (1, N) identities
      win: window length
      hop: hop length
      device: 'cuda'/'cpu'/'mps'

    Returns:
      z_full: (1, T, L) averaged latent trajectory
              (same as model output["z"], stitched)
    """
    model.eval()
    X, M, Bx, I = X.to(device), M.to(device), Bx.to(device), I.to(device)

    B, N, T = X.shape
    L = model.cfg.latent_dim

    z_full = torch.zeros((B, T, L), device=device)
    weight = torch.zeros((B, T, 1), device=device) # count overlaps per timestep

    for t0 in range(0, T, hop):
        t1 = min(t0 + win, T)

        # slice
        Xw  = X[:, :, t0:t1]
        Mw  = M[:, :, t0:t1]
        Bxw = Bx[:, :, t0:t1]

        # pad if needed
        if t1 - t0 < win:
            pad_t = win - (t1 - t0)
            Xw  = nn.functional.pad(Xw,  (0, pad_t))
            Mw  = nn.functional.pad(Mw,  (0, pad_t))
            Bxw = nn.functional.pad(Bxw, (0, pad_t))

        out = model(Xw, Mw, Bxw, I) # dict with "z": (B, win, L)

        # drop padded tail
        zw = out["z"][:, :t1 - t0, :] # (B, seg_len, L)

        z_full[:, t0:t1, :] += zw
        weight[:, t0:t1, :] += 1.0

    z_full = z_full / weight.clamp_min(1.0)

    return z_full

