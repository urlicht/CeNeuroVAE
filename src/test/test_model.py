import torch
from ceneurovae.model import VAEConfig, NeuroBehaviorVAE

def make_tensors(B=2, N=3, T=10, Tb=1, device="cpu"):
    X = torch.randn(B, N, T, device=device)
    M = torch.ones(B, N, T, device=device)
    Bx = torch.randn(B, Tb, T, device=device)
    I = torch.randint(0, 5, (B, N), dtype=torch.long, device=device)
    return X, M, Bx, I

def test_vae_forward_shapes_and_keys():
    cfg = VAEConfig(n_identities=6, behavior_dim=1, neuron_token_dim=8, model_dim=32, decoder_hidden=16, latent_dim=4, n_layers=1, n_heads=2)
    model = NeuroBehaviorVAE(cfg).to("cpu")
    X, M, Bx, I = make_tensors(B=1, N=4, T=12, Tb=1)
    out = model(X, M, Bx, I)
    # keys
    for k in ("reconstruction", "mu", "z", "loss_rec", "loss_kl", "loss_sum"):
        assert k in out
    # shapes
    assert out["reconstruction"].shape == X.shape
    assert out["mu"].shape[0:2] == (1, 12)
    assert out["z"].shape[0:2] == (1, 12)