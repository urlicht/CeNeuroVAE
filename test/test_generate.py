import torch
from ceneurovae.model import VAEConfig, NeuroBehaviorVAE
from ceneurovae.pred import get_full_sequence_reconstruction, get_full_sequence_latent

def test_sliding_window_reconstruction_and_latent():
    cfg = VAEConfig(n_identities=4, behavior_dim=1, neuron_token_dim=8, model_dim=32, decoder_hidden=16, latent_dim=6, n_layers=1, n_heads=2)
    model = NeuroBehaviorVAE(cfg).to("cpu")
    model.eval()
    B = 1; N = 3; T = 50; Tb = 1
    X = torch.randn(B, N, T)
    M = torch.ones(B, N, T)
    Bx = torch.randn(B, Tb, T)
    I = torch.randint(0, cfg.n_identities, (B, N), dtype=torch.long)
    recon = get_full_sequence_reconstruction(model, X, M, Bx, I, win=20, hop=10, device="cpu")
    assert recon.shape == X.shape
    z_full = get_full_sequence_latent(model, X, M, Bx, I, win=20, hop=10, device="cpu")
    assert z_full.shape == (B, T, cfg.latent_dim)