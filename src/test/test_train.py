import torch
from ceneurovae.model import VAEConfig, NeuroBehaviorVAE
from ceneurovae.data import build_loaders
from ceneurovae.optimizer import build_optimizer
from ceneurovae.train import train_epoch, val_epoch

def test_train_and_val_epoch_run():
    # tiny config and tiny dataset
    cfg = VAEConfig(n_identities=4, behavior_dim=1, neuron_token_dim=8, model_dim=32, decoder_hidden=16, latent_dim=4, n_layers=1, n_heads=2)
    model = NeuroBehaviorVAE(cfg).to("cpu")
    # create synthetic datasets dict expected by build_loaders
    datasets = {}
    for i in range(2):
        uid = f"ds{i}"
        N, T, Tb = 3, 20, 1
        X = torch.randn(N, T).numpy().astype("float32")
        M = torch.ones(N, T).numpy().astype("float32")
        Bx = torch.randn(Tb, T).numpy().astype("float32")
        I = (torch.arange(N) % 3).numpy().astype("int64")
        datasets[uid] = {"X": X, "M": M, "Bx": Bx, "I": I, "uid": uid}
    loader_train, loader_val, _, _ = build_loaders(datasets, window_T=8, stride=8, batch_size=2, num_workers=0)
    optim = build_optimizer(model, lr=1e-3)
    train_stats = train_epoch(model, loader_train, optim, device="cpu")
    val_stats = val_epoch(model, loader_val, device="cpu")
    for k in ("train_rec", "train_kl", "train_sum"):
        assert k in train_stats
    for k in ("val_rec", "val_kl", "val_sum"):
        assert k in val_stats