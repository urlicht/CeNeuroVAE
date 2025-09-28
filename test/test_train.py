import torch
from ceneurovae.model import VAEConfig, NeuroBehaviorVAE
from ceneurovae.data import build_loaders
from ceneurovae.optimizer import build_optimizer, build_scheduler
from ceneurovae.train import train_epoch, val_epoch
from ceneurovae.train import fit_model

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

def test_fit_model_run():
    # tiny config and tiny dataset
    cfg = VAEConfig(n_identities=4, behavior_dim=1, neuron_token_dim=8, model_dim=32, decoder_hidden=16,
                    latent_dim=4, n_layers=1, n_heads=2)
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
    scheduler_seq, scheduler_plateau, cosine_finished_fn = build_scheduler(optim, 10, warmup_epochs=2)

    history, best_epoch, best_state = fit_model(
        model=model,
        loader_train=loader_train,
        loader_val=loader_val,
        n_epoch=10,
        optim=optim,
        scheduler_seq=scheduler_seq,
        scheduler_plateau=scheduler_plateau,
        cosine_finished_fn=cosine_finished_fn,
        device="cpu",
        early_stop_patience=5,
        early_stop_min_delta=0.01,
        path_best=None,
        grad_clip=1.0,
    )

    assert isinstance(history, list)
    assert len(history) > 0
    assert "epoch" in history[0]
    assert "train_sum" in history[0]
    assert "val_sum" in history[0]
    assert best_epoch > 0
    assert best_state is not None
