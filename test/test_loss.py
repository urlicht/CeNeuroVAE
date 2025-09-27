import torch
from ceneurovae.model.loss import loss_mse, loss_huber

def test_loss_mse_masking():
    # simple example: two timesteps, two neurons
    recon = torch.tensor([[[1.0, 2.0],[3.0, 4.0]]])  # (B=1, N=2, T=2)
    X     = torch.tensor([[[0.0, 2.0],[3.0, 0.0]]])
    M     = torch.tensor([[[1.0, 1.0],[1.0, 0.0]]])  # mask out one entry
    loss = loss_mse(recon, X, M)
    # compute by hand: squared errors at masked positions: (1^2 + 0^2 + 0^2) = 1; divide by masked count 3 -> 1/3
    assert abs(loss.item() - (1.0/3.0)) < 1e-6

def test_loss_huber_basic():
    recon = torch.tensor([[[0.0]]])
    X     = torch.tensor([[[2.0]]])
    M     = torch.tensor([[[1.0]]])
    loss = loss_huber(recon, X, M)
    # with delta = 1.0 and diff = -2.0 -> huber = delta*(abs(diff)-0.5*delta) = 1*(2-0.5)=1.5
    assert abs(loss.item() - 1.5) < 1e-6