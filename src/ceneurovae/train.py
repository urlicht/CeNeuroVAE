import torch
import torch.nn as nn
import time
from .model.loss import step_kl_schedule
from .optimizer import current_lr

def train_epoch(model, loader, optim, device, grad_clip=1.0):
  model.train()
  loss_rec = loss_kl = loss_sum = n = 0

  for X, M, Bx, I, _ in loader:
    X, M, Bx, I = X.to(device), M.to(device), Bx.to(device), I.to(device)
    out = model(X, M, Bx, I)
    loss = out["loss_sum"]
    optim.zero_grad(set_to_none=True)
    loss.backward()

    if grad_clip is not None and grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # prevent gradient explosion
    optim.step()

    loss_rec += float(out["loss_rec"])
    loss_kl  += float(out["loss_kl"])
    loss_sum += float(loss)
    n += 1

  return {"train_rec": loss_rec / n, "train_kl": loss_kl / n, "train_sum": loss_sum / n}

def val_epoch(model, loader, device):
  model.eval()
  loss_rec = loss_kl = loss_sum = n = 0
  with torch.no_grad():
    for X, M, Bx, I, _ in loader:
      X, M, Bx, I = X.to(device), M.to(device), Bx.to(device), I.to(device)
      out = model(X, M, Bx, I)

      loss_rec += out["loss_rec"].item()
      loss_kl += out["loss_kl"].item()
      loss_sum += out["loss_sum"].item()
      n += 1

  return {"val_rec": loss_rec / n, "val_kl": loss_kl /n , "val_sum": loss_sum / n}

def fit_model(model, loader_train, loader_val, n_epoch, optim, scheduler_seq, scheduler_plateau, cosine_finished_fn, device):
    best_loss_val = float("inf")
    list_loss, list_lr = [], []

    for epoch in range(1, n_epoch + 1):
        t1 = time.time_ns()

        # --- training, validation ---
        train_ = train_epoch(model, loader_train, optim, device)
        val_   = val_epoch(model, loader_val, device)

        # any KL anneal etc.
        step_kl_schedule(model, epoch, n_epoch)

        # --- schedulers (per-epoch) ---
        # Step the sequential warmup+cosine every epoch (AFTER optimizer stepping has occurred in train).
        scheduler_seq.step()

        # If we reserved a tail and have a plateau scheduler, step it only AFTER the cosine has finished.
        if scheduler_plateau is not None and cosine_finished_fn(epoch):
            # Step on a validation metric (use what you actually optimize; here val_sum).
            scheduler_plateau.step(val_["val_sum"])

        t2 = time.time_ns()
        lr_ = current_lr(optim)
        print(f"{epoch} {((t2-t1)/1e9):.2f}s  "
              f"train: {train_['train_sum']:.4f}  val: {val_['val_sum']:.4f}  "
              f"kl_train: {train_['train_kl']:.4f}  kl_val: {val_['val_kl']:.4f}  lr: {lr_}")

        list_loss.append((train_, val_))
        list_lr.append(lr_)

    return list_loss, list_lr