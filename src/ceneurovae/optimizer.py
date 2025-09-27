import torch

def build_optimizer(model, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.95)):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

def build_scheduler(
    optimizer,
    total_epochs: int,
    warmup_epochs: int = 10,
    min_lr_ratio: float = 0.1,
    use_plateau: bool = True,
    cosine_portion: float = 1.0, # 1.0 = cosine spans the entire post-warmup phase.
):
    """
    schedulers components (per-epoch stepping):
      1) linear warmup from lr*1e-3 -> lr over `warmup_epochs`
      2) CosineAnnealingLR down to base_lr * min_lr_ratio over `cosine_steps`
      3) (optional) ReduceLROnPlateau, AFTER the cosine ends
    """
    assert 0 < cosine_portion <= 1.0, "cosine_portion must be in (0, 1]."
    assert total_epochs > 0 and warmup_epochs >= 0 and total_epochs > warmup_epochs

    # Capture base_lrs BEFORE any scheduler mutates them
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    base_lr_min = min(base_lrs)

    # Split remaining epochs into cosine phase and (optional) tail
    remaining = total_epochs - warmup_epochs
    cosine_steps = max(1, int(round(remaining * cosine_portion)))
    tail_epochs = max(0, remaining - cosine_steps)

    scheds, milestones = [], []

    if warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        scheds.append(warm)
        milestones.append(warmup_epochs)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,                        # <- exact cosine span (per-epoch)
        eta_min=base_lr_min * min_lr_ratio
    )
    scheds.append(cosine)

    seq = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=scheds, milestones=milestones
    )

    plateau = None
    if use_plateau and tail_epochs > 0:
        # Plateau only meaningful if you left a tail (cosine_portion < 1.0).
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            threshold=1e-3,
            cooldown=0,
            min_lr=base_lr_min * 0.02,
        )

    # Helper: tell the caller when cosine is done so they can step plateau.
    def cosine_finished(epoch_idx_1based: int) -> bool:
        # Warmup uses [1 .. warmup_epochs]
        # Cosine uses the next `cosine_steps` epochs: (warmup_epochs+1) .. (warmup_epochs+cosine_steps)
        return epoch_idx_1based > (warmup_epochs + cosine_steps)

    return seq, plateau, cosine_finished

def current_lr(optimizer):
    # handling for multiple param groups, otherwise return a singel value in float
    lrs = [g["lr"] for g in optimizer.param_groups]
    return lrs[0] if len(lrs) == 1 else lrs