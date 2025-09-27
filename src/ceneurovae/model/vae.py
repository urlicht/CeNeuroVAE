import torch
import torch.nn as nn
from utility import VAEConfig, PositionalEncoding
from loss import loss_mse, loss_huber

class NeuroBehaviorVAE(nn.Module):
  """
  behavior-conditioned, masked VAE on neurons

  input tensors:
  X: neural signals. (B, N, T)
  M: mask for observations. (B, N, T) {0, 1}
  Bx: behaviors. (B, Tb, T)
  I: identity info per neuron. 0 for unknown. (B, N), long
  """

  def __init__(self, cfg: VAEConfig):
    super().__init__()
    self.cfg = cfg

    # bias and gain
    self.id_to_affine = nn.Linear(cfg.neuron_embed_dim, 2)  # -> (gain, bias)

    # reconstruction loss
    match getattr(self.cfg, "loss_rec_type", "mse"):
        case "mse":
            self.f_loss_rec = loss_mse
        case "huber":
            self.f_loss_rec = loss_huber

    # neuron embedding
    self.neuron_scalar_mlp = nn.Sequential(
        nn.Linear(1, cfg.neuron_token_dim),
        nn.GELU(),
        nn.Linear(cfg.neuron_token_dim, cfg.neuron_token_dim),
        nn.GELU()
    )

    # behavior projection
    self.beh_width = max(32, cfg.model_dim // 16)
    self.beh_proj = nn.Sequential(
        nn.Linear(cfg.behavior_dim, self.beh_width), # reduce behavior influence
        nn.GELU()
    )

    # combine pooled neurons and behaviors tokens
    self.combined_proj = nn.Sequential(
        nn.Linear(cfg.neuron_token_dim + self.beh_width, cfg.model_dim),
        nn.GELU(),
        nn.Dropout(cfg.dropout)
    )

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=cfg.model_dim,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.model_dim * 4,
        dropout=cfg.dropout,
        batch_first=True,
        activation="gelu"
    )

    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
    self.pos_encoder = PositionalEncoding(cfg.model_dim)

    # posterior heads per timepoint
    self.to_mu = nn.Linear(cfg.model_dim, cfg.latent_dim)
    self.to_logvar = nn.Linear(cfg.model_dim, cfg.latent_dim)

    # neuron identity embedding (0 is unlabeled/unknown)
    self.id_embed = nn.Embedding(cfg.n_identities, cfg.neuron_embed_dim)

    # decoder: combine z_t, behavior_t and return time embeddings
    self.dec_time = nn.Sequential(
        nn.Linear(cfg.latent_dim + cfg.behavior_dim, cfg.decoder_hidden),
        nn.GELU(),
        nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
        nn.GELU()
    )

    self.neuron_head = nn.Sequential(
        nn.Linear(cfg.decoder_hidden + cfg.neuron_embed_dim, cfg.decoder_hidden),
        nn.GELU(),
        nn.Linear(cfg.decoder_hidden, 1)
    )

    self.dropout = nn.Dropout(cfg.dropout)

  # -- helpers --
  def _masked_mean(self, x: torch.Tensor, m: torch.Tensor, dim: int, eps: float=1e-6) -> torch.Tensor:
    M = m.float()
    X_M = (x * M).sum(dim=dim)
    n = M.sum(dim=dim).clamp_min(eps)

    return X_M / n

  # -- forward pass --
  def forward(self, X, M, Bx, I) -> dict:
    """
    returns dict:
    - reconsruction (B, N, T)
    - mu (B, T, L)
    - logvar (B, T, L)
    - z (B, T, L)
    """
    B, N, T = X.shape
    Tb = Bx.size(1)
    assert Bx.shape == (B, Tb, T)
    assert M.shape == (B, N, T)
    assert I.shape == (B, N)

    # -- encoder --
    ## neurons
    x_flat = X.permute(0, 2, 1).contiguous().view(B * T, N, 1) # (B*T,N,1)
    m_flat = M.permute(0, 2, 1).contiguous().view(B * T, N, 1)
    n_embed = self.neuron_scalar_mlp(x_flat) # (B*T,Dn)
    n_embed = n_embed * m_flat # masking

    # masked mean
    n_token = self._masked_mean(n_embed, m_flat, dim=1) # (B*T, Dn)
    n_token = n_token.view(B, T, -1) # (B, T, Dn)

    ## behaviors
    b_token = self.beh_proj(Bx.permute(0, 2, 1)) # (B, T, Dm/f) from (B,Tb,T)->(B,T,Tb)

    ## combine
    enc_in = torch.cat((n_token, b_token), -1) # (B, T, Dn + Dm/f)
    enc_in = self.combined_proj(enc_in)
    enc_in = self.pos_encoder(enc_in)
    h = self.encoder(enc_in) # (B, T, Dm)

    # map to gaussian parameters
    mu = self.to_mu(h) # (B, T, L)
    logvar = self.to_logvar(h) # (B, T, L)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std) # random noise
    z = mu + eps * std # generate laten sample

    # -- decoder --
    # reconstruct time embeddings from z and behaviors
    b_t = Bx.permute(0, 2, 1) # (B, T, Tb)

    # drop behavior to rely on Z
    if self.training and torch.rand(()) < 0.1:
      b_t = torch.zeros_like(b_t) # force reliance on z sometimes

    dec_time = self.dec_time(torch.cat((z, b_t), -1)) # (B, T, Hd)

    # neuron identities embeddings
    id_e = self.id_embed(I) # (B, N, Ei)

    # additional affine parameters per neuron for gain and bias
    aff = self.id_to_affine(id_e) # (B,N,2)
    gain, bias = aff[...,0], aff[...,1] # (B,N)

    # prediction per neuron
    dt_exp = dec_time.unsqueeze(2).expand(B, T, N, dec_time.size(-1))
    id_exp = id_e.unsqueeze(1).expand(B, T, N, id_e.size(-1))
    nh_in = torch.cat((dt_exp, id_exp), dim=-1) # (B, T, N, Hd+Ei)
    pred = self.neuron_head(nh_in).squeeze(-1) # (B, T, N)

    pred = pred * gain.unsqueeze(1) + bias.unsqueeze(1)

    # reconstructed traces
    recon = pred.permute(0, 2, 1).contiguous() # (B, N, T) for loss

    # -- losses --
    # reconstruction loss
    loss_rec = self.f_loss_rec(recon, X, M)

    # KL to N(0,1)
    kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_true = kl_elem.mean()

    # posterior collapse prevention
    tau = getattr(self, "tau_freebits", 0.0)
    if tau > 0:
        kl_dim = kl_elem.mean(dim=(0,1)) # (L,)
        loss_kl = torch.clamp(kl_dim, min=tau).mean()
    else:
        loss_kl = kl_true

    loss_sum = loss_rec + self.cfg.beta_kl * loss_kl

    return {
        "reconstruction": recon,
        "mu": mu,
        "z": z,
        "loss_rec": loss_rec,
        "loss_kl": loss_kl,
        "loss_sum": loss_sum
    }