import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class VAEConfig:
    n_identities: int  # total unique neuron identity labels + 1 for 'unknown'
    behavior_dim: int  # number of behaviors
    neuron_embed_dim: int = 32
    latent_dim: int = 24
    model_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    neuron_token_dim: int = 64  # dim after per-neuron scalar -> embedding
    decoder_hidden: int = 256
    dropout: float = 0.1
    beta_kl: float = 0.2  # weight on KL term
    loss_rec_type: str = "mse"  # "mse" or "huber"
    tau_freebits: float = 0.0  # free bits threshold for KL term

class PositionalEncoding(nn.Module):
  """
  sinusoidal positional encoding for sequence length of T
  input: (B, T, D)
  """
  def __init__(self, d_model: int, max_len: int = 4096):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (- math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * divisor)
    pe[:, 1::2] = torch.cos(position * divisor)

    self.register_buffer("pe", pe.unsqueeze(0)) # (1, max_len, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, D)
    T = x.size(1)

    return x + self.pe[:, :T, :]