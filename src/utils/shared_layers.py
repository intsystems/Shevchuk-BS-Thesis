import torch
from torch import nn
import torch.nn.functional as F
from train.config import TrainConfig

class Projector(nn.Module):
    def __init__(self, config: TrainConfig, in_dim: int = 200):
        super().__init__()
        hidden = config.model.projector_hidden_dim
        out    = config.model.projector_out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
            nn.BatchNorm1d(out),
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), p=2, dim=-1)
