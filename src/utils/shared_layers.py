import torch
from torch import nn
import torch.nn.functional as F
from train.config import TrainConfig

class Projector(nn.Module):
    def __init__(self, config: TrainConfig, in_dim=200):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.model.projector_hidden_dim),
            nn.GELU(),
            nn.Linear(config.model.projector_hidden_dim, config.model.projector_hidden_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)
