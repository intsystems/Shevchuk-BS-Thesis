import torch
from torch import nn

from src.utils.shared_layers import Projector
from src.utils.utils import count_params

class FMRIEncoderROI(nn.Module):
    def __init__(self, roi_dim, embed_dim = 128):
        super().__init__()
        self.roi_dim = roi_dim
        self.embed_dim = embed_dim

        self.backbone = nn.Sequential(
            nn.Linear(roi_dim, 512, bias=True),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, 512, bias=True),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.4),

            nn.Linear(512, 256, bias=True),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.proj = Projector(256, embed_dim)

    def forward(self, x):
        h = self.backbone(x)
        z = self.proj(h)
        #z = F.normalize(z, p=2, dim=-1)
        return z
