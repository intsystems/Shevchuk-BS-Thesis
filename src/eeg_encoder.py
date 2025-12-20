import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.shared_layers import Projector
from src.utils.utils import count_params

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            w = self.weight
            out_ch = w.size(0)

            w_flat = w.view(out_ch, -1)
            norms = w_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8) 

            scale = torch.clamp(self.max_norm / norms, max=1.0)
            scale = scale.view(out_ch, 1, 1, 1)

            self.weight.mul_(scale)

        return super().forward(x)

class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        dropout: float = 0.25,
        pool1: int = 4,
        pool2: int = 8,
        separable_kernel: int = 16,
        use_max_norm: bool = True,
        max_norm: float = 1.0,
        proj_dim = 128,
    ):
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes

        Conv = Conv2dWithConstraint if use_max_norm else nn.Conv2d

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )

        self.depthwise_spatial = nn.Sequential(
            Conv(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
                padding=(0, 0),
                max_norm=max_norm,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(p=dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(p=dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self._forward_features(dummy)
            feat_dim = out.shape[1] * out.shape[2] * out.shape[3]

        self.classifier = nn.Linear(feat_dim, n_classes)
        self.elu = nn.ELU(inplace=True)
        self.proj = Projector(n_classes, proj_dim)

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.depthwise_spatial(x)
        x = self.separable(x)
        return x

    def forward(self, x):
        """
        x: (N, C, T)
        """
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        self.elu(x)
        x = self.proj(x)
        return x