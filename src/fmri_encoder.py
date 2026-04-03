import torch.nn as nn

from src.utils.shared_layers import Projector


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class FMRIEncoder1D(nn.Module):
    """
    1D ResNet operating on parcellated fMRI ROI vectors.
    Input per TR: [B, n_roi] → treated as a 1D signal [B, 1, n_roi].
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = BasicBlock1D(32, 64, stride=2)
        self.layer2 = BasicBlock1D(64, 128, stride=2)
        self.layer3 = BasicBlock1D(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.proj = Projector(256, embed_dim)

    def forward(self, x):
        """
        x: [B, n_roi]
        returns: [B, embed_dim]
        """
        x = x.unsqueeze(1)          # [B, 1, n_roi]
        x = self.stem(x)            # [B, 32, n_roi/2]
        x = self.layer1(x)          # [B, 64, n_roi/4]
        x = self.layer2(x)          # [B, 128, n_roi/8]
        x = self.layer3(x)          # [B, 256, n_roi/16]
        x = self.pool(x).squeeze(-1)  # [B, 256]
        x = self.dropout(x)
        x = self.proj(x)            # [B, embed_dim]
        return x
