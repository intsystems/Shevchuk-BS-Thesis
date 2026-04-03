import torch.nn as nn

from src.utils.shared_layers import Projector


class FMRIEncoder1D(nn.Module):
    """
    MLP encoder for parcellated fMRI ROI vectors.
    Input per TR: [B, n_roi] — treated as a flat feature vector, not a 1D signal.
    Using a plain MLP avoids imposing a false 1D spatial ordering on Schaefer ROIs.
    """
    def __init__(self, n_roi: int = 200, hidden_dim: int = 512, embed_dim: int = 128, dropout: float = 0.4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_roi, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.proj = Projector(hidden_dim // 2, embed_dim)

    def forward(self, x):
        """
        x: [B, n_roi]
        returns: [B, embed_dim]
        """
        x = self.net(x)
        return self.proj(x)
