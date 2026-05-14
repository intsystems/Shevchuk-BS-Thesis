import os
import sys
import torch
import torch.nn as nn
import random

from src.utils.shared_layers import Projector
from train.config import TrainConfig

import torch.nn.functional as F

def _load_neurostorm_backbone(ckpt_path: str, freeze: bool = True) -> nn.Module:
    """
    Loads NeuroSTORMMAE from a Lightning checkpoint and returns only the encoder.
    Strips the 'model.' prefix added by LightningModel wrapper.
    """
    neurostorm_root = os.path.join(os.path.dirname(__file__), "..", "NeuroSTORM")
    sys.path.insert(0, os.path.abspath(neurostorm_root))
    from models.neurostorm import NeuroSTORMMAE  # noqa: PLC0415

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    backbone = NeuroSTORMMAE(
        img_size=hp["img_size"],
        in_chans=hp["in_chans"],
        embed_dim=hp["embed_dim"],
        window_size=hp["window_size"],
        first_window_size=hp["first_window_size"],
        patch_size=hp["patch_size"],
        depths=hp["depths"],
        num_heads=hp["num_heads"],
        c_multiplier=hp["c_multiplier"],
        last_layer_full_MSA=hp["last_layer_full_MSA"],
        drop_rate=hp["attn_drop_rate"],
        drop_path_rate=hp["attn_drop_rate"],
        attn_drop_rate=hp["attn_drop_rate"],
        mask_ratio=hp["mask_ratio"],
        spatial_mask=hp["spatial_mask"],
        time_mask=hp["time_mask"],
    )

    sd = ckpt["state_dict"]
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    backbone.load_state_dict(model_sd, strict=True)

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

    return backbone

class FmriAugmentor:
    def __init__(self, config: TrainConfig):
        self.config = config.data.fmri_aug

        self.smooth_sigma = config["smooth_sigma"]
        self.noise_sigma = config["noise_sigma"]
        self.amplitude = config["amplitude"] #tuple (low, high)
        self.ratio_to_mask = config["ratio_to_mask"] #tuple (low, high), low and hign in [0,1]
    def __call__(self, x):
        #x = self._amplitude_scale(x)
        x = self._masking(x)
        #x = self.hrf_jitter_tensor(x)
        return x

    def _smoothening(self, x):

        B, X, Y, Z, T = x.shape

        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, 1, X, Y, Z)

        k = max(3, int(2 * self.config.smooth_sigma + 1))
        k = k if k % 2 == 1 else k + 1

        coords = torch.arange(0, k) - k // 2 #coord grid around 0
        g = torch.exp(-(coords)**2 / (2 * self.config.smooth_sigma ** 2))

        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
        kernel = kernel / kernel.sum() #making it an prob distr
        kernel = kernel[None, None, :, :, :]

        return F.conv3d(x, kernel, padding=k//2).reshape(B, T, X, Y, Z).permute(0,2,3,4,1)
    def _gaussian(self, x):
        return x + torch.randn_like(x) * self.config.noise_sigma
    
    def _amplitude_scale(self, x):
        low, high = self.config.amplitude
        return x * torch.zeros(x.size[0], 1, 1, 1, 1).uniform_(low, high)
    
    def hrf_jitter_tensor(x):
        # x shape: (B, X, Y, Z, T)
        shift = random.choice([-1, 0, 1])
    
        if shift == 0:
            return x
        
        B, X_dim, Y_dim, Z_dim, T_dim = x.shape
        zeros_pad = torch.zeros((B, X_dim, Y_dim, Z_dim, 1), dtype=x.dtype, device=x.device)
        
        if shift == 1:
            # Shift forward: drop the last frame, pad a zero frame at the beginning
            x_shifted = torch.cat([zeros_pad, x[..., :-1]], dim=-1)
        elif shift == -1:
            # Shift backward: drop the first frame, pad a zero frame at the end
            x_shifted = torch.cat([x[..., 1:], zeros_pad], dim=-1)
        
        return x_shifted

    def _masking(self, x):
        B, X, Y, Z, T = x.shape

        device = x.device
        low_ratio, high_ratio = self.config.ratio_to_mask

        def get_bounds(dim_size):
            low = int(dim_size * low_ratio)
            high = int(dim_size * high_ratio)
            start = torch.randint(0, dim_size, (B, 1), device=device)
            delta = torch.randint(low, high + 1, (B, 1), device=device)
            end = torch.clamp(start + delta, max=dim_size)
            return start, end

        x0, x1 = get_bounds(X)
        y0, y1 = get_bounds(Y)
        z0, z1 = get_bounds(Z)

        range_x = torch.arange(X, device=device)
        range_y = torch.arange(Y, device=device)
        range_z = torch.arange(Z, device=device)

        mask_x = (range_x >= x0) & (range_x < x1)
        mask_y = (range_y >= y0) & (range_y < y1)
        mask_z = (range_z >= z0) & (range_z < z1)

        final_mask = mask_x[:, :, None, None] & mask_y[:, None, :, None] & mask_z[:, None, None, :]

        mask = (~final_mask).float() 

        mask = self._smoothening(mask.unsqueeze(-1))

        return x * mask
    
class FMRIProjectionHead(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()

        self.mlp = Projector(config)

    def forward(self, x):
        """
        Input: (B, 288, 2, 2, 2, T)
        """
        # 1. Spatiotemporal Pooling: усредняем по X, Y, Z и T (индексы 2, 3, 4, 5)
        x_pooled = x.mean(dim=(2, 3, 4, 5)) # Ожидаемый выход: (B, 288)
        
        # 2. Проекция в общее пространство
        return self.mlp(x_pooled)

class FMRIEncoderVolume(nn.Module):
    """
    Volumetric fMRI encoder: NeuroSTORM backbone → Global Average Pool → Projector.

    Input:  (B, 1, 96, 96, 96, T)   — T time frames (NeuroSTORM was pretrained with T=20)
    Output: (B, proj_out_dim)        — L2-normalized embedding

    Encoder output (B, 288, 2, 2, 2, T) is pooled over all spatial and temporal
    dimensions giving a (B, 288) vector, then projected to proj_out_dim.

    Args:
        ckpt_path:    path to the NeuroSTORM Lightning checkpoint (.ckpt)
        proj_out_dim: dimensionality of the final embedding
        freeze:       if True, backbone weights are frozen (fine-tune projector only)
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.backbone = _load_neurostorm_backbone(config.model.Neurostorm_ckpt, freeze=config.train.freeze_backbone)
        self.projector = FMRIProjectionHead(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, X, Y, Z, T)  — dataset/augmentor format
        NeuroSTORM expects (B, 1, X, Y, Z, T): add the singleton channel dim.
        """
        if x.dim() == 5:
            x = x.unsqueeze(1)                          # (B, 1, X, Y, Z, T)
        latent, _ = self.backbone.forward_encoder(x)   # (B, 288, 2, 2, 2, T)
        pooled = latent.mean(dim=(2, 3, 4, 5))         # (B, 288)
        return self.projector(pooled)                   # (B, proj_out_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone in eval mode when frozen so BN/dropout behave correctly
        if all(not p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self


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
