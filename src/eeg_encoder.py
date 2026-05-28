import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils.shared_layers import Projector
from train.config import TrainConfig
from braindecode.models import Labram

Labram_OUT_DIM = 200

class EEGAugmentor:
    def __init__(self, config: TrainConfig):
        self.config = config.data.eeg_aug
        #input shape [B, C, 200 * T], T - number of seconds

    def __call__(self, x):
        x = self._freq_masking(x)
        x = self._amplitude_scale(x)
        x = self._pink_noise(x)
        #x = self._time_shift(x)
        x = self._channel_drop(x)
        return x
    def _gaussian(self, x):
        noise = torch.randn_like(x) * self.config.sigma
        return x + noise
    
    def _pink_noise(self, x):
        """
        Generates pink noise (1/f) by scaling white noise in the frequency domain.
        Input shape: [B, C, L]
        """
        batch, channels, length = x.shape
        device = x.device
        
        # 1. Generate white noise in time domain
        white_noise = torch.randn_like(x) * self.config.sigma
        
        # 2. Transform to frequency domain
        # rfft is faster for real-valued signals like EEG
        noise_fft = torch.fft.rfft(white_noise)
        
        # 3. Create 1/f filter
        # freqs contains values from 0 to Nyquist
        freqs = torch.fft.rfftfreq(length, device=device)
        
        # We want Power ~ 1/f, so Amplitude ~ 1/sqrt(f)
        # Avoid division by zero at the DC component (index 0)
        mask = torch.zeros_like(freqs)
        mask[1:] = 1.0 / torch.sqrt(freqs[1:])
        mask[0] = mask[1] # Smoothen the DC component
        
        # 4. Apply mask and transform back to time domain
        pink_noise_fft = noise_fft * mask
        pink_noise = torch.fft.irfft(pink_noise_fft, n=length)
        
        # 5. Scale to desired intensity
        # We normalize to unit variance first, then scale by self.config.sigma
        pink_noise = (pink_noise / pink_noise.std()) * self.config.sigma
        
        return x + pink_noise
    
    def _channel_drop(self, x):
        mask = torch.bernoulli(
            torch.ones(x.shape[0], x.shape[1], 1, device=x.device) * (1 - self.config.channel_drop_prob)
        )
        return x * mask

    def _time_shift(self, x):
        shifts = torch.randint(-self.config.max_time_shift, self.config.max_time_shift,
                               (x.shape[0],), device=x.device)
        return torch.stack([torch.roll(x[i].squeeze(), shifts[i].item(), dims=(-1)) for i in range(x.shape[0])])

    def _freq_masking(self, x):
        fft = torch.fft.rfft(x, dim=-1)
        num_freqs = fft.shape[-1]

        apply_mask = (torch.rand((x.shape[0], x.shape[1]), device=x.device)
                      > self.config.bandwidth_mask_prob).unsqueeze(-1)

        freq_to_mask_start = torch.randint(0, num_freqs,
                                           size=(x.shape[0], x.shape[1]),
                                           device=x.device).unsqueeze(-1)
        bandwidth = torch.randint(0, max(1, math.floor(45 * self.config.ratio_of_freq_to_drop)),
                                  size=(x.shape[0], x.shape[1]),
                                  device=x.device).unsqueeze(-1)
        freq_to_mask_end = freq_to_mask_start + bandwidth

        indices = torch.arange(num_freqs, device=x.device).reshape(1, 1, num_freqs)

        mask = ~((indices >= freq_to_mask_start) & (indices <= freq_to_mask_end))
        mask = mask | ~apply_mask

        return torch.fft.irfft(fft * mask, n=x.shape[2])

    def _amplitude_scale(self, x):
        low, high = self.config.amplitude
        amp = torch.zeros((x.shape[0], 1, 1), device=x.device).uniform_(low, high)
        return x * amp

class EEGProjectionHead(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        # 2-слойный MLP
        self.mlp = Projector(config, in_dim=config.model.Labram_out_dim)

    def forward(self, x):
        """
        Вход: (B, L, 200), где L - количество токенов времени (T / sr)
        """

        return self.mlp(x) # Ожидаемый выход: (B, 256)

class EEGEncoder(nn.Module):
    """
    EEG encoder: LaBraM backbone → CLS token → Projector → L2-normalized embedding.

    Args:
        ch_names:     list of channel name strings matching the input EEG data
        proj_out_dim: output embedding dimensionality
        pretrained:   if True, load braindecode/labram-pretrained weights
    """
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.ch_names = config.data.ch_names
        self.freeze_backbone = config.train.freeze_backbone
        n_chans = len(self.ch_names)

        if config.model.labram_pretrained:
            self.backbone = Labram.from_pretrained("braindecode/labram-pretrained")
        else:
            self.backbone = Labram(
                n_chans=n_chans,
                n_times=400,
                sfreq=200,
                n_outputs=2,
            )

        if config.train.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # cache LaBraM's 128-channel reference montage for ch_name → index lookup
        # convention: positional index 0 is CLS, channels start at 1
        self._ref_names = [c["ch_name"].upper() for c in self.backbone.chs_info]
        self._input_chans_cache: dict[tuple, torch.Tensor] = {}

        self.projector = Projector(config)

    def _build_input_chans(self, ch_names, device):
        """Build LaBraM input_chans tensor [CLS_idx=0, ch1_idx+1, ch2_idx+1, ...]."""
        key = tuple(ch_names)
        if key not in self._input_chans_cache:
            idx = []
            for c in ch_names:
                cu = c.upper()
                if cu not in self._ref_names:
                    raise ValueError(f"Channel {c!r} not in LaBraM reference montage")
                idx.append(self._ref_names.index(cu) + 1)
            self._input_chans_cache[key] = torch.tensor([0] + idx, dtype=torch.long)
        return self._input_chans_cache[key].to(device)

    def forward(self, x: torch.Tensor, ch_names=None) -> torch.Tensor:
        """
        x: (B, C, T) — channels in some consistent order across the batch
        ch_names: list of channel name strings, length C. Required.
        returns: (B, proj_out_dim)
        """
        if ch_names is None:
            ch_names = self.ch_names
        input_chans = self._build_input_chans(ch_names, x.device)
        if self.training:
            from torch.utils.checkpoint import checkpoint
            # Recompute activations during backward instead of caching them.
            # Labram has 12 blocks each producing ~415 MB attention matrices in bf16;
            # storing all of them for backprop exceeds GPU budget at batch 32.
            cls = checkpoint(
                lambda x_: self.backbone.forward_features(x_, input_chans=input_chans),
                x, use_reentrant=False,
            )
        else:
            cls = self.backbone.forward_features(x, input_chans=input_chans)
        if cls.dim() == 3:                # safety: (B, num_tokens, D) → take CLS
            cls = cls[:, 0]
        return self.projector(cls)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
