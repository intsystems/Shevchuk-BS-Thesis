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
        # We normalize to unit variance first, then scale by self.sigma
        pink_noise = (pink_noise / pink_noise.std()) * self.sigma
        
        return x + pink_noise
    
    def _channel_drop(self, x):
        mask = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], 1) * self.config.channel_drop_prob)

        return x * mask

    def _time_shift(self, x):
        shifts = torch.randint(-self.max_time_shift, self.max_time_shift, (x.shape[0],))
        print(shifts)

        return torch.stack([torch.roll(x[i].squeeze(), shifts[i], dims=(-1)) for i in range(x.shape[0])])
    
    def _freq_masking(self, x):
        fft = torch.fft.rfft(x, dim=-1) #applying fft along time axis

        print(fft.shape)

        num_freqs = fft.shape[-1]

        #generate freq to mask out
        
        apply_mask = (torch.rand((x.shape[0], x.shape[1])) > self.config.bandwidth_mask_prob).unsqueeze(-1)

        freq_to_mask_start = torch.randint(0, num_freqs, size=(x.shape[0], x.shape[1])).unsqueeze(-1)
        bandwidth = torch.randint(0, math.floor(45 * self.config.ratio_of_freq_to_drop), size=(x.shape[0], x.shape[1])).unsqueeze(-1)
        freq_to_mask_end = freq_to_mask_start + bandwidth

        print(freq_to_mask_end.shape)

        indices = torch.arange(num_freqs).reshape(1, 1, num_freqs)

        mask = ~(
            (indices >= freq_to_mask_start) & (indices <= freq_to_mask_end)
        ) # (B,C,F)

        mask = mask | ~apply_mask

        return torch.fft.irfft(fft * mask, n=x.shape[2])

    def _amplitude_scale(self, x):
        low, high = self.amplitude

        amp = torch.zeros((x.shape[0], 1, 1)).uniform_(low, high)

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

        self.projector = Projector(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, proj_out_dim)  L2-normalized
        """
        feat = self.backbone(x, ch_names=self.ch_names, return_features=True)
        cls = feat["cls_token"]          # (B, 200)
        return self.projector(cls)       # (B, proj_out_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone in eval mode when frozen so dropout/BN behave correctly
        if all(not p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self
