import torch
import torch.nn as nn
import math

from BIOT.model.biot import BIOTEncoder
from src.utils.shared_layers import Projector
from train.config import TrainConfig

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

class EEGEncoderBIOT(nn.Module):
    def __init__(
        self,
        n_channels: int = 60,
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        n_fft: int = 200,
        hop_length: int = 100,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.encoder = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_channels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.proj = Projector(emb_size, proj_dim)

    def forward(self, x):
        """
        x: [B, C, T]
        returns: [B, proj_dim]
        """
        x = self.encoder(x)
        x = self.proj(x)
        return x

    def load_pretrained(self, path: str):
        """
        Load pretrained BIOT weights.
        channel_tokens and index are dropped — they are sensor-layout-specific
        and will be trained from scratch for the target montage.
        """
        state = torch.load(path, map_location="cpu")

        # Some checkpoints nest weights under 'biot.' prefix
        if any(k.startswith("biot.") for k in state):
            state = {k[len("biot."):]: v for k, v in state.items() if k.startswith("biot.")}

        # Drop sensor-layout-specific weights — keep them randomly initialized
        state.pop("channel_tokens.weight", None)
        state.pop("index", None)

        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        print(f"Loaded pretrained BIOT weights from {path}")
        print(f"  Missing (reinitialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
