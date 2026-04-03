import torch
import torch.nn as nn

from BIOT.model.biot import BIOTEncoder
from src.utils.shared_layers import Projector


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
        Load pretrained BIOT weights from a different-channel checkpoint.
        channel_tokens are tiled cyclically from the pretrained channel count
        to the target channel count (e.g. 16 → 64).
        index is skipped — the new model already has the correct 0..n_channels-1 index.
        """
        state = torch.load(path, map_location="cpu")

        # Some checkpoints nest weights under 'biot.' prefix
        if any(k.startswith("biot.") for k in state):
            state = {k[len("biot."):]: v for k, v in state.items() if k.startswith("biot.")}

        # Tile channel_tokens to match the target channel count
        if "channel_tokens.weight" in state:
            pretrained = state["channel_tokens.weight"]          # [n_pre, 256]
            n_pre = pretrained.shape[0]
            n_target = self.encoder.channel_tokens.num_embeddings
            repeats = (n_target + n_pre - 1) // n_pre
            tiled = pretrained.repeat(repeats, 1)[:n_target]    # [n_target, 256]
            state["channel_tokens.weight"] = tiled
            print(f"  channel_tokens tiled {n_pre} → {n_target}")

        # Drop index — new model already holds the correct 0..n_channels-1 values
        state.pop("index", None)

        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        print(f"Loaded pretrained BIOT weights from {path}")
        print(f"  Missing (reinitialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
