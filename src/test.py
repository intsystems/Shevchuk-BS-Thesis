import torch
from src.eeg_encoder import EEGAugmentor
from src.fmri_encoder import FmriAugmentor

import torch
import unittest

import torch.nn.functional as F 
from train.config import TrainConfig
from src.utils.preprocess_data import extract_all_tar_gz
from src.utils.dataset import SimultEEG_fMRI

import pandas as pd

class TestMasking(unittest.TestCase):
    def setUp(self):
        # Имитируем минимальный класс, в котором живет метод _masking
        class MockModel:
            def __init__(self, ratio_to_mask, smooth_sigma):
                self.ratio_to_mask = ratio_to_mask
                self.smooth_sigma = smooth_sigma
            
            # Вставляем сюда нашу функцию
            def _smoothening(self, x):

                B, X, Y, Z, T = x.shape

                x = x.permute(0, 4, 1, 2, 3).reshape(B * T, 1, X, Y, Z)

                k = max(3, int(2 * self.smooth_sigma + 1))
                k = k if k % 2 == 1 else k + 1

                coords = torch.arange(0, k) - k // 2 #coord grid around 0
                g = torch.exp(-(coords)**2 / (2 * self.smooth_sigma ** 2))

                kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
                kernel = kernel / kernel.sum() #making it an prob distr
                kernel = kernel[None, None, :, :, :]

                return F.conv3d(x, kernel, padding=k//2).reshape(B, T, X, Y, Z).permute(0,2,3,4,1)

            def _masking(self, x):
                B, X, Y, Z, T = x.shape
                print(f"input:{x.shape}")
                device = x.device
                low_ratio, high_ratio = self.ratio_to_mask

                # 1. Generate random box parameters on the correct device
                # Reshape to (B, 1) for broadcasting against (X,), (Y,), or (Z,)
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

                print(y0)
                print(y1)

                # 2. Create 1D coordinate ranges
                # Shapes: (X,), (Y,), (Z,)
                range_x = torch.arange(X, device=device)
                range_y = torch.arange(Y, device=device)
                range_z = torch.arange(Z, device=device)

                # 3. Use broadcasting to create the 3D mask without meshgrid
                # (B, X) * (B, Y) * (B, Z) -> (B, X, Y, Z)
                mask_x = (range_x >= x0) & (range_x < x1)
                mask_y = (range_y >= y0) & (range_y < y1)
                mask_z = (range_z >= z0) & (range_z < z1)

                # Combine dimensions using unsqueeze to align: (B, X, 1, 1), (B, 1, Y, 1), (B, 1, 1, Z)
                final_mask = mask_x[:, :, None, None] & mask_y[:, None, :, None] & mask_z[:, None, None, :]
                print(f"final_mask{final_mask.shape}")
                # 4. Invert and convert to float (0 for masked area, 1 for keeping)
                # Cast to float before smoothing
                mask = (~final_mask).float() 
                print(mask[0,:,y0.item() + 1,z0.item() + 1])

                # 5. Apply smoothing (ensure _smoothening handles B, X, Y, Z)
                mask = self._smoothening(mask.unsqueeze(-1))
                #print(mask)
                print(mask[0,:,y0.item() + 1,z0.item() + 1, 0])
                

                # 6. Apply to input (unsqueeze T dimension if final_mask is only 4D)
                # result shape: (B, X, Y, Z, T)
                return x * mask

        self.model = MockModel(ratio_to_mask=(0.2, 0.5), smooth_sigma=2)

    def test_output_shape(self):
        """Проверка сохранения формы тензора"""
        x = torch.ones((2, 16, 16, 16, 5)) # B, X, Y, Z, T
        output = self.model._masking(x)
        self.assertEqual(x.shape, output.shape)

    def test_masking_happens(self):
        """Проверка того, что часть данных действительно зануляется"""
        x = torch.ones((1, 10, 10, 10, 2))
        output = self.model._masking(x)
        
        # Считаем количество нулей
        num_zeros = (output == 0).sum().item()
        # При ratio (0.2, 0.5) нули должны появиться
        self.assertGreater(num_zeros, 0, "Маска не применилась (нет нулей)")
        self.assertLess(num_zeros, x.numel(), "Маска затерла весь тензор")

    def test_randomness(self):
        """Проверка, что маски разные для разных проходов"""
        x = torch.ones((1, 20, 20, 20, 1))
        out1 = self.model._masking(x)
        out2 = self.model._masking(x)
        
        # Вероятность того, что две случайные маски совпадут — ничтожна
        self.assertFalse(torch.equal(out1, out2), "Маски идентичны, рандом не работает")

def test_neurostorm_transform():
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from src.utils.preprocess_data import transform_to_neurostorm_space

    NII_PATH = "data/projects/EEG_FMRI/data_indi_preproc/sub-01/ses-01/func/sub-01_ses-01_task-checker_bold/func_preproc/func_pp_nofilt_sm0.mni152.3mm.nii.gz"
    T_IDX = 0

    print("Loading fMRI...")
    img = nib.load(NII_PATH)
    original = np.asarray(img.dataobj, dtype=np.float32)
    print(f"Original shape: {original.shape}")

    print("Transforming to NeuroSTORM space...")
    transformed = transform_to_neurostorm_space(original)
    print(f"Transformed shape: {transformed.shape}")

    def get_slices(vol, t=0):
        v = vol[..., t]
        cx, cy, cz = v.shape[0] // 2, v.shape[1] // 2, v.shape[2] // 2
        return v[:, :, cz], v[:, cy, :], v[cx, :, :]

    orig_slices  = get_slices(original, T_IDX)
    tran_slices  = get_slices(transformed, T_IDX)
    titles = ["Axial", "Coronal", "Sagittal"]

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)

    for col, (orig, tran, title) in enumerate(zip(orig_slices, tran_slices, titles)):
        vmin, vmax = orig.min(), orig.max()

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(np.rot90(orig), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"Original {title}\n{original.shape[:3]}", fontsize=9)
        ax.axis("off")

        print(f"Orig:{np.sum(orig == 0) / np.array(orig.shape).prod()}")
        print(f"Transf:{np.sum(tran == 0) / np.array(tran.shape).prod()}")

        ax = fig.add_subplot(gs[1, col])
        ax.imshow(np.rot90(tran), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"Transformed {title}\n{transformed.shape[:3]}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"transform_to_neurostorm_space  |  t={T_IDX}", fontsize=12)
    plt.savefig("neurostorm_transform_check.png", dpi=150, bbox_inches="tight")
    print("Saved: neurostorm_transform_check.png")
    plt.show()

    assert transformed.shape[:3] == (96, 96, 96), f"Expected (96,96,96,...), got {transformed.shape}"
    assert transformed.shape[3] == original.shape[3], "T dimension changed"
    assert not np.isnan(transformed).any(), "NaN in output"
    print("All checks passed.")

def test_neurostorm_encoder():
    """
    Tests that NeuroSTORM MAE encoder loads correctly from checkpoint
    and produces expected output shape from a random (1, 1, 96, 96, 96, 20) input.

    Run from the repo root:
        python -m src.test

    Requires: mamba_ssm, monai, einops  (available in Colab with GPU)
    """
    import sys
    import os
    import torch

    # Add NeuroSTORM repo to path so its internal imports resolve
    neurostorm_root = os.path.join(os.path.dirname(__file__), "..", "NeuroSTORM")
    sys.path.insert(0, os.path.abspath(neurostorm_root))

    from models.neurostorm import NeuroSTORMMAE

    CKPT_PATH = os.path.join(os.path.dirname(__file__), "pt_neurostorm_mae_ratio0.5.ckpt")

    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    print(f"  img_size      : {hp['img_size']}")
    print(f"  patch_size    : {hp['patch_size']}")
    print(f"  embed_dim     : {hp['embed_dim']}")
    print(f"  depths        : {hp['depths']}")
    print(f"  mask_ratio    : {hp['mask_ratio']}")
    print(f"  spatial_mask  : {hp['spatial_mask']}")
    print(f"  time_mask     : {hp['time_mask']}")

    print("\nBuilding NeuroSTORMMAE...")
    model = NeuroSTORMMAE(
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

    # Strip 'model.' prefix added by the LightningModel wrapper
    sd = ckpt["state_dict"]
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=True)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    print("Weights loaded successfully.")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on: {device}")

    # Input: (B, C, X, Y, Z, T) matching the checkpoint's img_size [96, 96, 96, 20]
    x = torch.randn(1, 1, 96, 96, 96, 20, device=device)
    print(f"\nInput  shape : {tuple(x.shape)}")

    with torch.no_grad():
        latent, mask = model.forward_encoder(x)

    print(f"Latent shape : {tuple(latent.shape)}")
    print(f"Mask   shape : {tuple(mask.shape)}")
    print(f"Latent mean  : {latent.mean().item():.4f}")
    print(f"Latent std   : {latent.std().item():.4f}")
    print(f"Masked tokens: {int(mask.sum().item())} / {mask.numel()}")

    # Expected encoder output: (B, 288, 2, 2, 2, 20) based on architecture comments
    assert latent.ndim == 6, f"Expected 6-dim latent, got {latent.ndim}"
    assert latent.shape[0] == 1, "Batch size should be 1"
    print("\nAll assertions passed.")


def test_fmri_encoder_volume():
    """
    Tests FMRIEncoderVolume end-to-end:
      (B, 1, 96, 96, 96, 20) → backbone → GAP → projector → (B, 128)

    Run from repo root:
        python -m src.test
    """
    import torch
    from src.fmri_encoder import FMRIEncoderVolume

    CKPT_PATH = "src/pt_neurostorm_mae_ratio0.5.ckpt"
    B = 2

    print("Building FMRIEncoderVolume (frozen backbone)...")
    encoder = FMRIEncoderVolume(ckpt_path=CKPT_PATH, proj_out_dim=128, freeze=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    print(f"Running on: {device}")

    x = torch.randn(B, 1, 96, 96, 96, 20, device=device)
    print(f"Input  shape : {tuple(x.shape)}")

    encoder.eval()
    with torch.no_grad():
        z = encoder(x)

    print(f"Output shape : {tuple(z.shape)}")        # expect (2, 128)
    print(f"Output norm  : {z.norm(dim=-1).tolist()}")  # expect [1.0, 1.0] (L2 normalized)

    assert z.shape == (B, 128), f"Expected ({B}, 128), got {z.shape}"
    assert torch.allclose(z.norm(dim=-1), torch.ones(B, device=device), atol=1e-5), \
        "Output is not L2-normalized"
    print("All assertions passed.")


if __name__ == '__main__':
    test_fmri_encoder_volume()