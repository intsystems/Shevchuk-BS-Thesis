import torch
from src.eeg_encoder import EEGAugmentor
from src.fmri_encoder import FmriAugmentor

import torch
import unittest

import torch.nn.functional as F 
from train.config import TrainConfig
from src.utils.preprocess_data import extract_all_tar_gz
from src.utils.dataset import SimultEEG_fMRI, ContrastiveBatchSampler, collate_fn
from torch.utils.data import DataLoader

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


def test_labram_raw():
    from braindecode.models import Labram
    import torch

    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
                'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10',
                'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3',
                'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6',
                'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Fpz', 'CPz']

    B, C, T = 2, len(ch_names), 3000  # 30 sec * 200 Hz

    print(f"Input: ({B}, {C}, {T})")
    print("Loading pretrained LaBraM...")
    model = Labram.from_pretrained("braindecode/labram-pretrained")
    model.eval()

    print(model.n_times)
    print(model.neural_tokenizer)
    print(model.position_embedding.shape)

    x = torch.randn(B, C, T)

    with torch.no_grad():
        out = model(x, ch_names=ch_names, return_features=True)

    print("Output keys:", list(out.keys()) if isinstance(out, dict) else type(out))
    if isinstance(out, dict):
        for k, v in out.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"  output shape: {out.shape}")


def test_fmri_preprocessing_steps():
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    from src.utils.preprocess_data import spatial_resampling, select_middle_96

    NII_PATH = (
        "data/projects/EEG_FMRI/data_indi_preproc/"
        "sub-01/ses-01/func/sub-01_ses-01_task-checker_bold/"
        "func_preproc/func_pp_nofilt_sm0.mni152.3mm.nii.gz"
    )
    OUT_PATH = "fmri_preproc_steps.png"
    T_IDX = 0

    img = nib.load(NII_PATH)
    raw = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z, T)
    header = img.header
    print(f"Raw shape: {raw.shape}, voxel size: {header.get_zooms()[:3]}")

    # Step 1: raw
    step1 = raw[..., T_IDX]

    # Step 2: spatial resample 3mm → 2mm
    step2_4d = spatial_resampling(raw, header, target_voxel_size=(2, 2, 2))
    step2 = step2_4d[..., T_IDX]
    print(f"After resample: {step2_4d.shape}")

    # Step 3: select_middle_96 (crop/pad to 96x96x96)
    step3_4d = select_middle_96(step2_4d).numpy()
    step3 = step3_4d[..., T_IDX]
    print(f"After select_middle_96: {step3_4d.shape}")

    # Step 4: brain mask derived from MNI volume (background = exactly 0 after MNI reg)
    brain_mask = (step3_4d != 0).any(axis=-1)  # (96, 96, 96) bool
    print(f"Brain voxels: {brain_mask.sum()} / {brain_mask.size} ({100*brain_mask.mean():.1f}%)")
    step4 = brain_mask.astype(np.float32)

    # Step 5: apply mask (zero out background), no clip
    step5_4d = step3_4d.copy()
    step5_4d[~brain_mask] = 0
    step5 = step5_4d[..., T_IDX]

    # Step 6: global z-norm only inside brain, fill background with min
    brain_vals = step5_4d[brain_mask]
    mu, sigma = brain_vals.mean(), brain_vals.std()
    step6_4d = step5_4d.copy()
    step6_4d[brain_mask] = (step5_4d[brain_mask] - mu) / sigma
    step6_4d[~brain_mask] = 0
    step6 = step6_4d[..., T_IDX]
    print(f"Z-norm: mean={mu:.2f}, std={sigma:.2f}")

    steps = [
        ("1. Raw (3mm MNI)", step1),
        ("2. Resampled (2mm)", step2),
        ("3. select_middle_96", step3),
        ("4. Zero mask", step4),
        ("5. Mask applied + clip<0", step5),
        ("6. Z-normalized", step6),
    ]

    # --- Figure 1: 6 preprocessing steps (axial only) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (title, vol) in zip(axes.flat, steps):
        cz = vol.shape[2] // 2
        sl = np.rot90(vol[:, :, cz])
        vmin, vmax = np.percentile(vol, 1), np.percentile(vol, 99)
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{title}\nshape={vol.shape}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"fMRI preprocessing steps  |  t={T_IDX}", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.show()

    # --- Figure 2: original vs final, 3 planes ---
    cx, cy, cz = np.array(step6.shape) // 2
    cx0, cy0, cz0 = np.array(step1.shape) // 2

    rows = [
        ("Original .nii.gz (3mm MNI)", step1,
         cx0, cy0, cz0,
         np.percentile(step1, 1), np.percentile(step1, 99), "gray"),
        ("Final: z-normalized (96³, 2mm)", step6,
         cx, cy, cz,
         -3, 3, "RdBu_r"),
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    for row_idx, (title, vol, rx, ry, rz, vmin, vmax, cmap) in enumerate(rows):
        plane_slices = [
            (np.rot90(vol[:, :, rz]), f"Axial z={rz}"),
            (np.rot90(vol[:, ry, :]), f"Coronal y={ry}"),
            (np.rot90(vol[rx, :, :]), f"Sagittal x={rx}"),
        ]
        for col_idx, (sl, plane_title) in enumerate(plane_slices):
            ax = axes2[row_idx, col_idx]
            ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            if row_idx == 0:
                ax.set_title(plane_title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.axis("off")

    fig2.suptitle("Original vs Final preprocessed  |  t=0", fontsize=12)
    plt.tight_layout()
    plt.savefig("fmri_compare.png", dpi=150, bbox_inches="tight")
    print("Saved: fmri_compare.png")
    plt.show()


def test_fmri_zero_mask():
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt

    BASE = (
        "data/projects/EEG_FMRI/data_indi_preproc/"
        "sub-01/ses-01/func/sub-01_ses-01_task-checker_bold/"
    )
    NII_PATH  = BASE + "func_preproc/func_pp_nofilt_sm0.mni152.3mm.nii.gz"
    MASK_PATH = BASE + "func_seg/global_mask.nii.gz"
    OUT_PATH  = "fmri_zero_mask.png"

    vol  = np.asarray(nib.load(NII_PATH).dataobj,  dtype=np.float32)  # (X, Y, Z, T)
    gmask = np.asarray(nib.load(MASK_PATH).dataobj, dtype=np.float32)  # (X, Y, Z)
    print(f"Volume shape:      {vol.shape}")
    print(f"Global mask shape: {gmask.shape}")

    zero_mask = (vol == 0).all(axis=-1).astype(np.float32)  # 1 where always-zero
    gmask_bin = (gmask > 0).astype(np.float32)              # 1 where brain

    print(f"Zero voxels:        {int(zero_mask.sum())} / {zero_mask.size} ({100*zero_mask.mean():.1f}%)")
    print(f"Global mask voxels: {int(gmask_bin.sum())} / {gmask_bin.size} ({100*gmask_bin.mean():.1f}%)")
    print("Note: different spaces — no direct overlap computed")

    rows = [
        ("Zero-voxel mask  [MNI 3mm space]", zero_mask, "Reds"),
        ("global_mask  [native func space]", gmask_bin, "Blues"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row_idx, (title, data, cmap) in enumerate(rows):
        cx, cy, cz = np.array(data.shape) // 2
        slices = [
            (data[:, :, cz], f"Axial  z={cz}"),
            (data[:, cy, :], f"Coronal  y={cy}"),
            (data[cx, :, :], f"Sagittal  x={cx}"),
        ]
        for col_idx, (sl, subtitle) in enumerate(slices):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.rot90(sl), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            if row_idx == 0:
                ax.set_title(subtitle, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.axis("off")

    fig.suptitle("Row 1: always-zero voxels  |  Row 2: global_mask.nii.gz\n"
                 "(different spaces — not directly comparable)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.show()


def test_eeg_preprocessing():
    """
    Compares raw .set vs preprocessed .npy EEG:
      - PSD before/after bandpass shows band rejection
      - per-channel mean/std after z-norm should be 0/1
      - no NaN or Inf
    """
    import mne
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from pathlib import Path
    from train.config import TrainConfig
    from src.utils.preprocess_data import preprocess_eeg

    SET_PATH = Path(
        "data/projects/EEG_FMRI/data_indi_preproc/"
        "sub-01/ses-01/eeg/sub-01_ses-01_task-inscapes_eeg.set"
    )
    config = TrainConfig()
    SR_proc = config.data.eeg_sr  # 200 Hz

    # load raw BEFORE preprocessing
    raw = mne.io.read_raw_eeglab(str(SET_PATH), preload=True)
    raw_sr = int(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True)
    raw_data = raw.get_data(picks=picks).astype(np.float32)  # (C, T) raw volts
    print(f"Raw:  shape={raw_data.shape}  sr={raw_sr} Hz")

    # run preprocessing → writes .npy
    proc_data = preprocess_eeg(SET_PATH, config)              # (C, T) z-normed
    print(f"Proc: shape={proc_data.shape}  sr={SR_proc} Hz")

    # sanity checks
    assert not np.isnan(proc_data).any(), "NaN in preprocessed EEG!"
    assert not np.isinf(proc_data).any(), "Inf in preprocessed EEG!"

    ch_means = proc_data.mean(axis=-1)
    ch_stds  = proc_data.std(axis=-1)
    print(f"Per-channel mean: min={ch_means.min():.4f}  max={ch_means.max():.4f}  (should be ~0)")
    print(f"Per-channel std:  min={ch_stds.min():.4f}   max={ch_stds.max():.4f}  (should be ~1)")

    # PSD comparison
    CH = 0
    f_raw,  pxx_raw  = welch(raw_data[CH],  fs=raw_sr,  nperseg=raw_sr)
    f_proc, pxx_proc = welch(proc_data[CH], fs=SR_proc, nperseg=SR_proc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(f_raw,  pxx_raw,  lw=0.8, label="raw")
    ax.axvline(config.data.lower_freq,  color="r", ls="--", label=f"low  {config.data.lower_freq} Hz")
    ax.axvline(config.data.higher_freq, color="g", ls="--", label=f"high {config.data.higher_freq} Hz")
    ax.set_xlim(0, min(raw_sr / 2, 150))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (V²/Hz)")
    ax.set_title(f"Raw PSD — ch {CH}  (sr={raw_sr} Hz)")
    ax.legend()

    ax = axes[1]
    ax.semilogy(f_proc, pxx_proc, lw=0.8, color="orange", label="preprocessed")
    ax.axvline(config.data.lower_freq,  color="r", ls="--")
    ax.axvline(config.data.higher_freq, color="g", ls="--")
    ax.set_xlim(0, SR_proc / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"Preprocessed PSD — ch {CH}  (sr={SR_proc} Hz, z-normed)")
    ax.legend()

    fig.suptitle("EEG: PSD before vs after preprocessing", fontsize=12)
    plt.tight_layout()
    plt.savefig("eeg_psd_compare.png", dpi=150, bbox_inches="tight")
    print("Saved: eeg_psd_compare.png")

    # per-channel mean and std after z-norm
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    ch_idx = np.arange(len(ch_means))

    axes2[0].bar(ch_idx, ch_means)
    axes2[0].axhline(0, color="r", lw=1)
    axes2[0].set_xlabel("Channel")
    axes2[0].set_ylabel("Mean")
    axes2[0].set_title("Per-channel mean (should be 0)")

    axes2[1].bar(ch_idx, ch_stds, color="orange")
    axes2[1].axhline(1, color="r", lw=1)
    axes2[1].set_xlabel("Channel")
    axes2[1].set_ylabel("Std")
    axes2[1].set_title("Per-channel std (should be 1)")

    fig2.suptitle("Z-norm check after preprocessing", fontsize=12)
    plt.tight_layout()
    plt.savefig("eeg_znorm_check.png", dpi=150, bbox_inches="tight")
    print("Saved: eeg_znorm_check.png")
    plt.show()


if __name__ == '__main__':

    # config = TrainConfig()

    # dataset = SimultEEG_fMRI(config, subjects=["sub-01"])
    # sampler = ContrastiveBatchSampler(dataset, config)

    # loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # for batch in loader:
    #     print(batch.keys())
    #     print(batch["eeg"].shape)
    #     print(batch["fmri"].shape)

    # print(len(sampler))
    #test_labram_raw()

    df = pd.read_csv("sub-03_ses-01_task-tp_run-02_channels.tsv", sep="\t")
    print(df["name"])