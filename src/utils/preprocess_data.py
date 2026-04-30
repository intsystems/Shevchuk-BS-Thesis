import mne
import numpy as np
import os
import requests
from tqdm import tqdm
import tarfile
import torch
import torch.nn.functional as F
import nibabel as nib
from nilearn import image
from train.config import TrainConfig
from pathlib import Path
import h5py
import shutil

#FOR NATVIEW DATASET

def bandpass_filter(filename, lower_freq, higher_freq):
    """
    Used to filter out freq outside of [lower_freq, higher_freq]

    filename - path to .set file in dataset

    Returns: None
    """

    signal = mne.io.read_raw_eeglab(filename, preload=True)

    signal.filter(lower_freq, higher_freq, picks="eeg", method="fir", phase="zero")

    signal.export(filename, fmt="eeglab", overwrite=True)

def downsample_eeg(filename, target_frequency):
    signal = mne.io.read_raw_eeglab(filename, preload=True)

    signal = signal.resample(target_frequency)

    signal.export(filename, fmt="eeglab", overwrite=True)

def save_as_npy(filename):
    """
    Used to transform EEG data to .npy format
    """
    signal = mne.io.read_raw_eeglab(filename, preload=True)

    picks = mne.pick_types(signal.info, eeg=True)

    eeg_data = signal.get_data(picks=picks)

    out_path = filename.with_suffix(".npy")
    np.save(out_path, eeg_data.astype(np.float32))

#copied from https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/main/datasets/preprocessing_volume.py
def select_middle_96(vector):
    start_index, end_index = [], []

    #padding to 96 cube
    sizes = np.expand_dims(np.flip(vector.shape[:3]), axis=-1)

    sizes = np.hstack([sizes, sizes]).reshape(6)

    padding = tuple([0,0] + [(96 - dim) // 2 for dim in sizes] if len(vector.shape) == 4 else [(96 - dim) // 2 for dim in sizes])
    
    vector = F.pad(torch.from_numpy(vector), pad=padding)

    for i in range(3):
        if vector.shape[i] > 96:
            start_index.append((vector.shape[i] - 96) // 2)
            end_index.append(start_index[-1] + 96)
        else:
            start_index.append(0)
            end_index.append(vector.shape[i])

    if len(vector.shape) == 3:
        result = vector[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
    elif len(vector.shape) == 4:
        result = vector[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2], :]
    
    return result

def spatial_resampling(data, header, target_voxel_size=(2, 2, 2)):
    current_voxel_size = header.get_zooms()[:3]
    scale_factors = [current / target for current, target in zip(current_voxel_size, target_voxel_size)]
    new_dims = [int(np.round(dim * scale)) for dim, scale in zip(data.shape[:3], scale_factors)]
    
    data = data.astype(np.float32)
    
    if data.ndim == 4:
        data_tensor = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(1)
    elif data.ndim == 3:
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    
    resampled_tensor = F.interpolate(data_tensor, size=new_dims, mode='trilinear', align_corners=False)
    
    if data.ndim == 4:
        resampled_data = resampled_tensor.squeeze(1).permute(1, 2, 3, 0).numpy()
    else:
        resampled_data = resampled_tensor.squeeze(0).squeeze(0).numpy()
    
    return resampled_data

def preprocess_fmri(path_to_file: str, config: TrainConfig):
    """
    path_to_file : MNI-registered 4D fMRI (X, Y, Z, T)

    Brain mask is derived directly from the MNI volume: after MNI registration
    background voxels are exactly zero across all TRs.
    """
    img = nib.load(path_to_file)
    data = np.asarray(img.dataobj, dtype=np.float32)
    header = img.header

    data = spatial_resampling(data, header, target_voxel_size=config.data.target_voxel_size)
    data = select_middle_96(data).numpy()           # (96, 96, 96, T)

    # brain mask: voxels that are non-zero in at least one TR
    # MNI registration leaves background exactly 0
    brain_mask = (data != 0).any(axis=-1)           # (96, 96, 96) bool

    # z-normalize using only brain voxels (preserves negative BOLD values)
    brain_vals = data[brain_mask]
    mu  = brain_vals.mean()
    std = brain_vals.std()
    data[brain_mask] = (data[brain_mask] - mu) / (std + 1e-8)
    # background stays 0

    return torch.from_numpy(data).to(torch.float16)  # (96, 96, 96, T)

def preprocess_eeg(set_path: Path, config: TrainConfig) -> np.ndarray:
    """
    set_path : path to .set EEGLab file
    Returns  : (C, T) float32 array, z-normalized per channel
    """
    bandpass_filter(set_path, config.data.lower_freq, config.data.higher_freq)
    downsample_eeg(set_path, config.data.target_eeg_freq)
    save_as_npy(set_path)                       # saves to set_path.with_suffix('.npy')

    npy_path = set_path.with_suffix(".npy")
    data = np.load(npy_path).astype(np.float32) # (C, T)

    mean = data.mean(axis=-1, keepdims=True)
    std  = data.std(axis=-1, keepdims=True)
    data = (data - mean) / (std + 1e-8)

    np.save(npy_path, data)                     # overwrite with z-normed version
    return data

def _eeg_qc_plot(set_path: Path, config: TrainConfig, sub_id: str, activity: str):
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    import mne

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True)
    raw_sr = int(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True)
    raw_data = raw.get_data(picks=picks).astype(np.float32)  # (C, T)

    preprocess_eeg(set_path, config)
    proc_data = np.load(set_path.with_suffix(".npy")).astype(np.float32)  # (C, T)
    proc_sr = config.data.eeg_sr

    ch_mean_raw  = raw_data.mean(axis=-1)
    ch_std_raw   = raw_data.std(axis=-1)
    ch_mean_proc = proc_data.mean(axis=-1)
    ch_std_proc  = proc_data.std(axis=-1)

    print(f"\n=== EEG QC  {sub_id}/{activity} ===")
    print(f"Raw  — mean: [{ch_mean_raw.min():.3e}, {ch_mean_raw.max():.3e}]  "
          f"std: [{ch_std_raw.min():.3e}, {ch_std_raw.max():.3e}]")
    print(f"Proc — mean: [{ch_mean_proc.min():.4f}, {ch_mean_proc.max():.4f}]  "
          f"std: [{ch_std_proc.min():.4f}, {ch_std_proc.max():.4f}]")

    f_raw,  pxx_raw  = welch(raw_data[0],  fs=raw_sr,  nperseg=raw_sr)
    f_proc, pxx_proc = welch(proc_data[0], fs=proc_sr, nperseg=proc_sr)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # PSD before
    axes[0, 0].semilogy(f_raw, pxx_raw, lw=0.8)
    axes[0, 0].axvline(config.data.lower_freq,  color="r", ls="--", label=f"{config.data.lower_freq} Hz")
    axes[0, 0].axvline(config.data.higher_freq, color="g", ls="--", label=f"{config.data.higher_freq} Hz")
    axes[0, 0].set_xlim(0, min(raw_sr / 2, 150))
    axes[0, 0].set_title(f"PSD before  (sr={raw_sr} Hz)")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Power (V²/Hz)")
    axes[0, 0].legend()

    # PSD after
    axes[0, 1].semilogy(f_proc, pxx_proc, lw=0.8, color="orange")
    axes[0, 1].axvline(config.data.lower_freq,  color="r", ls="--")
    axes[0, 1].axvline(config.data.higher_freq, color="g", ls="--")
    axes[0, 1].set_xlim(0, proc_sr / 2)
    axes[0, 1].set_title(f"PSD after  (sr={proc_sr} Hz, z-normed)")
    axes[0, 1].set_xlabel("Frequency (Hz)")

    # per-channel mean
    ch_idx = np.arange(len(ch_mean_proc))
    axes[1, 0].bar(ch_idx, ch_mean_proc)
    axes[1, 0].axhline(0, color="r", lw=1)
    axes[1, 0].set_title("Per-channel mean after (should be ~0)")
    axes[1, 0].set_xlabel("Channel")

    # per-channel std
    axes[1, 1].bar(ch_idx, ch_std_proc, color="orange")
    axes[1, 1].axhline(1, color="r", lw=1)
    axes[1, 1].set_title("Per-channel std after (should be ~1)")
    axes[1, 1].set_xlabel("Channel")

    fig.suptitle(f"EEG QC  —  {sub_id} / {activity}", fontsize=12)
    plt.tight_layout()
    out = f"eeg_qc_{sub_id}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[QC] saved {out}")


def preprocess_dataset(config: TrainConfig):
    dataset_path = Path(config.data.data_dir)

    with h5py.File(config.data.output_h5, 'a') as h5f:
        for i in range(config.data.start_sub, config.data.end_sub + 1):
            sub_id = f"sub-{i:02d}"
            download_natview_subjects(start_sub=i, end_sub=i)

            if config.data.use_parcellation:
                fmri_files = list((dataset_path / sub_id).rglob("*" + config.data.parcellation_suffix + ".nii.gz"))
            else:
                fmri_files = list((dataset_path / sub_id).rglob(config.data.no_parcellation_prefix + "*.nii.gz"))

            first_eeg_done = False

            for f_path in tqdm(fmri_files, desc=f"{sub_id}"):
                if any(excl in f_path.parents[1].name for excl in config.data.excluded_activities):
                    continue

                f_path_p = f_path.relative_to(config.data.data_dir).parts
                activity_parts = f_path_p[3].split("_")
                task_part = activity_parts[2:3]
                run_part  = activity_parts[3:4]
                activity = "_".join(task_part + run_part) if run_part and "run" in run_part[0] else (task_part[0] if task_part else "unknown")

                grp = f"{sub_id}/{activity}"
                if f"{grp}/fmri" in h5f and f"{grp}/eeg" in h5f:
                    print(f"[SKIP] {grp}")
                    continue

                try:
                    fmri_data = preprocess_fmri(f_path, config)
                    fmri_data = fmri_data.to(torch.float16).permute(3, 0, 1, 2)

                    task_name = f_path.parents[1].name.replace("_bold", "")
                    eeg_npy = f_path.parents[3] / "eeg" / f"{task_name}_eeg.npy"
                    eeg_set = eeg_npy.with_suffix(".set")

                    if not first_eeg_done:
                        _eeg_qc_plot(eeg_set, config, sub_id, activity)
                        first_eeg_done = True

                    preprocess_eeg(eeg_set, config)
                    eeg_data = np.load(eeg_npy).astype(np.float16).T  # (T_eeg, C)

                    if f"{grp}/fmri" not in h5f:
                        fmri_win_trs = int(np.ceil(config.data.eeg_win_sec / config.data.tr))
                        h5f.create_dataset(f"{grp}/fmri", data=fmri_data,
                                           chunks=(fmri_win_trs, 96, 96, 96),
                                           compression="gzip", compression_opts=4)
                    if f"{grp}/eeg" not in h5f:
                        eeg_win_samples = int(config.data.eeg_win_sec * config.data.eeg_sr)
                        h5f.create_dataset(f"{grp}/eeg", data=eeg_data,
                                           chunks=(eeg_win_samples, eeg_data.shape[1]),
                                           compression="gzip", compression_opts=4)

                    print(f"[OK] {grp}")
                except Exception as e:
                    print(f"[ERR] {grp}: {e}")

            shutil.rmtree(dataset_path / sub_id)


def download_natview_subjects(
    start_sub=1,
    end_sub=22,
    out_dir="data",
    base_url="https://fcp-indi.s3.amazonaws.com/data/Projects/NATVIEW_EEGFMRI/preproc_data_gz"
):
    """
    Downloads NATVIEW EEG-fMRI dataset for subjects sub-01 ... sub-22.

    Archives contain paths like  projects/EEG_FMRI/data_indi_preproc/sub-XX/...
    Extracting into out_dir="data" produces  data/projects/EEG_FMRI/data_indi_preproc/sub-XX/...
    which matches config.data.data_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    data_dir = Path(out_dir) / "projects" / "EEG_FMRI" / "data_indi_preproc"

    for i in range(start_sub, end_sub + 1):
        sub_id = f"sub-{i:02d}"
        if (data_dir / sub_id).exists():
            print(f"[SKIP] {sub_id}")
            continue

        url = f"{base_url}/{sub_id}.tar.gz"
        tar_path = Path(out_dir) / f"{sub_id}.tar.gz"

        print(f"[DOWNLOAD] {sub_id}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(tar_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=sub_id
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"[EXTRACT] {sub_id}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=out_dir)
        tar_path.unlink()
        print(f"[DONE] {sub_id}")


def extract_all_tar_gz(out_dir="data"):
    """
    Extracts all *.tar.gz in out_dir.
    Archives must contain paths starting with projects/EEG_FMRI/data_indi_preproc/sub-XX/...
    so files land at out_dir/projects/EEG_FMRI/data_indi_preproc/sub-XX/...
    """
    for fname in os.listdir(out_dir):
        if not fname.endswith(".tar.gz"):
            continue
        tar_path = os.path.join(out_dir, fname)
        print(f"[EXTRACT] {fname}")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=out_dir)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

if __name__ == "__main__":
    config = TrainConfig()

    preprocess_dataset(config)