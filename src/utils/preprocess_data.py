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

def _process_subject(sub_id, fmri_files, tmp_path, config):
    """Write one subject into a fresh tmp h5. Returns True if any data was written."""
    wrote_any = False
    with h5py.File(tmp_path, 'w') as tmp:
        first_eeg_done = False
        for f_path in tqdm(fmri_files, desc=sub_id):
            if any(excl in f_path.parents[1].name for excl in config.data.excluded_activities):
                continue

            f_path_p = f_path.relative_to(config.data.data_dir).parts
            activity_parts = f_path_p[3].split("_")
            ses_id = activity_parts[1:2]
            task_part = activity_parts[2:3]
            run_part  = activity_parts[3:4]
            activity = "_".join(task_part + run_part) if run_part and "run" in run_part[0] else (task_part[0] if task_part else "unknown")

            grp = f"{sub_id}/{ses_id[0]}/{activity}"

            try:
                fmri_data = preprocess_fmri(f_path, config)
                fmri_data = fmri_data.to(torch.float16).permute(3, 0, 1, 2)

                task_name = f_path.parents[1].name.replace("_bold", "")
                eeg_npy = f_path.parents[3] / "eeg" / f"{task_name}_eeg.npy"
                eeg_set = eeg_npy.with_suffix(".set")


                preprocess_eeg(eeg_set, config)
                eeg_data = np.load(eeg_npy).astype(np.float16).T  # (T_eeg, C)

                tmp.create_dataset(f"{grp}/fmri", data=fmri_data,
                                   chunks=(1, 96, 96, 96),
                                   compression="lzf")
                tmp.create_dataset(f"{grp}/eeg", data=eeg_data,
                                   chunks=(int(config.data.tr * config.data.eeg_sr),
                                    eeg_data.shape[1]),
                                   compression="lzf")
                tmp.flush()
                wrote_any = True
                print(f"[OK] {grp}")
            except Exception as e:
                print(f"[ERR] {grp}: {e}")
    return wrote_any


def preprocess_dataset(config: TrainConfig):
    dataset_path = Path(config.data.data_dir)
    h5_path  = Path(config.data.output_h5)
    tmp_path = h5_path.with_suffix(".tmp.h5")

    for i in range(config.data.start_sub, config.data.end_sub + 1):
        sub_id = f"sub-{i:02d}"

        # skip if already in the main file; remove if corrupted
        if h5_path.exists():
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    if sub_id in h5f:
                        print(f"[SKIP] {sub_id}")
                        continue
            except OSError:
                print(f"[WARN] {h5_path} corrupted — removing")
                h5_path.unlink()

        download_natview_subjects(start_sub=i, end_sub=i)

        if config.data.use_parcellation:
            fmri_files = list((dataset_path / sub_id).rglob("*" + config.data.parcellation_suffix + ".nii.gz"))
        else:
            fmri_files = list((dataset_path / sub_id).rglob(config.data.no_parcellation_prefix + "*.nii.gz"))

        # process into a temp file — main file stays untouched until success
        wrote = _process_subject(sub_id, fmri_files, tmp_path, config)

        if wrote:
            # merge tmp → main
            with h5py.File(tmp_path, 'r') as tmp, h5py.File(h5_path, 'a') as h5f:
                for key in tmp.keys():
                    if key not in h5f:
                        tmp.copy(key, h5f)
                h5f.flush()
            print(f"[MERGED] {sub_id} → {h5_path}")

        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(dataset_path / sub_id, ignore_errors=True)


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