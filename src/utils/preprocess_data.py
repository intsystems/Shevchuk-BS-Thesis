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

    print(np.mean(data, axis=-1))

    np.save(npy_path, data)                     # overwrite with z-normed version
    return data

def preprocess_dataset(config: TrainConfig):
    for i in range(config.data.start_sub, config.data.end_sub + 1):
        download_natview_subjects(start_sub=i, end_sub=i, out_dir=config.data.data_dir)

        dataset_path = Path(config.data.data_dir)

        if config.data.use_parcellation:
            fmri_files = dataset_path.rglob("*" + config.data.parcellation_suffix + ".nii.gz")
        else:
            fmri_files = dataset_path.rglob(config.data.no_parcellation_prefix + "*.nii.gz")

        with h5py.File(config.data.output_h5, 'w') as h5f:
            for f_path in tqdm(fmri_files, desc="Конвертация в H5"):
                if any(excluded_activity in f_path.parents[1].name for excluded_activity in config.data.excluded_activities):
                    continue

                f_path_p = f_path.relative_to(config.data.data_dir).parts

                sub_id = f_path_p[0]
                activity = f_path_p[3].split("_")
                activity = activity[2] + "_" + activity[3] if "run" in activity[3] else activity[2]

                print(activity)
                print(sub_id)     

                try:
                    # fMRI: resample + crop + z-norm → (T, 96, 96, 96)
                    fmri_data = preprocess_fmri(f_path, config)
                    fmri_data = fmri_data.to(torch.float16).permute(3, 0, 1, 2)

                    task_name = f_path.parents[1].name.replace("_bold", "")
                    eeg_npy = f_path.parents[3] / "eeg" / f"{task_name}_eeg.npy"
                    eeg_set = eeg_npy.with_suffix(".set")
                    preprocess_eeg(eeg_set, config)
                    eeg_data = torch.Tensor(np.load(eeg_npy).astype(np.float16).T)  # (T_eeg, C)

                    grp = f"{sub_id}/{activity}"
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

                except Exception as e:
                    print(f"Ошибка в файле {f_path}: {e}")
        
        shutil.rmtree(dataset_path)


def download_natview_subjects(
    start_sub=1,
    end_sub=22,
    out_dir="data",
    base_url="https://fcp-indi.s3.amazonaws.com/data/Projects/NATVIEW_EEGFMRI/preproc_data_gz"
):
    """
    Downloads NATVIEW EEG-fMRI dataset for subjects sub-01 ... sub-22

    Arguments:
    start(int) - first subject index (default: 1)
    end(int) - last subject index (default: 22)
    out_dir(str) - directory to save downloaded files
    base_url(str) - base URL of dataset
    """

    os.makedirs(out_dir, exist_ok=True)

    for i in range(start_sub, end_sub + 1):
        sub_id = f"sub-{i:02d}"
        url = f"{base_url}/{sub_id}.tar.gz"
        out_path = os.path.join(out_dir, f"{sub_id}.tar.gz")

        sub_dir = os.path.join(out_dir, sub_id)
        if os.path.exists(sub_dir):
            print(f"[SKIP] {sub_id} already downloaded")
            continue

        print(f"[DOWNLOAD] {sub_id}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=sub_id
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"[EXTRACT] {sub_id}")
        with tarfile.open(out_path, "r:gz") as tar:
            tar.extractall(path=out_dir)
        os.remove(out_path)
        print(f"[DELETED] {sub_id}.tar.gz")

def extract_all_tar_gz(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for fname in os.listdir(data_dir):
        if fname.endswith(".tar.gz"):
            path = os.path.join(data_dir, fname)
            print(f"[EXTRACT] {fname}")
            try:
                with tarfile.open(path, "r:gz") as tar:
                    for member in tar.getmembers():
                        try:
                            # Create parent directories before extracting
                            if member.isfile():
                                member_path = os.path.join(data_dir, member.name)
                                member_dir = os.path.dirname(member_path)
                                os.makedirs(member_dir, exist_ok=True)
                                # Extract individual file
                                tar.extract(member, path=data_dir)
                            elif member.isdir():
                                member_path = os.path.join(data_dir, member.name)
                                os.makedirs(member_path, exist_ok=True)
                        except Exception as e:
                            print(f"  [WARN] Ошибка при извлечении {member.name}: {e}")
                            continue
            except Exception as e:
                print(f"  [ERROR] Ошибка при открытии {fname}: {e}")
                continue

if __name__ == "__main__":
    config = TrainConfig()

    preprocess_dataset(config)