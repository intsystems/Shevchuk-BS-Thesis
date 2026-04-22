import mne
import numpy as np
import os
import requests
from tqdm import tqdm
import tarfile
import torch

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

def transform_to_neurostorm_space(fmri_volume: np.ndarray, scaling_factor: float = 1.5) -> np.ndarray:
    """
    Апсемплинг fMRI с шага 3мм до 2мм и приведение к кубу 96x96x96.
    Ожидает на входе numpy массив формы (61, 73, 61, T).
    """
    # 1. Конвертация и перестановка в формат, который требует F.interpolate
    # (Batch, Channel, Depth, Height, Width) -> (T, 1, X, Y, Z)
    x = torch.tensor(fmri_volume, dtype=torch.float32)
    x = x.permute(3, 0, 1, 2).unsqueeze(1) 
    
    # 2. Пространственное ресемплирование (3mm -> 2mm)
    # Коэффициент масштабирования = 3/2 = 1.5. 
    # Размер 61x73x61 превратится примерно в 92x110x92
    x_resampled = F.interpolate(
        x, 
        scale_factor=scaling_factor, 
        mode='trilinear', 
        align_corners=False
    )
    
    _, _, new_X, new_Y, new_Z = x_resampled.shape
    target = 96
    
    # 3. Симметричный Padding (Дополнение нулями там, где < 96)
    # F.pad принимает отступы с конца: (Z_left, Z_right, Y_left, Y_right, X_left, X_right)
    pad_Z_left = max((target - new_Z) // 2, 0)
    pad_Z_right = max(target - new_Z - pad_Z_left, 0)
    
    pad_Y_left = max((target - new_Y) // 2, 0)
    pad_Y_right = max(target - new_Y - pad_Y_left, 0)
    
    pad_X_left = max((target - new_X) // 2, 0)
    pad_X_right = max(target - new_X - pad_X_left, 0)
    
    x_padded = F.pad(
        x_resampled, 
        (pad_Z_left, pad_Z_right, pad_Y_left, pad_Y_right, pad_X_left, pad_X_right), 
        mode='constant', 
        value=0.0  # Нули сработают как нейтральный фон для NeuroSTORM
    )
    
    # 4. Center Crop (Обрезка лишнего там, где > 96, например по оси Y)
    crop_X_start = max((x_padded.shape[2] - target) // 2, 0)
    crop_Y_start = max((x_padded.shape[3] - target) // 2, 0)
    crop_Z_start = max((x_padded.shape[4] - target) // 2, 0)
    
    x_final = x_padded[
        :, :,
        crop_X_start : crop_X_start + target,
        crop_Y_start : crop_Y_start + target,
        crop_Z_start : crop_Z_start + target
    ]
    
    # 5. Возврат к исходному формату (X, Y, Z, T) для сохранения на диск
    return x_final.squeeze(1).permute(1, 2, 3, 0).numpy()

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