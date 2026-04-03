import mne
import numpy as np
import os
import requests
from tqdm import tqdm
import tarfile

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

def save_as_npy(filename):
    """
    Used to transform EEG data to .npy format
    """
    signal = mne.io.read_raw_eeglab(filename, preload=True)

    picks = mne.pick_types(signal.info, eeg=True)

    eeg_data = signal.get_data(picks=picks)

    out_path = filename.with_suffix(".npy")
    np.save(out_path, eeg_data.astype(np.float32))

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
    for fname in os.listdir(data_dir):
        if fname.endswith(".tar.gz"):
            path = os.path.join(data_dir, fname)
            print(f"[EXTRACT] {fname}")
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(path=data_dir)