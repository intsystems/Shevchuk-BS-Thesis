from torch.utils.data import Dataset, Sampler
import numpy as np
from pathlib import Path
import nibabel as nib
import random
from collections import defaultdict
import copy

from src import config
from src.utils.preprocess_data import save_as_npy, bandpass_filter

class SimultEEG_fMRI(Dataset):
    def __init__(
        self,
        dirname,
        eeg_sr=250,
        tr=2.1,
        eeg_win_sec=2,
        hrf_shifts_sec=[5.0],
        stride_tr=1,
        min_margin_sec=0.0,
        n_eeg_channels=60,
    ):
        """
        Arguments:
        dirname(str) -- name of the directory where sub-* directories are located
        eeg_sr(int) -- sampling rate in Hz
        tr(float) -- how fast voxel in fMRI is updated (in seconds)
        eeg_win_sec(float) -- size of the window for EEG, can be equal to tr
        hrf_shifts_sec(float) -- array of shifts(for multiple positives), each element ishow much to shift to the left to capture eeg signal, because fmri and eeg are not aligned in time
        n_eeg_channels(int) -- number of eeg channels
        """
        super().__init__();

        self.dirname = dirname
        self.root = Path(dirname).resolve()
        self.eeg_sr = float(eeg_sr)
        self.tr = float(tr)
        self.eeg_win_sec = float(eeg_win_sec)
        self.hrf_shifts_sec = np.array(hrf_shifts_sec)
        self.stride_tr = int(stride_tr)
        self.min_margin_sec = float(min_margin_sec)
        self.n_eeg_channels = int(n_eeg_channels)

        self.pairs = []   #list of dicts with eeg_path, fmri_path, n_tr, etc.
        self.index = []   #list of (pair_id, tr_idx)

        self.meta = []

        self._preprocess()
        self._build_pairs_and_index()
    
    @classmethod
    def from_subset(cls, dataset, indices):
        new_ds = cls.__new__(cls)
        new_ds.__dict__ = copy.deepcopy(dataset.__dict__)
        new_ds.index = [dataset.index[i] for i in indices]
        new_ds.meta  = [dataset.meta[i] for i in indices]
        return new_ds

    def __len__(self):
        return len(self.index)

    def _build_pairs_and_index(self):
        to_find = config.no_parcellation_prefix
        if config.use_parcellation == True:
            to_find = config.parcellation_suffix
        
        fmri_npy_files = list(self.root.rglob(f"*{to_find}*.npy"))
        global_id = 0
        for fmri_path in fmri_npy_files:
            corresp_name  = fmri_path.parents[1].name

            #finding corresponding eeg for fmri
            eeg_candidates = sorted(self.root.rglob(f"{corresp_name[:-5]}_eeg.npy"))
            if len(eeg_candidates) == 0:
                continue
            eeg_path = eeg_candidates[0]

            fmri = np.load(fmri_path, mmap_mode="r")

            #number of frames
            if config.use_parcellation == True:
                n_tr = fmri.shape[1]
            else:
                n_tr = fmri.shape[3]

            del fmri

            parts = corresp_name.split("_")

            run = 1
            if "run" in parts[3]:
                run = parts[3][-1:]

            sub = next(p for p in parts if p.startswith("sub-"))
            ses = next(p for p in parts if p.startswith("ses-"))

            pair_id = len(self.pairs)
            
            t_min = self.min_margin_sec + np.max(self.hrf_shifts_sec) + self.eeg_win_sec / 2
            min_tr = int(np.ceil(t_min / self.tr))
            max_tr = n_tr - int(np.ceil(self.min_margin_sec + self.eeg_win_sec / 2))

            self.pairs.append({
                "eeg_path": eeg_path,
                "fmri_path": fmri_path,
                "n_tr": max_tr - min_tr + 1,
                "run": run,
                "sub": sub,
                "ses": ses,
                "activity": parts[2],
            })

            global_id += max_tr - min_tr + 1

            for tr_idx in range(min_tr, max_tr, self.stride_tr):
                self.index.append((pair_id, tr_idx))
                self.meta.append({
                    "sub": sub,
                    "run": run,
                    "tr": tr_idx,
                })

        if len(self.index) == 0:
            raise RuntimeError("No fMRI/EEG pairs was found")

    def _preprocess(self):
        """
            Detects EEG data, applies filterband, transforms data to .npy, and Z-normalizes
            Detects fMRI data, transforms it to .npy, and Z-normalizes
            Then forms slices of the data based on TR of fMRI
        """

        DATA_ROOT = Path(self.dirname).resolve()
        func_dirs = DATA_ROOT.glob("sub-*/ses-*/func")

        # --- 1. fMRI PREPROCESSING & NORMALIZATION ---
        if config.use_parcellation == True:
            #gather .tsv files and transform them into .npy
            for func_root in func_dirs:
                pattern = f"*{config.parcellation_suffix}*.tsv"
                for tsv_path in func_root.rglob(pattern):
                    arr = np.loadtxt(tsv_path, delimiter="\t")
                    
                    # Assume shape is [n_roi, time] based on your __getitem__
                    # Z-normalize per ROI across the entire time run
                    mean_roi = np.mean(arr, axis=1, keepdims=True)
                    std_roi = np.std(arr, axis=1, keepdims=True)
                    arr_norm = (arr - mean_roi) / (std_roi + 1e-8)
                    
                    np.save(tsv_path.with_suffix(".npy"), arr_norm)
                    print(f"OK (Z-scored): {tsv_path}")
        else:
            #without parcellation we must extract raw fmri data
            func_preproc_dirs = func_dirs.glob("func_preproc")
            for dir in func_preproc_dirs:
                pattern = f"{config.no_parcellation_prefix}*.nii.gz"
                for file in dir.rglob(pattern):
                    img = nib.load(file)
                    data = img.get_fdata()

                    # Assume shape is [X, Y, Z, time] based on your __getitem__
                    # Z-normalize per voxel across the entire time run
                    mean_vox = np.mean(data, axis=3, keepdims=True)
                    std_vox = np.std(data, axis=3, keepdims=True)
                    data_norm = (data - mean_vox) / (std_vox + 1e-8)

                    np.save(file.with_suffix(".npy"), data_norm)
                    print(f"OK (Z-scored): {file.name}")

        # --- 2. EEG PREPROCESSING & NORMALIZATION ---
        eeg_dirs = DATA_ROOT.glob("sub-*/ses-*/eeg")
        for eeg_root in eeg_dirs:
            pattern = f"*.set"
            for file_path in eeg_root.rglob(pattern):
                if "checkeroff" in file_path.name or "checkerout" in file_path.name or "checker_recording" in file_path.name:
                    continue
                
                bandpass_filter(file_path, config.lower_freq, config.higher_freq)
                save_as_npy(file_path)
                
                # Load the newly created .npy to apply Z-normalization
                npy_path = file_path.with_suffix(".npy")
                if npy_path.exists():
                    eeg_data = np.load(npy_path)
                    
                    # Assume shape is [n_ch, time]
                    # Z-normalize per channel across the entire recording
                    mean_ch = np.mean(eeg_data, axis=1, keepdims=True)
                    std_ch = np.std(eeg_data, axis=1, keepdims=True)
                    eeg_norm = (eeg_data - mean_ch) / (std_ch + 1e-8)
                    
                    np.save(npy_path, eeg_norm)
                    print(f"OK (Filtered & Z-scored): {npy_path.name}")

    def __getitem__(self, idx):
        pair_id, tr_idx = self.index[idx]
        info = self.pairs[pair_id]

        fmri = np.load(info["fmri_path"], mmap_mode="r")  #[n_roi, t] or [x,y,z,t]
        eeg = np.load(info["eeg_path"], mmap_mode="r")    #[n_ch, t]
        
        t_fmri = tr_idx * self.tr

        #getting previous eeg window(accounting hrf_shift) prior to fmri window
        t_center = t_fmri - self.hrf_shifts_sec

        eeg_windows = [];

        if config.use_parcellation:
            fmri_tr = np.array(fmri[:, tr_idx], dtype=np.float32)
        else:
            fmri_tr = np.array(fmri_tr[:, :, :, tr_idx], dtype=np.float32)

        n_samples = int(self.eeg_win_sec * self.eeg_sr)
        n_ch = self.n_eeg_channels
        
        for center in t_center:          
            half = 0.5 * self.eeg_win_sec
            t_start = center - half

            ind_start = int(round(t_start * self.eeg_sr))
            ind_end = ind_start + n_samples

            eeg_win = np.array(eeg[:, ind_start:ind_end], dtype=np.float32)
            
            if eeg_win.shape[0] > n_ch:
                eeg_win = eeg_win[:n_ch, :]
            elif eeg_win.shape[0] < n_ch:
                pad_ch = n_ch - eeg_win.shape[0]
                eeg_win = np.pad(eeg_win, ((0, pad_ch), (0, 0)), mode='constant')
            
            if eeg_win.shape[1] < n_samples:
                pad_t = n_samples - eeg_win.shape[1]
                eeg_win = np.pad(eeg_win, ((0, 0), (0, pad_t)), mode='constant')

            eeg_windows.append(eeg_win)
            
        sample = {
            "eeg": eeg_windows,               #[K, C, T]
            "fmri": fmri_tr,              #[R]
            "tr_idx": int(tr_idx),
            "sub": info["sub"],
            "ses": info["ses"],
            "t_fmri": float(t_fmri),
            "t_eeg_centers": t_center,
        }
        return sample

#custom batch sampler to support correct batch formation for contrastive learning
class ContrastiveBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size=128,
        subs_per_batch=8,
        min_temp_dist=5,
        drop_last=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.subs_per_batch = subs_per_batch
        self.min_temp_dist = min_temp_dist
        self.drop_last = drop_last

        assert batch_size % subs_per_batch == 0
        self.M = batch_size // subs_per_batch

        #idx grouping by subject
        self.by_sub = defaultdict(list)
        for idx, meta in enumerate(dataset.meta):
            self.by_sub[meta["sub"]].append(idx)

        self.subs = list(self.by_sub.keys())

    def __iter__(self):
        subs = self.subs[:]
        random.shuffle(subs)

        while True:
            if len(subs) < self.subs_per_batch:
                break

            chosen_subs = random.sample(subs, self.subs_per_batch)
            batch = []

            for sub in chosen_subs:
                indices = self.by_sub[sub][:]
                random.shuffle(indices)

                picked = []
                picked_meta = []

                for idx in indices:
                    meta = self.dataset.meta[idx]

                    ok = True
                    for m in picked_meta:
                        if (
                            meta["run"] == m["run"]
                            and abs(meta["tr"] - m["tr"]) < self.min_temp_dist
                        ):
                            ok = False
                            break

                    if ok:
                        picked.append(idx)
                        picked_meta.append(meta)

                    if len(picked) == self.M:
                        break

                if len(picked) < self.M:
                    break

                batch.extend(picked)

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                yield batch
            else:
                break

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
