from torch.utils.data import Dataset, Sampler
import numpy as np
from pathlib import Path
import nibabel as nib
import pandas as pd
import random
from collections import defaultdict
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from train.config import TrainConfig
from src.utils.preprocess_data import save_as_npy, downsample_eeg, bandpass_filter, transform_to_neurostorm_space

def _convert_tsv(tsv_path):
    arr = pd.read_csv(tsv_path, sep="\t", header=None).to_numpy(dtype=np.float32)
    np.save(tsv_path.with_suffix(".npy"), arr)
    return tsv_path.name


def _convert_nii(args):
    import torch
    torch.set_num_threads(1)
    from src.utils.preprocess_data import transform_to_neurostorm_space

    file, scaling_factor = args
    out = file.with_suffix("").with_suffix(".npy")  # .nii.gz -> .npy
    img = nib.load(file)
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = transform_to_neurostorm_space(data, scaling_factor)
    np.save(out, data)
    return file.name


class SimultEEG_fMRI(Dataset):
    def __init__(
        self,
        config: TrainConfig,  
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
        self.config = config.data

        self.dirname = self.config.data_dir
        self.root = Path(self.dirname).resolve()

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
        to_find = self.config.no_parcellation_prefix
        if self.config.use_parcellation:
            to_find = self.config.parcellation_suffix
        
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
            if self.config.use_parcellation:
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
            
            t_min = np.max(self.config.hrf_shifts_sec)
            min_tr = int(np.ceil(t_min / self.config.tr))
            max_tr = int(np.ceil(n_tr - np.ceil(self.config.eeg_win_sec / self.config.tr) - np.max(self.config.hrf_shifts_sec)))

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

            for tr_idx in range(min_tr, max_tr, self.config.stride_tr):
                self.index.append((pair_id, tr_idx))
                self.meta.append({
                    "sub": sub,
                    "run": pair_id,  # unique per recording file, so temporal distance is only enforced within the same fMRI/EEG pair
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
        if self.config.use_parcellation == True:
            tsv_files = [
                p for p in DATA_ROOT.rglob(f"*{self.config.parcellation_suffix}*.tsv")
                if not p.with_suffix(".npy").exists()
            ]
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(_convert_tsv, p): p for p in tsv_files}
                for future in tqdm(as_completed(futures), total=len(tsv_files), desc="tsv->npy"):
                    future.result()
        else:
            scaling_factor = self.config.orig_fmri_res / self.config.target_fmri_res

            nii_files = [
                f for func_dir in func_dirs
                for f in func_dir.rglob(f"{self.config.no_parcellation_prefix}*.nii.gz")
                if not f.with_suffix("").with_suffix(".npy").exists()
            ]
            for f in tqdm(nii_files, desc="nii.gz->npy"):
                _convert_nii((f, scaling_factor))

        # --- 2. EEG PREPROCESSING & NORMALIZATION ---
        eeg_dirs = DATA_ROOT.glob("sub-*/ses-*/eeg")
        for eeg_root in eeg_dirs:
            pattern = f"*.set"
            for file_path in eeg_root.rglob(pattern):
                if "checkeroff" in file_path.name or "checkerout" in file_path.name or "checker_recording" in file_path.name:
                    continue
                
                bandpass_filter(file_path, self.config.lower_freq, self.config.higher_freq)
                downsample_eeg(file_path, self.config.target_eeg_freq)
                save_as_npy(file_path)

    def __getitem__(self, idx):
        pair_id, tr_idx = self.index[idx]
        info = self.pairs[pair_id]

        fmri = np.load(info["fmri_path"], mmap_mode="r")  #[n_roi, t] or [x,y,z,t]
        eeg = np.load(info["eeg_path"], mmap_mode="r")    #[n_ch, t]
        
        t_fmri = tr_idx * self.config.tr

        #getting previous eeg window(accounting hrf_shift) prior to fmri window
        ts_eeg = t_fmri - np.array(self.config.hrf_shifts_sec)

        eeg_windows = [];

        fmri_start = tr_idx
        fmri_end = int(tr_idx + np.ceil(self.config.eeg_win_sec / self.config.tr))

        print(fmri_start)
        print(fmri_end)
        print(type(fmri_end))

        if self.config.use_parcellation:
            fmri_tr = np.array(fmri[:, fmri_start:fmri_end], dtype=np.float32)
        else:
            fmri_tr = np.array(fmri[:, :, :, fmri_start:fmri_end], dtype=np.float32)

        n_samples = int(self.config.eeg_win_sec * self.config.eeg_sr)
        n_ch = self.config.n_eeg_channels
        
        for t_eeg in ts_eeg:
            ind_start = int(round(t_eeg * self.config.eeg_sr))
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
            "fmri": fmri_tr,              #[R, T] or [X,Y,Z,T]
            "tr_idx": int(tr_idx),
            "sub": info["sub"],
            "ses": info["ses"],
            "t_fmri": float(t_fmri),
            "ts_eeg": ts_eeg,
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
        max_batches=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.subs_per_batch = subs_per_batch
        self.min_temp_dist = min_temp_dist
        self.drop_last = drop_last
        self.max_batches = max_batches

        assert batch_size % subs_per_batch == 0
        self.M = batch_size // subs_per_batch

        #idx grouping by subject
        self.by_sub = defaultdict(list)
        for idx, meta in enumerate(dataset.meta):
            self.by_sub[meta["sub"]].append(idx)

        self.subs = list(self.by_sub.keys())

    def __iter__(self):
        if len(self.subs) < self.subs_per_batch:
            return

        n_yielded = 0
        max_attempts = 1000

        for _ in range(max_attempts):
            if self.max_batches is not None and n_yielded >= self.max_batches:
                break

            chosen_subs = random.sample(self.subs, self.subs_per_batch)
            batch = []
            success = True

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
                    success = False
                    break

                batch.extend(picked)

            if success and len(batch) == self.batch_size:
                yield batch
                n_yielded += 1
            elif not self.drop_last and len(batch) > 0:
                yield batch
                n_yielded += 1

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
