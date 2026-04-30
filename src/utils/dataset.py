from torch.utils.data import Dataset, Sampler
import numpy as np
import torch
from pathlib import Path
import nibabel as nib
import pandas as pd
import random
from collections import defaultdict
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random

from train.config import TrainConfig
from src.utils.preprocess_data import save_as_npy, downsample_eeg, bandpass_filter

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
        subjects: list = None,  # e.g. ["sub-01", "sub-02"]; None = all subjects
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

        self.subjects = set(subjects) if subjects is not None else None

        self.pairs = []   #list of dicts with eeg_path, fmri_path, n_tr, etc.
        self.index = []   #list of (pair_id, tr_idx)

        self.meta = []

        #self._preprocess()
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

            run = parts[2]
            if "run" in parts[3]:
                run += "_" + parts[3]

            sub = next(p for p in parts if p.startswith("sub-"))

            if self.subjects is not None and sub not in self.subjects:
                continue

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
                "activity": parts[2],
            })

            global_id += max_tr - min_tr + 1

            for tr_idx in range(min_tr, max_tr, self.config.stride_tr):
                self.index.append((pair_id, tr_idx))
                self.meta.append({
                    "sub": sub,
                    "run": run,  # unique per recording file, so temporal distance is only enforced within the same fMRI/EEG pair
                    "tr": tr_idx,
                })

        if len(self.index) == 0:
            raise RuntimeError("No fMRI/EEG pairs was found")
    def _get_mmap(self, path):
        """Return a cached mmap for path, opening it once per worker."""
        path = str(path)
        if not hasattr(self, "_mmap_cache"):
            self._mmap_cache = {}
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.load(path, mmap_mode="r")
        return self._mmap_cache[path]

    def __getitem__(self, idx):
        pair_id, tr_idx = self.index[idx]
        meta = self.meta[idx]
        info = self.pairs[pair_id]

        fmri = self._get_mmap(info["fmri_path"])  #[n_roi, t] or [x,y,z,t]
        eeg  = self._get_mmap(info["eeg_path"])   #[n_ch, t]
        
        tr_idx = min(max(tr_idx + random.choice(self.config.fmri_aug.tr_jitter), 0), fmri.shape[-1] - 1)
        print(tr_idx)

        t_fmri = tr_idx * self.config.tr

        #getting previous eeg window(accounting hrf_shift) prior to fmri window
        ts_eeg = t_fmri - np.array(self.config.hrf_shifts_sec)

        eeg_windows = [];

        fmri_start = tr_idx
        fmri_end = int(tr_idx + np.ceil(self.config.eeg_win_sec / self.config.tr))

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
            "eeg":torch.Tensor(eeg_windows),               #[K, C, T]
            "fmri": torch.Tensor(fmri_tr),              #[R, T] or [X,Y,Z,T]
            "tr_idx": int(tr_idx),
            "sub": meta["sub"],
            "run": meta["run"],
            "t_fmri": float(t_fmri),
            "ts_eeg": ts_eeg[0],
        }
        return sample

def collate_fn(batch):
    keys_to_remove = ["tr_idx", "sub", "run", "t_fmri", "ts_eeg"]
    batch = sorted(batch, key=lambda x: (x["tr_idx"], x["sub"]))
    batch = [{k: v for k, v in elem.items() if k not in keys_to_remove} for elem in batch]

    return {
        key: torch.stack([d[key] for d in batch])
        for key in batch[0]
    }

class ContrastiveBatchSampler(Sampler):
    def __init__(self, dataset: SimultEEG_fMRI, config: TrainConfig):
        """
        num_timestamps (M): Сколько разных моментов времени в одном батче.
        num_subjects (K): Сколько субъектов брать для каждого момента времени.
        margin_tr: Минимальное расстояние между таймкодами (15 TR = 31.5 сек).
        Размер итогового батча = M * K.
        """
        self.dataset = dataset
        self.M = config.data.num_timestamps
        self.K = config.data.num_subjects
        self.margin = config.data.margin_tr
        
        # Индексация датасета: run (видео) -> tr (таймкод) -> список глобальных индексов
        self.run_tr_indices = defaultdict(lambda: defaultdict(list))
        for idx, meta in enumerate(self.dataset.meta):
            # 'run' должен уникально идентифицировать конкретное видео (например, 'task-movie1_run-01')
            self.run_tr_indices[meta['run']][meta['tr']].append(idx)
            
    def __iter__(self):
        # Перемешиваем порядок видео, чтобы не учить одно видео подряд
        runs = list(self.run_tr_indices.keys())
        random.shuffle(runs)
        
        for run in runs:
            # Получаем все доступные таймкоды для этого видео
            available_trs = list(self.run_tr_indices[run].keys())
            random.shuffle(available_trs)
            
            # Пока есть доступные таймкоды, пытаемся собрать батч
            while available_trs:
                selected_trs = []
                
                # Жадный поиск M таймкодов с учетом margin
                for tr in available_trs:
                    # Проверяем, что текущий TR отстоит от всех уже выбранных минимум на margin
                    if all(abs(tr - sel_tr) >= self.margin for sel_tr in selected_trs):
                        selected_trs.append(tr)
                        
                    if len(selected_trs) == self.M:
                        break
                        
                # Если не смогли набрать M независимых таймкодов (конец видео), 
                # прерываем цикл для этого видео. Жесткий размер батча важен для стабильности лосса.
                if len(selected_trs) < self.M:
                    break
                    
                # Удаляем выбранные таймкоды из пула текущей эпохи
                for tr in selected_trs:
                    available_trs.remove(tr)
                    
                # Формируем итоговые индексы для батча
                batch_indices = []
                for tr in selected_trs:
                    subjs = self.run_tr_indices[run][tr]
                    
                    # Если субъектов больше K, берем случайные K
                    if len(subjs) >= self.K:
                        sampled_subjs = random.sample(subjs, self.K)
                    # Если меньше, берем сколько есть (или можно использовать замену)
                    else:
                        sampled_subjs = subjs 
                        
                    batch_indices.extend(sampled_subjs)
                    
                yield batch_indices

    def __len__(self):
        # Приблизительная оценка количества батчей для прогресс-бара Lightning
        # (Кол-во уникальных TR / margin) * кол-во видео // M
        total_batches = 0
        for run, tr_dict in self.run_tr_indices.items():
            max_tr = max(tr_dict.keys())
            min_tr = min(tr_dict.keys())
            # Оценка: сколько независимых таймкодов помещается в видео
            independent_trs = (max_tr - min_tr) // self.margin
            print(independent_trs)
            total_batches += independent_trs // self.M
        return total_batches