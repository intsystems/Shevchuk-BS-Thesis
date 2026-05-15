import os
import re
import copy
import json
import random
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from train.config import TrainConfig


class SimultEEG_fMRI(Dataset):
    def __init__(self, config: TrainConfig, subjects: list = None):
        super().__init__()
        self.config   = config.data
        self.subjects = set(subjects) if subjects is not None else None

        # per-recording channel list: key 'sub-XX_ses-XX_task-..._run-XX'
        with open(self.config.channels_json, "r", encoding="utf-8") as f:
            self.channels_per_rec = json.load(f)

        self.pairs = []   # list of dicts
        self.index = []   # list of (pair_id, tr_idx)
        self.meta  = []

        self._build_pairs_and_index()

    @classmethod
    def from_subset(cls, dataset, indices):
        new_ds = cls.__new__(cls)
        new_ds.__dict__ = copy.deepcopy(dataset.__dict__)
        new_ds.index = [dataset.index[i] for i in indices]
        new_ds.meta  = [dataset.meta[i]  for i in indices]
        return new_ds

    def __len__(self):
        return len(self.index)

    def _build_pairs_and_index(self):
        win_trs     = int(np.ceil(self.config.eeg_win_sec / self.config.tr))
        min_tr_off  = int(np.ceil(np.max(self.config.hrf_shifts_sec) / self.config.tr))

        with h5py.File(self.config.output_h5, 'r') as h5f:
            for sub_id in sorted(h5f.keys()):
                if self.subjects is not None and sub_id not in self.subjects:
                    continue

                sub_grp = h5f[sub_id]

                # support both sub/activity and sub/ses/activity structures
                grp_paths = []  # list of (h5_path_str, act_key)
                for level2_key in sorted(sub_grp.keys()):
                    level2 = sub_grp[level2_key]
                        # sub/ses/activity
                    for act_key in sorted(level2.keys()):
                        act = level2[act_key]
                        if "fmri" in act and "eeg" in act:
                            grp_paths.append((f"{sub_id}/{level2_key}/{act_key}", act_key))

                for h5_grp_path, act_key in grp_paths:
                    grp  = h5f[h5_grp_path]
                    fmri = grp["fmri"]
                    eeg  = grp["eeg"]

                    # look up native channel list for this recording
                    rec_key = h5_grp_path.replace("/", "_")   # 'sub-03_ses-01_task-dme_run-01'
                    rec_channels = self.channels_per_rec.get(rec_key)
                    if rec_channels is None:
                        # no channel info — skip rather than guess
                        continue
                    # h5 stores eeg as (T, C). Trust C from h5; if it doesn't match the JSON
                    # length the JSON is from a different preprocessing run — skip.
                    if eeg.shape[1] != len(rec_channels):
                        continue

                    n_tr = fmri.shape[0] if not self.config.use_parcellation else fmri.shape[1]

                    min_tr = min_tr_off

                    # max_tr is bounded by both fMRI and EEG length:
                    # last EEG sample needed is (t_fmri - min(hrf_shifts) + eeg_win_sec) * eeg_sr
                    max_tr_fmri = n_tr - win_trs
                    max_t_fmri  = (eeg.shape[0] / self.config.eeg_sr
                                   + min(self.config.hrf_shifts_sec)
                                   - self.config.eeg_win_sec)
                    max_tr_eeg  = int(max_t_fmri / self.config.tr)
                    max_tr      = min(max_tr_fmri, max_tr_eeg)

                    if max_tr <= min_tr:
                        continue

                    # strip _run-XX for the activity label; keep full path as run id
                    activity = re.sub(r'_run-\d+', '', act_key)

                    pair_id = len(self.pairs)
                    self.pairs.append({
                        "h5_grp": h5_grp_path,
                        "n_tr":   max_tr - min_tr,
                        "run":    h5_grp_path,   # unique per recording → used by ContrastiveBatchSampler
                        "sub":    sub_id,
                        "activity": activity,
                        "channels": rec_channels,   # list of channel names in h5 order
                    })

                    for tr_idx in range(min_tr, max_tr, self.config.stride_tr):
                        self.index.append((pair_id, tr_idx))
                        self.meta.append({"sub": sub_id, "run": act_key, "activity": activity, "tr": tr_idx})

        if len(self.index) == 0:
            raise RuntimeError("No fMRI/EEG pairs found in h5 file")

    def _get_h5(self):
        """Per-process cached h5 file handle (safe with DataLoader workers)."""
        pid = os.getpid()
        cache = getattr(self, "_h5_cache", None)
        if cache is None or cache["pid"] != pid:
            if cache is not None:
                try:
                    cache["file"].close()
                except Exception:
                    pass
            self._h5_cache = {"pid": pid, "file": h5py.File(self.config.output_h5, 'r')}
        return self._h5_cache["file"]

    def __getitem__(self, idx):
        pair_id, tr_idx = self.index[idx]
        meta = self.meta[idx]
        info = self.pairs[pair_id]

        h5f  = self._get_h5()
        grp  = h5f[info["h5_grp"]]
        fmri = grp["fmri"]   # (T, 96, 96, 96) or (n_roi, T)
        eeg  = grp["eeg"]    # (T_eeg, C)

        n_fmri_frames = fmri.shape[0] if not self.config.use_parcellation else fmri.shape[1]
        tr_idx = min(max(tr_idx + random.choice(self.config.fmri_aug.tr_jitter), 0),
                     n_fmri_frames - 1)

        t_fmri = tr_idx * self.config.tr
        ts_eeg = t_fmri - np.array(self.config.hrf_shifts_sec)

        fmri_start = tr_idx
        fmri_end   = int(tr_idx + np.ceil(self.config.eeg_win_sec / self.config.tr))

        if self.config.use_parcellation:
            fmri_tr = fmri[:, fmri_start:fmri_end].astype(np.float32)   # (n_roi, T_win)
        else:
            fmri_tr = fmri[fmri_start:fmri_end].astype(np.float32)      # (T_win, X, Y, Z)
            # downstream code (FmriAugmentor, NeuroSTORM) expects T as the last dim
            fmri_tr = np.transpose(fmri_tr, (1, 2, 3, 0))                # (X, Y, Z, T_win)

        n_samples = int(self.config.eeg_win_sec * self.config.eeg_sr)
        n_ch      = eeg.shape[1]              # native channel count for this recording
        eeg_len   = eeg.shape[0]

        eeg_windows = []
        for t_eeg in ts_eeg:
            ind_start = int(round(t_eeg * self.config.eeg_sr))
            ind_end   = ind_start + n_samples

            # clamp to valid range and read
            r_start = max(ind_start, 0)
            r_end   = min(ind_end, eeg_len)
            chunk   = eeg[r_start:r_end, :].astype(np.float32)   # (t, C)
            chunk   = chunk.T                                      # (C, t)

            # pad if the window goes out of bounds (time axis only, no channel padding)
            pad_left  = max(0, -ind_start)
            pad_right = max(0, ind_end - eeg_len)
            if pad_left or pad_right:
                chunk = np.pad(chunk, ((0, 0), (pad_left, pad_right)))

            if chunk.shape != (n_ch, n_samples):
                raise RuntimeError(
                    f"bad chunk shape {chunk.shape}: expected ({n_ch}, {n_samples}); "
                    f"eeg.shape={tuple(eeg.shape)}, eeg_len={eeg_len}, "
                    f"ind=[{ind_start}:{ind_end}], r=[{r_start}:{r_end}], "
                    f"pad=(L={pad_left}, R={pad_right}), h5_grp={info['h5_grp']}, t_eeg={t_eeg}"
                )

            eeg_windows.append(chunk)

        return {
            "eeg":      torch.tensor(np.stack(eeg_windows)),  # (K, C_native, T)
            "fmri":     torch.tensor(fmri_tr),                # (X, Y, Z, T_win) or (n_roi, T_win)
            "ch_names": info["channels"],                     # list[str], len = C_native
            "tr_idx":   int(tr_idx),
            "sub":      meta["sub"],
            "run":      meta["run"],
            "t_fmri":   float(t_fmri),
            "ts_eeg":   float(ts_eeg[0]),
        }


def collate_fn(batch):
    """
    Each sample may have its own EEG channel set (different recordings drop different
    channels). To produce a single (B, C, T) tensor for LaBraM we restrict every sample
    to the *intersection* of channel names present in the batch and select those
    channels in a consistent canonical order.

    The returned 'ch_names' is the per-batch channel list (Python list of strings)
    that the encoder uses to build LaBraM's input_chans positional-embedding indices.
    """
    batch = sorted(batch, key=lambda x: (x["tr_idx"], x["sub"]))

    # 1. intersection of channels across the batch
    common = set(batch[0]["ch_names"])
    for s in batch[1:]:
        common &= set(s["ch_names"])
    if not common:
        raise RuntimeError("empty channel intersection in batch")

    # 2. canonical order: by order of appearance in the first sample's ch_names
    first_order = batch[0]["ch_names"]
    common_ordered = [c for c in first_order if c in common]

    # 3. for each sample, gather the common channels in canonical order
    eeg_tensors  = []
    fmri_tensors = []
    for s in batch:
        # column index of each common channel in this sample's native list
        idx = torch.tensor(
            [s["ch_names"].index(c) for c in common_ordered],
            dtype=torch.long,
        )
        eeg = s["eeg"]                       # (K, C_native, T)
        eeg = eeg.index_select(dim=-2, index=idx)   # (K, |common|, T)
        eeg_tensors.append(eeg)
        fmri_tensors.append(s["fmri"])

    return {
        "eeg":      torch.stack(eeg_tensors),
        "fmri":     torch.stack(fmri_tensors),
        "ch_names": common_ordered,
    }


class ContrastiveBatchSampler(Sampler):
    def __init__(self, dataset: SimultEEG_fMRI, config: TrainConfig):
        self.dataset = dataset
        self.M      = config.data.num_timestamps
        self.K      = config.data.num_subjects
        self.margin = config.data.margin_tr

        self.run_tr_indices = defaultdict(lambda: defaultdict(list))
        for idx, meta in enumerate(self.dataset.meta):
            self.run_tr_indices[meta['run']][meta['tr']].append(idx)

    def __iter__(self):
        runs = list(self.run_tr_indices.keys())
        random.shuffle(runs)

        for run in runs:
            # only TRs with ≥ K subjects are usable — guarantees fixed batch shape (M, K)
            available_trs = [tr for tr, subjs in self.run_tr_indices[run].items()
                             if len(subjs) >= self.K]
            random.shuffle(available_trs)

            while available_trs:
                selected_trs = []
                for tr in available_trs:
                    if all(abs(tr - s) >= self.margin for s in selected_trs):
                        selected_trs.append(tr)
                    if len(selected_trs) == self.M:
                        break

                if len(selected_trs) < self.M:
                    break

                for tr in selected_trs:
                    available_trs.remove(tr)

                batch_indices = []
                for tr in selected_trs:
                    subjs = self.run_tr_indices[run][tr]
                    sampled = random.sample(subjs, self.K)
                    batch_indices.extend(sampled)

                yield batch_indices

    def __len__(self):
        total = 0
        for run, tr_dict in self.run_tr_indices.items():
            span = max(tr_dict.keys()) - min(tr_dict.keys())
            total += (span // self.margin) // self.M
        return total