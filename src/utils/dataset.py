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
        self.config              = config.data
        self.using_aug           = config.train.using_aug
        self.use_precomputed_fmri = config.model.use_precomputed_fmri
        self.fmri_features_h5   = config.model.fmri_features_h5
        self.subjects            = set(subjects) if subjects is not None else None

        # per-recording channel list: key 'sub-XX_ses-XX_task-..._run-XX'
        with open(self.config.channels_json, "r", encoding="utf-8") as f:
            self.channels_per_rec = json.load(f)

        self.pairs = []   # list of dicts
        self._index_list = []   # list of (pair_id, tr_idx), converted to np array after build
        self.meta  = []

        self._build_pairs_and_index()

    @classmethod
    def from_subset(cls, dataset, indices):
        new_ds = cls.__new__(cls)
        new_ds.__dict__ = copy.deepcopy(dataset.__dict__)
        new_ds.index = dataset.index[indices]          # numpy fancy index → new contiguous array
        new_ds.meta  = [dataset.meta[i] for i in indices]
        return new_ds

    def __len__(self):
        return len(self.index)

    def _build_pairs_and_index(self):
        win_trs     = int(np.ceil(self.config.eeg_win_sec / self.config.tr))
        min_tr_off  = int(np.ceil((np.max(self.config.hrf_shifts_sec) + self.config.eeg_win_sec) / self.config.tr))

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
                    # last EEG sample needed is (t_fmri - min(hrf_shifts)) * eeg_sr
                    max_tr_fmri = n_tr - win_trs
                    max_t_fmri  = (eeg.shape[0] / self.config.eeg_sr
                                   + min(self.config.hrf_shifts_sec))
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
                        self._index_list.append((pair_id, tr_idx))
                        self.meta.append({"sub": sub_id, "run": act_key, "activity": activity, "tr": tr_idx})

        if len(self._index_list) == 0:
            raise RuntimeError("No fMRI/EEG pairs found in h5 file")
        # numpy array avoids Python ref-count copy-on-write in DataLoader workers
        self.index = np.array(self._index_list, dtype=np.int64)  # (N, 2)
        del self._index_list

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
            # rdcc_nbytes: limit chunk cache to 64 MB per worker (default is unlimited growth)
            self._h5_cache = {"pid": pid, "file": h5py.File(
                self.config.output_h5, 'r', rdcc_nbytes=64 * 1024 * 1024
            )}
        return self._h5_cache["file"]

    def _get_features_h5(self):
        """Per-process cached handle for precomputed fMRI features."""
        pid = os.getpid()
        cache = getattr(self, "_feat_cache", None)
        if cache is None or cache["pid"] != pid:
            if cache is not None:
                try:
                    cache["file"].close()
                except Exception:
                    pass
            self._feat_cache = {"pid": pid, "file": h5py.File(self.fmri_features_h5, 'r')}
        return self._feat_cache["file"]

    def __getitem__(self, idx):
        pair_id, tr_idx = self.index[idx]
        meta = self.meta[idx]
        info = self.pairs[pair_id]

        h5f = self._get_h5()
        grp = h5f[info["h5_grp"]]
        eeg = grp["eeg"]    # (T_eeg, C)

        if self.use_precomputed_fmri:
            jitter = random.choice(self.config.fmri_aug.tr_jitter) if self.using_aug else 0
            tr_idx = int(tr_idx) + jitter
            feat_h5 = self._get_features_h5()
            feat_grp = feat_h5[info["h5_grp"]]
            tr_indices = feat_grp["tr_indices"][:]           # (M,) sorted int array
            pos = np.searchsorted(tr_indices, tr_idx)
            pos = int(np.clip(pos, 0, len(tr_indices) - 1))
            fmri_tr = feat_grp["features"][pos]              # (288,) float32
        else:
            fmri = grp["fmri"]   # (T, 96, 96, 96) or (n_roi, T)
            n_fmri_frames = fmri.shape[0] if not self.config.use_parcellation else fmri.shape[1]
            jitter = random.choice(self.config.fmri_aug.tr_jitter) if self.using_aug else 0
            tr_idx = min(max(int(tr_idx) + jitter, 0), n_fmri_frames - 1)

            fmri_start = tr_idx
            fmri_end   = int(tr_idx + np.ceil(self.config.eeg_win_sec / self.config.tr))

            if self.config.use_parcellation:
                fmri_tr = fmri[:, fmri_start:fmri_end].astype(np.float32)   # (n_roi, T_win)
            else:
                fmri_tr = fmri[fmri_start:fmri_end]                         # (T_win, X, Y, Z) float16
                fmri_tr = np.transpose(fmri_tr, (1, 2, 3, 0))               # (X, Y, Z, T_win) float16

        t_fmri = tr_idx * self.config.tr
        ts_eeg = t_fmri - np.array(self.config.hrf_shifts_sec) - self.config.eeg_win_sec

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
            "eeg":      torch.from_numpy(np.stack(eeg_windows)),  # (K, C_native, T)
            "fmri":     torch.from_numpy(fmri_tr),                # (X, Y, Z, T_win) or (n_roi, T_win)
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
    # Group by (run, tr) so the K samples sharing one slot stay contiguous
    # in the flattened batch — the contrastive loss reshape expects this.
    batch = sorted(batch, key=lambda x: (x["run"], x["tr_idx"], x["sub"]))

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

    # batch-local integer codes for the identity-controlled contrastive term:
    # sub_id groups rows by subject, slot_id groups by task-moment (run, tr).
    # Codes only need to be consistent WITHIN the batch (used for equality masks).
    subs  = [s["sub"] for s in batch]
    slots = [(s["run"], int(s["tr_idx"])) for s in batch]
    sub_code  = {v: i for i, v in enumerate(dict.fromkeys(subs))}
    slot_code = {v: i for i, v in enumerate(dict.fromkeys(slots))}

    return {
        "eeg":      torch.stack(eeg_tensors),
        "fmri":     torch.stack(fmri_tensors),
        "ch_names": common_ordered,
        "sub_id":   torch.tensor([sub_code[v]  for v in subs],  dtype=torch.long),
        "slot_id":  torch.tensor([slot_code[v] for v in slots], dtype=torch.long),
    }


class ContrastiveBatchSampler(Sampler):
    """
    Hierarchical R x T sampler keyed by (subject, activity) blocks. Each batch:
      - R distinct blocks, where a block is one (subject, activity) recording
      - T margin-separated TRs from each block (all the SAME subject)
    so total batch size = R * T = M.

    Why blocks are single-subject:
    The contrastive loss reduces its loss via whatever feature separates the
    positive from the in-batch negatives. If the hard (high-similarity) negatives
    were different subjects, "which subject is this" would solve matching via
    anatomy / electrode-placement / spectral fingerprint alone — a shortcut that
    learns no temporal alignment and stays at chance on unseen subjects.

    By drawing all T TRs of a block from ONE subject, each query gets T-1 in-block
    negatives that are same-subject / different-moment. Subject identity is then
    constant across the pool and carries no discriminative information — the model
    can only separate them via TR-level EEG-fMRI alignment, which is the signal we
    actually want. The (R-1) * T cross-block slots remain as easy negatives for
    diversity.

    A block is keyed by (sub, act_key); the same subject doing the same act_key
    across multiple sessions is merged (one TR index may map to several samples,
    one is chosen at random). Each slot is a single sample — num_subjects has no
    effect here.

    `margin_tr` applies between two TRs of the same block (so the T same-subject
    slots are temporally decorrelated). Across blocks no margin is needed — BOLD
    signals are unrelated regardless of TR index.
    """
    def __init__(self, dataset: SimultEEG_fMRI, config: TrainConfig):
        self.dataset = dataset
        self.T       = config.data.num_timestamps
        self.R       = config.data.num_recordings
        self.margin  = config.data.margin_tr
        self.M       = self.R * self.T

        # block (subject, activity) -> tr -> list of dataset indices
        block_trs: dict = defaultdict(lambda: defaultdict(list))
        for idx, meta in enumerate(self.dataset.meta):
            block_trs[(meta['sub'], meta['run'])][meta['tr']].append(idx)
        self.block_trs = block_trs
        self.block_usable_trs = {
            blk: list(trs.keys())
            for blk, trs in block_trs.items()
            if len(trs) >= self.T
        }

    def __iter__(self):
        # consumable per-block TR pool — each TR drawn at most once per epoch
        remaining = {blk: list(trs) for blk, trs in self.block_usable_trs.items()}

        while True:
            eligible = [blk for blk, trs in remaining.items() if len(trs) >= self.T]
            if len(eligible) < self.R:
                return

            random.shuffle(eligible)

            batch_indices = []
            picked = 0
            for blk in eligible:
                if picked == self.R:
                    break

                # greedy: pick T margin-separated TRs from this block's pool
                pool = list(remaining[blk])
                random.shuffle(pool)
                picked_trs = []
                for tr in pool:
                    if all(abs(tr - s) >= self.margin for s in picked_trs):
                        picked_trs.append(tr)
                        if len(picked_trs) == self.T:
                            break

                if len(picked_trs) < self.T:
                    continue  # margin-pack failed for this block this batch

                for tr in picked_trs:
                    remaining[blk].remove(tr)
                    batch_indices.append(random.choice(self.block_trs[blk][tr]))
                picked += 1

            if picked < self.R:
                return  # epoch tail — cannot fill another full batch

            yield batch_indices

    def __len__(self):
        # upper-bound estimate: total usable TRs across eligible blocks, divided
        # by M. Greedy margin-pack failures may slightly reduce actual count.
        total = sum(len(trs) for trs in self.block_usable_trs.values())
        return total // self.M