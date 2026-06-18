"""
Within-recording block permutation test (Ojala & Garriga 2010 style).

For each (subject, recording) block in the val set, compute the n×n cosine-similarity
matrix between normalized fMRI and EEG embeddings. The diagonal entries are matched
positive pairs; off-diagonal are negatives.

Test statistic:
  t = mean(all positive sims across blocks) - mean(all negative sims across blocks)

Null distribution: independently shuffle the EEG window order WITHIN each block,
preserving the block's autocorrelation structure while breaking the EEG↔fMRI pairing.

Efficiency: similarity matrices are precomputed once. Each permutation only needs
  S[arange(n), random_perm].sum()   (O(n) per block, O(N_total) per permutation)
because S.sum() is invariant under column permutation.

p-value = (count(t_perm ≥ t_obs) + 1) / (K + 1)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from train.config import TrainConfig
from train.train import build_model, split_dataset


# ── config ────────────────────────────────────────────────────────────────────
CKPT = "checkpoints/epoch=epoch=09-mrr=val/mrr_e2f=0.0303.ckpt"
K    = 5000
BATCH   = 64
N_WORKERS = 0    # 0 avoids multiprocessing issues with h5py file handles
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
SEED    = 42
# ──────────────────────────────────────────────────────────────────────────────


def _block_collate(batch):
    """Minimal collate: common channels, stacked tensors, preserves sub/run strings."""
    common = set(batch[0]["ch_names"])
    for s in batch[1:]:
        common &= set(s["ch_names"])
    if not common:
        raise RuntimeError("empty channel intersection in batch")

    first_order = batch[0]["ch_names"]
    common_ordered = [c for c in first_order if c in common]

    eeg_list, fmri_list = [], []
    for s in batch:
        col_idx = torch.tensor(
            [s["ch_names"].index(c) for c in common_ordered], dtype=torch.long
        )
        eeg_list.append(s["eeg"].index_select(dim=-2, index=col_idx))
        fmri_list.append(s["fmri"])

    return {
        "eeg":     torch.stack(eeg_list),
        "fmri":    torch.stack(fmri_list),
        "ch_names": common_ordered,
        "sub_idx": torch.tensor([s["sub_idx"] for s in batch], dtype=torch.long),
        "sub":     [s["sub"] for s in batch],
        "run":     [s["run"] for s in batch],
    }


def encode_val(model, val_ds, device, batch_size):
    """Encode all val samples; return normalized numpy arrays plus per-sample sub/run."""
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=N_WORKERS,
        collate_fn=_block_collate,
        drop_last=False,
    )

    model.eval()
    ze_parts, zf_parts, subs, runs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            eeg     = batch["eeg"].to(device)
            fmri    = batch["fmri"].to(device).float()
            ch      = batch["ch_names"]
            sub_idx = batch["sub_idx"].to(device)

            if eeg.dim() == 4 and eeg.size(1) == 1:
                eeg = eeg.squeeze(1)

            ze = model._encode_eeg(eeg, ch_names=ch, sub_idx=sub_idx)
            zf = model.fmri_encoder(fmri, sub_offset=model._fmri_offset(sub_idx))

            ze_parts.append(F.normalize(ze, dim=-1).cpu().numpy().astype(np.float32))
            zf_parts.append(F.normalize(zf, dim=-1).cpu().numpy().astype(np.float32))
            subs.extend(batch["sub"])
            runs.extend(batch["run"])

    return np.concatenate(ze_parts), np.concatenate(zf_parts), subs, runs


def build_blocks(z_e, z_f, subs, runs):
    """Group samples by (subject, recording) and return list of block dicts."""
    groups = defaultdict(list)
    for i, (s, r) in enumerate(zip(subs, runs)):
        groups[(s, r)].append(i)

    blocks = []
    for key, idxs in sorted(groups.items()):
        idx_arr = np.array(idxs)
        blocks.append({
            "key": key,
            "f":   z_f[idx_arr],   # (n, d) normalized fMRI embeddings
            "e":   z_e[idx_arr],   # (n, d) normalized EEG embeddings
        })
    return blocks


def precompute_sim_matrices(blocks):
    """
    For each block compute:
      S        — (n, n) cosine-similarity matrix (f @ e.T)
      row_idx  — np.arange(n), reused per permutation
      total    — S.sum(), invariant under column permutation

    Returns parallel lists: sim_mats, row_indices, totals, ns.
    """
    sim_mats, row_indices, totals, ns = [], [], [], []
    for b in blocks:
        n = len(b["f"])
        if n < 2:
            sim_mats.append(None)
            row_indices.append(None)
            totals.append(0.0)
            ns.append(n)
            continue
        S = b["f"] @ b["e"].T          # (n, n)
        sim_mats.append(S)
        row_indices.append(np.arange(n))
        totals.append(float(S.sum()))
        ns.append(n)
    return sim_mats, row_indices, totals, ns


def observed_stat(sim_mats, row_indices, totals, ns):
    """t_obs: diagonal is positive, off-diagonal is negative."""
    pos_sum = neg_sum = 0.0
    pos_count = neg_count = 0
    for S, ridx, tot, n in zip(sim_mats, row_indices, totals, ns):
        if S is None:
            continue
        diag = S[ridx, ridx].sum()
        pos_sum   += diag
        neg_sum   += tot - diag
        pos_count += n
        neg_count += n * (n - 1)
    return pos_sum / pos_count - neg_sum / neg_count, pos_count, neg_count


def permuted_stat(sim_mats, row_indices, totals, ns, rng, total_all, pos_count, neg_count):
    """One permutation: shuffle EEG columns within each block → O(N_total)."""
    perm_pos = 0.0
    for S, ridx, n in zip(sim_mats, row_indices, ns):
        if S is None:
            continue
        col_perm = rng.permutation(n)
        perm_pos += S[ridx, col_perm].sum()
    perm_neg = total_all - perm_pos
    return perm_pos / pos_count - perm_neg / neg_count


def main():
    config = TrainConfig()
    config.train.using_aug = False

    print("[INFO] building model …")
    model = build_model(config)

    print(f"[INFO] loading checkpoint: {CKPT}")
    ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    if missing:
        print(f"[WARN] missing keys: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")
    model = model.to(DEVICE).eval()

    print(f"[INFO] building val dataset (split_mode={config.data.split_mode}) …")
    _, val_ds, _ = split_dataset(config)
    print(f"[INFO] val_ds size = {len(val_ds)}")

    print("[INFO] encoding val set …")
    z_e, z_f, subs, runs = encode_val(model, val_ds, DEVICE, BATCH)
    print(f"[INFO] encoded {len(z_e)} samples into {z_e.shape[1]}-d embeddings")

    blocks = build_blocks(z_e, z_f, subs, runs)
    block_sizes = [len(b["f"]) for b in blocks]
    print(
        f"[INFO] {len(blocks)} recording blocks | "
        f"sizes: min={min(block_sizes)} max={max(block_sizes)} "
        f"mean={np.mean(block_sizes):.0f}"
    )

    print("[INFO] precomputing similarity matrices …")
    sim_mats, row_indices, totals, ns = precompute_sim_matrices(blocks)
    total_all = sum(totals)

    t_obs, pos_count, neg_count = observed_stat(sim_mats, row_indices, totals, ns)

    print(f"\n  t_obs = {t_obs:.6f}  "
          f"(pos pairs: {pos_count}, neg pairs: {neg_count})")

    # ── permutation loop ──────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    count_extreme = 0

    print(f"\n[INFO] running {K} within-block permutations …")
    for k in range(K):
        t_perm = permuted_stat(
            sim_mats, row_indices, totals, ns, rng,
            total_all, pos_count, neg_count
        )
        if t_perm >= t_obs:
            count_extreme += 1

        if (k + 1) % 500 == 0:
            running_p = (count_extreme + 1) / (k + 2)
            print(f"  [{k + 1:>5}/{K}]  running p ≤ {running_p:.4f}")

    p_val = (count_extreme + 1) / (K + 1)

    print(f"\n{'='*60}")
    print(f"  t_obs                  = {t_obs:.6f}")
    print(f"  K permutations         = {K}")
    print(f"  count(t_perm ≥ t_obs)  = {count_extreme}")
    print(f"  p-value                = {p_val:.4f}")
    if p_val < 0.001:
        verdict = "p < 0.001 — temporal alignment within recordings carries highly significant signal"
    elif p_val < 0.01:
        verdict = f"p = {p_val:.4f} — significant at α=0.01"
    elif p_val < 0.05:
        verdict = f"p = {p_val:.4f} — significant at α=0.05"
    else:
        verdict = f"p = {p_val:.4f} — NOT significant (no detectable temporal alignment)"
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
