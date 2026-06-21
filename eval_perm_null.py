"""
Permutation null test — runs three evaluations on the val set using the best checkpoint:

  1. real      — standard diagonal-positive MRR (same index = positive pair)
  2. perm_null — 10 random permutations of the EEG↔fMRI pairing, averaged
  3. shift_null — circular shifts of the positive labels by N//4, N//2, 3N//4

Interpretation:
  real >> perm_null ≈ random  → signal is genuine
  real ≈ perm_null  >> random → artifact (temporal position / slow dynamics)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from train.config import TrainConfig
from train.train import build_model, split_dataset
from src.utils.dataset import collate_fn


# ── config ────────────────────────────────────────────────────────────────────
CKPT = "checkpoints/epoch=epoch=09-mrr=val/mrr_e2f=0.0303.ckpt"
N_PERM = 20          # number of random permutations to average
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 64
N_WORKERS = 4
# ──────────────────────────────────────────────────────────────────────────────


def compute_mrr(z_en, z_fn, labels):
    """
    z_en, z_fn: [N, D] normalized
    labels: [N] long — z_en[i] is a positive for z_fn[labels[i]]
    Returns (mrr_e2f, mrr_f2e).
    """
    N = z_en.shape[0]

    # e2f: EEG query → find matching fMRI
    sim = z_en @ z_fn.T           # [N, N]
    true_sim = sim[torch.arange(N), labels].unsqueeze(1)   # [N, 1]
    rank = (sim >= true_sim).sum(dim=1).float()             # [N]
    mrr_e2f = (1.0 / rank).mean().item()

    # f2e: fMRI query → find matching EEG
    sim_T = z_fn @ z_en.T
    true_sim_T = sim_T[torch.arange(N), labels].unsqueeze(1)
    rank_T = (sim_T >= true_sim_T).sum(dim=1).float()
    mrr_f2e = (1.0 / rank_T).mean().item()

    # recall@k
    r1  = ((rank   <= 1).float().mean().item(), (rank_T <= 1).float().mean().item())
    r10 = ((rank   <= 10).float().mean().item(), (rank_T <= 10).float().mean().item())

    return mrr_e2f, mrr_f2e, r1, r10


def encode_val(model, val_loader, device):
    model.eval()
    all_ze, all_zf, all_mid = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            eeg    = batch["eeg"].to(device)
            fmri   = batch["fmri"].to(device).float()
            ch     = batch.get("ch_names")
            sub_idx = batch.get("sub_idx")
            if sub_idx is not None:
                sub_idx = sub_idx.to(device)

            if eeg.dim() == 4 and eeg.size(1) == 1:
                eeg = eeg.squeeze(1)

            ze = model._encode_eeg(eeg, ch_names=ch, sub_idx=sub_idx)
            zf = model.fmri_encoder(fmri, sub_offset=model._fmri_offset(sub_idx))

            all_ze.append(ze.cpu())
            all_zf.append(zf.cpu())
            all_mid.append(batch["moment_id"])

    z_e = torch.cat(all_ze, dim=0).float()
    z_f = torch.cat(all_zf, dim=0).float()
    moment_ids = torch.cat(all_mid, dim=0)
    return z_e, z_f, moment_ids


def main():
    config = TrainConfig()
    config.train.using_aug = False      # no augmentation at eval time

    print(f"[INFO] building model …")
    model = build_model(config)

    print(f"[INFO] loading checkpoint {CKPT} …")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[WARN] missing keys: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")

    model = model.to(DEVICE)
    model.eval()

    print(f"[INFO] building val dataset (split_mode={config.data.split_mode}) …")
    _, val_ds, _ = split_dataset(config)
    print(f"[INFO] val_ds size = {len(val_ds)}")

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=N_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
        multiprocessing_context="spawn" if N_WORKERS > 0 else None,
    )

    print(f"[INFO] encoding val set …")
    z_e, z_f, moment_ids = encode_val(model, val_loader, DEVICE)
    N = z_e.shape[0]
    z_en = F.normalize(z_e, dim=-1).to(DEVICE)
    z_fn = F.normalize(z_f, dim=-1).to(DEVICE)
    print(f"[INFO] encoded {N} val samples")

    random_baseline = 1.0 / N   # E[MRR] for uniform random ranking

    # ── 1. real MRR ───────────────────────────────────────────────────────────
    labels_real = torch.arange(N, device=DEVICE)
    mrr_e2f, mrr_f2e, r1, r10 = compute_mrr(z_en, z_fn, labels_real)
    print(f"\n{'='*60}")
    print(f"  REAL MRR:  e2f={mrr_e2f:.4f}  f2e={mrr_f2e:.4f}  "
          f"({mrr_e2f / random_baseline:.1f}× random)")
    print(f"  REAL R@1:  e2f={r1[0]:.4f}  f2e={r1[1]:.4f}")
    print(f"  REAL R@10: e2f={r10[0]:.4f}  f2e={r10[1]:.4f}")
    print(f"  random baseline MRR = {random_baseline:.4f}")

    # ── 2. random permutation null ────────────────────────────────────────────
    rng = np.random.default_rng(0)
    perm_mrr_e2f, perm_mrr_f2e = [], []
    for t in range(N_PERM):
        perm = torch.from_numpy(rng.permutation(N)).to(DEVICE)
        m_e2f, m_f2e, _, _ = compute_mrr(z_en, z_fn, perm)
        perm_mrr_e2f.append(m_e2f)
        perm_mrr_f2e.append(m_f2e)

    pm_e2f = float(np.mean(perm_mrr_e2f))
    pm_f2e = float(np.mean(perm_mrr_f2e))
    ps_e2f = float(np.std(perm_mrr_e2f))
    print(f"\n  PERM NULL MRR (n={N_PERM}): "
          f"e2f={pm_e2f:.4f}±{ps_e2f:.4f}  f2e={pm_f2e:.4f}  "
          f"({pm_e2f / random_baseline:.1f}× random)")

    # ── 3. circular time-shift null ───────────────────────────────────────────
    # Sort val samples by moment_id — this groups them by (activity, tr) so nearby
    # indices are temporally adjacent. Shifting the label assignment by a large
    # offset preserves autocorrelation structure while breaking content pairing.
    sort_order = torch.argsort(moment_ids)
    inv_order  = torch.argsort(sort_order)   # unsort: sorted_pos[i] in original space

    print(f"\n  SHIFT NULL MRR (circular shift of positive labels by k positions "
          f"in moment_id-sorted order):")
    for frac_num, frac_den in [(1, 4), (1, 2), (3, 4)]:
        shift = (N * frac_num) // frac_den
        # labels_shift[i] = the sample that is 'shift' positions away in sorted order
        shifted_sorted = (torch.arange(N) + shift) % N    # in sorted space
        # map back to original (unsorted) indices
        labels_shift = sort_order[shifted_sorted].to(DEVICE)
        m_e2f, m_f2e, _, _ = compute_mrr(z_en, z_fn, labels_shift)
        ratio = m_e2f / random_baseline
        print(f"    shift={shift:4d} ({frac_num}/{frac_den} of N):  "
              f"e2f={m_e2f:.4f}  f2e={m_f2e:.4f}  ({ratio:.1f}× random)")

    # ── verdict ───────────────────────────────────────────────────────────────
    real_vs_perm = mrr_e2f / pm_e2f if pm_e2f > 0 else float("inf")
    print(f"\n{'='*60}")
    print(f"  real / perm_null ratio: {real_vs_perm:.2f}×")
    if real_vs_perm > 3.0:
        print("  VERDICT: signal appears GENUINE (real >> permutation null)")
    elif real_vs_perm > 1.5:
        print("  VERDICT: MIXED — real above null but not clearly separated")
    else:
        print("  VERDICT: ARTIFACT — real ≈ permutation null")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
