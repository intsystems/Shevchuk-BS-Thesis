"""
Mann-Whitney U test and Wilcoxon signed-rank test for EEG-fMRI embedding alignment.

Independence problem:
  Within a block, positive (diagonal) and negative (off-diagonal) cosine similarities
  are computed from the SAME similarity matrix → they are NOT independent. A
  straightforward Mann-Whitney U on all pairs from all blocks would violate the
  independence assumption.

Solution — block-level split:
  1. Mann-Whitney U (independent groups):
       Randomly assign recording blocks to set A and set B.
       Group P = {mean diagonal similarity of each block in A}  → "positive" scores
       Group N = {mean off-diagonal similarity of each block in B} → "negative" scores
       Since A ∩ B = ∅ and blocks are independent recordings, P and N are independent.
       H0: P and N come from the same distribution.
       H1 (one-sided): P is stochastically greater than N.

  2. Wilcoxon signed-rank (paired, more powerful):
       For EVERY block b compute (pos_b, neg_b) = (mean diag, mean off-diag).
       Test the differences d_b = pos_b − neg_b against the null median = 0.
       Blocks are independent → the signed-rank test is valid.
       This uses ALL blocks for both groups, so it is more powerful.

Both tests report the effect size (rank-biserial correlation r).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from train.config import TrainConfig
from train.train import build_model, split_dataset


# ── config ────────────────────────────────────────────────────────────────────
CKPT      = "model.ckpt"
BATCH     = 64
N_WORKERS = 0
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED   = 0    # seed for A/B block split in Mann-Whitney
EXCL_LAG     = 10   # exclusion zone: ignore |i-j| <= EXCL_LAG in off-diagonal
                    # rationale: BOLD autocorrelation radius ≈ HRF_peak/TR ≈ 6/2.1 ≈ 3 TRs
                    # + EEG window overlap ≈ 15/2.1 ≈ 7 TRs → conservative k=10
# ──────────────────────────────────────────────────────────────────────────────


def _block_collate(batch):
    common = set(batch[0]["ch_names"])
    for s in batch[1:]:
        common &= set(s["ch_names"])
    if not common:
        raise RuntimeError("empty channel intersection")
    first_order = batch[0]["ch_names"]
    common_ordered = [c for c in first_order if c in common]

    eeg_list, fmri_list = [], []
    for s in batch:
        idx = torch.tensor([s["ch_names"].index(c) for c in common_ordered],
                           dtype=torch.long)
        eeg_list.append(s["eeg"].index_select(dim=-2, index=idx))
        fmri_list.append(s["fmri"])

    return {
        "eeg":      torch.stack(eeg_list),
        "fmri":     torch.stack(fmri_list),
        "ch_names": common_ordered,
        "sub_idx":  torch.tensor([s["sub_idx"] for s in batch], dtype=torch.long),
        "sub":      [s["sub"] for s in batch],
        "run":      [s["run"] for s in batch],
    }


def encode_val(model, val_ds, device, batch_size):
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=N_WORKERS, collate_fn=_block_collate,
                        drop_last=False)
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
    groups = defaultdict(list)
    for i, (s, r) in enumerate(zip(subs, runs)):
        groups[(s, r)].append(i)
    blocks = []
    for key, idxs in sorted(groups.items()):
        idx_arr = np.array(idxs)
        blocks.append({
            "key": key,
            "f":   z_f[idx_arr],
            "e":   z_e[idx_arr],
        })
    return blocks


def block_statistics(blocks, excl_lag=0):
    """For each block compute (mean_pos, mean_neg, n, subject).

    excl_lag > 0: exclude off-diagonal pairs with |i-j| <= excl_lag to avoid
    inflating mu_neg with temporally autocorrelated neighbours.
    """
    pos_means, neg_means, ns, subjects = [], [], [], []
    for b in blocks:
        n = len(b["f"])
        if n < 2:
            continue
        S = b["f"] @ b["e"].T      # (n, n)
        diag = S[np.arange(n), np.arange(n)]

        if excl_lag > 0:
            i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            far_mask = np.abs(i_idx - j_idx) > excl_lag
            np.fill_diagonal(far_mask, False)   # exclude diagonal itself
            if far_mask.sum() == 0:
                continue                         # block too small for this lag
            off_diag = S[far_mask]
        else:
            off_diag = S[~np.eye(n, dtype=bool)]

        pos_means.append(float(diag.mean()))
        neg_means.append(float(off_diag.mean()))
        ns.append(n)
        subjects.append(b["key"][0])            # "sub-XX"

    return (np.array(pos_means), np.array(neg_means),
            np.array(ns), np.array(subjects))


def rank_biserial(U, n1, n2):
    """Rank-biserial correlation from Mann-Whitney U."""
    return 1.0 - 2 * U / (n1 * n2)


def main():
    config = TrainConfig()
    config.train.using_aug = False

    print("[INFO] building model …")
    model = build_model(config)

    print(f"[INFO] loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(DEVICE).eval()

    print(f"[INFO] building val dataset …")
    _, val_ds, _ = split_dataset(config)
    print(f"[INFO] val_ds size = {len(val_ds)}")

    # ── load or compute embeddings ────────────────────────────────────────────
    cache_path = CKPT.replace(".ckpt", "_val_emb.npz")
    if os.path.exists(cache_path):
        print(f"[INFO] loading cached embeddings from {cache_path} …")
        cache = np.load(cache_path, allow_pickle=False)
        z_e, z_f = cache["z_e"], cache["z_f"]
        subs, runs = cache["subs"].tolist(), cache["runs"].tolist()
    else:
        print("[INFO] encoding val set …")
        z_e, z_f, subs, runs = encode_val(model, val_ds, DEVICE, BATCH)
        np.savez(cache_path, z_e=z_e, z_f=z_f,
                 subs=np.array(subs), runs=np.array(runs))
        print(f"[INFO] embeddings cached to {cache_path}")
    print(f"[INFO] {len(z_e)} samples  dim={z_e.shape[1]}")

    blocks = build_blocks(z_e, z_f, subs, runs)
    print(f"[INFO] {len(blocks)} recording blocks")

    plot_data = []   # collect (diffs, label, p, prop) for each test
    K_PERM = 10_000  # permutation iterations

    def run_permutation_test(pos, neg, label):
        """Paired permutation test — uses magnitudes, no distributional assumptions.

        Under H0 the sign of each d_b is exchangeable (±1 equally likely).
        We randomly flip signs of d_b values and compare the mean to t_obs.
        This is strictly more powerful than the sign test while remaining exact.
        """
        d      = pos - neg
        t_obs  = d.mean()
        rng_p  = np.random.default_rng(SPLIT_SEED)
        count  = 0
        null   = np.empty(K_PERM)
        for i in range(K_PERM):
            signs   = rng_p.choice([-1.0, 1.0], size=len(d))
            t_perm  = (d * signs).mean()
            null[i] = t_perm
            if t_perm >= t_obs:
                count += 1
        p    = (count + 1) / (K_PERM + 1)
        prop = (d > 0).mean()
        z    = (t_obs - null.mean()) / null.std()

        print(f"\n{'='*62}")
        print(f"  {label}  [PERMUTATION]")
        print(f"  n              = {len(d)}")
        print(f"  t_obs (mean d) = {t_obs:+.6f}")
        print(f"  null mean±std  = {null.mean():+.6f} ± {null.std():.6f}")
        print(f"  z-score        = {z:.2f}σ")
        print(f"  p (K={K_PERM})  = {p:.2e}")
        print(f"  prop pos>neg   = {prop:.3f}")
        print(f"  {'SIGNIFICANT p < 0.05' if p < 0.05 else 'NOT significant'}")
        plot_data.append((d, label + " [perm]", p, prop))
        return p, prop

    def run_wilcoxon_transformed(pos, neg, label):
        """Wilcoxon signed-rank on signed-log-transformed differences.

        T(d) = sign(d) * log(1 + |d|) is monotone → preserves H0 (median = 0)
        and compresses the right tail, improving symmetry so Wilcoxon's
        distributional assumption is better satisfied.
        Skewness before/after is reported to verify the improvement.
        """
        from scipy.stats import skew
        d   = pos - neg
        d_t = np.sign(d) * np.log1p(np.abs(d))

        sk_orig  = float(skew(d))
        sk_trans = float(skew(d_t))

        w, p = stats.wilcoxon(d_t, alternative="greater")
        B_   = len(d_t)
        r    = w / (B_ * (B_ + 1) / 2)
        size = "large" if r > 0.5 else "medium" if r > 0.3 else "small"

        print(f"\n{'='*62}")
        print(f"  {label}  [Wilcoxon + signed-log]")
        print(f"  n                  = {B_}")
        print(f"  skewness raw / T   = {sk_orig:+.3f} → {sk_trans:+.3f}")
        print(f"  W                  = {w:.1f}   p = {p:.2e}")
        print(f"  effect size r      = {r:.4f}  ({size})")
        print(f"  median T(d)        = {np.median(d_t):+.6f}")
        print(f"  {'SIGNIFICANT p < 0.05' if p < 0.05 else 'NOT significant'}")
        plot_data.append((d_t, label + " [T]", p, r))
        return p, r

    def run_sign_test(pos, neg, label):
        """Sign test — no distributional assumptions.

        H0: P(pos_b > neg_b) = 0.5   (alignment no better than chance)
        H1: P(pos_b > neg_b) > 0.5   (one-sided)

        Test statistic k = number of units where pos > neg.
        Under H0: k ~ Binomial(n_pairs, 0.5).
        Effect size: proportion p_hat = k / n_pairs.
        """
        d     = pos - neg
        n     = len(d)
        k     = int((d > 0).sum())        # ties (d==0) excluded from n
        n_eff = int((d != 0).sum())
        k_eff = int((d > 0).sum())
        result = stats.binomtest(k_eff, n_eff, p=0.5, alternative="greater")
        p     = result.pvalue
        prop  = k_eff / n_eff if n_eff > 0 else 0.0
        ci    = result.proportion_ci(confidence_level=0.95)

        print(f"\n{'='*62}")
        print(f"  {label}")
        print(f"  n              = {n}  (non-tied: {n_eff})")
        print(f"  k (pos > neg)  = {k_eff} / {n_eff}  ({prop:.3f})")
        print(f"  95% CI prop    = [{ci.low:.3f}, {ci.high:.3f}]")
        print(f"  p (binomial)   = {p:.2e}")
        print(f"  median diff    = {np.median(d):+.6f}")
        print(f"  {'SIGNIFICANT p < 0.05' if p < 0.05 else 'NOT significant'}")
        plot_data.append((d, label, p, prop))
        return p, prop

    # ── block distribution across subjects ───────────────────────────────────
    pos_means, neg_means, _, block_subs = block_statistics(blocks, excl_lag=0)
    B = len(pos_means)
    unique_subs, counts = np.unique(block_subs, return_counts=True)
    S = len(unique_subs)
    print(f"\n[INFO] {B} blocks across {S} subjects "
          f"(mean {B/S:.1f} blocks/subject, "
          f"min {counts.min()} max {counts.max()})")
    for sub, c in zip(unique_subs, counts):
        print(f"       {sub}: {c} blocks")

    # ════════════════════════════════════════════════════════════════════════
    # TEST 1a — Wilcoxon, block level (all blocks, potential pseudo-replication)
    # ════════════════════════════════════════════════════════════════════════
    run_sign_test(pos_means, neg_means, "TEST 1a — sign  (block level)")
    run_permutation_test(pos_means, neg_means, "TEST 1a — perm  (block level)")
    run_wilcoxon_transformed(pos_means, neg_means, "TEST 1a — wilcox (block level)")

    sub_pos = np.array([pos_means[block_subs == s].mean() for s in unique_subs])
    sub_neg = np.array([neg_means[block_subs == s].mean() for s in unique_subs])
    run_sign_test(sub_pos, sub_neg,
                  f"TEST 1b — sign  (subject level, n={S})")
    run_permutation_test(sub_pos, sub_neg,
                         f"TEST 1b — perm  (subject level, n={S})")
    run_wilcoxon_transformed(sub_pos, sub_neg,
                              f"TEST 1b — wilcox (subject level, n={S})")

    pos_excl, neg_excl, _, subs_excl = block_statistics(blocks, excl_lag=EXCL_LAG)
    run_sign_test(pos_excl, neg_excl,
                  f"TEST 1c — sign  (block + excl_lag={EXCL_LAG})")
    run_permutation_test(pos_excl, neg_excl,
                         f"TEST 1c — perm  (block + excl_lag={EXCL_LAG})")
    run_wilcoxon_transformed(pos_excl, neg_excl,
                              f"TEST 1c — wilcox (block + excl_lag={EXCL_LAG})")

    unique_subs_excl = np.unique(subs_excl)
    sub_pos_excl = np.array([pos_excl[subs_excl == s].mean() for s in unique_subs_excl])
    sub_neg_excl = np.array([neg_excl[subs_excl == s].mean() for s in unique_subs_excl])
    run_sign_test(sub_pos_excl, sub_neg_excl,
                  f"TEST 1d — sign  (subject + excl_lag={EXCL_LAG}, most conservative)")
    run_permutation_test(sub_pos_excl, sub_neg_excl,
                         f"TEST 1d — perm  (subject + excl_lag={EXCL_LAG}, most conservative)")
    run_wilcoxon_transformed(sub_pos_excl, sub_neg_excl,
                              f"TEST 1d — wilcox (subject + excl_lag={EXCL_LAG}, most conservative)")

    # ════════════════════════════════════════════════════════════════════════
    # TEST 2 — Mann-Whitney U  (independent groups via block split)
    # ════════════════════════════════════════════════════════════════════════
    rng   = np.random.default_rng(SPLIT_SEED)
    perm  = rng.permutation(B)
    half  = B // 2
    A_idx, B_idx = perm[:half], perm[half:]
    group_P = pos_means[A_idx]
    group_N = neg_means[B_idx]
    u_stat, mw_p = stats.mannwhitneyu(group_P, group_N, alternative="greater")
    r_mw = rank_biserial(u_stat, len(group_P), len(group_N))
    print(f"\n{'='*62}")
    print("  TEST 2 — Mann-Whitney U  (independent block split)")
    print(f"  A={len(A_idx)} pos blocks  |  B={len(B_idx)} neg blocks")
    print(f"  U = {u_stat:.0f}   p = {mw_p:.2e}   r = {r_mw:.4f}")
    print(f"  {'SIGNIFICANT p < 0.05' if mw_p < 0.05 else 'NOT significant'}")
    print(f"{'='*62}\n")

    # ── plot distributions (permutation variants only) ────────────────────────
    perm_data = [(d, lbl, p, prop) for d, lbl, p, prop in plot_data
                 if "[perm]" in lbl]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flat

    for ax, (diffs, label, p, prop) in zip(axes, perm_data):
        n     = len(diffs)
        n_eff = int((diffs != 0).sum())
        k     = int((diffs > 0).sum())
        med   = np.median(diffs)
        n_bins = min(40, max(8, n // 6))

        ax.hist(diffs, bins=n_bins, color="steelblue", alpha=0.55,
                edgecolor="white", linewidth=0.4)

        # light green shading for x > 0
        ax.axvspan(0, max(diffs) * 1.1, alpha=0.07, color="#4caf50", zorder=0)

        # individual coloured points as rug for small n
        if n <= 30:
            colors = ["#ef5350" if d < 0 else "#4caf50" for d in diffs]
            ax.scatter(diffs, np.full(n, -0.4), s=55, color=colors,
                       edgecolor="k", linewidth=0.5, zorder=4, clip_on=False)

        ax.axvline(0,   color="0.35", lw=1.2, ls="--", label="null (0)")
        ax.axvline(med, color="#1565c0", lw=1.6, ls="-",
                   label=f"median {med:+.4f}")

        short = label.split("(", 1)[1].rstrip(")") if "(" in label else label
        ax.set_title(f"{label.split('—')[0].strip()}\n({short})", fontsize=9)
        ax.set_xlabel("$d_b = \\mu^+_b - \\mu^-_b$", fontsize=9)
        ax.set_ylabel("count", fontsize=9)

        info = f"k={k}/{n_eff}  p={p:.1e}  prop={prop:.3f}"
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Distribution of $d_b = \\mu^+_b - \\mu^-_b$ — sign test variants\n"
                 "(blue line = median, shaded = positive half)",
                 fontsize=11)
    plt.tight_layout()
    out = CKPT.replace(".ckpt", "_wilcoxon_dists.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[INFO] plot saved to {out}")


if __name__ == "__main__":
    main()
