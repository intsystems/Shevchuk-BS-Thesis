"""
CCA and CCM analysis of EEG-fMRI embedding trajectories.

For each recording block (subject × activity) with n >= MIN_BLOCK:
  CCA  — canonical correlations between Γ_s^e and Γ_s^f (n_b × 128 matrices).
         Implemented via SVD of the whitened cross-covariance; more stable
         than sklearn CCA when n_b << d.
  CCM  — convergent cross mapping on the first pair of canonical variates
         (x_t, y_t) reduced to 1-D.  Prediction skill ρ(L) is computed
         for increasing library sizes L; convergence indicates shared dynamics.

Results are averaged first within subject, then across subjects.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict

from train.config import TrainConfig
from train.train import build_model, split_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── config ────────────────────────────────────────────────────────────────────
CKPT       = "model.ckpt"
BATCH      = 64
N_WORKERS  = 0
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MIN_BLOCK  = 20      # minimum block length to include in analysis
CCA_COMPS  = 5       # number of canonical correlations to report
CCM_E      = 2       # delay-embedding dimension
CCM_TAU    = 1       # delay in TR steps
CCM_NN     = CCM_E + 1   # nearest neighbours (standard for CCM)
OUT_PATH   = "cca_ccm_results.png"
# ──────────────────────────────────────────────────────────────────────────────


# ── reuse collate / encode / build_blocks from eval_mannwhitney ──────────────
def _block_collate(batch):
    common = set(batch[0]["ch_names"])
    for s in batch[1:]:
        common &= set(s["ch_names"])
    if not common:
        raise RuntimeError("empty channel intersection")
    first_order   = batch[0]["ch_names"]
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


def encode_val(model, val_ds):
    loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=N_WORKERS, collate_fn=_block_collate,
                        drop_last=False)
    model.eval()
    ze_parts, zf_parts, subs, runs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            eeg     = batch["eeg"].to(DEVICE)
            fmri    = batch["fmri"].to(DEVICE).float()
            ch      = batch["ch_names"]
            sub_idx = batch["sub_idx"].to(DEVICE)
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
        idx = np.array(idxs)
        blocks.append({"key": key, "e": z_e[idx], "f": z_f[idx]})
    return blocks


# ── CCA ───────────────────────────────────────────────────────────────────────
def svd_cca(X, Y, n_components):
    """
    Canonical correlations via SVD of the whitened cross-covariance.

    X, Y : (n, d) — already on the sphere (L2-normalised rows).
    Returns (corrs, A, B) where corrs are canonical correlations and
    A, B are the canonical weight matrices (d × k).
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    # Economy SVD for whitening
    Ux, Sx, Vxt = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, Vyt = np.linalg.svd(Y, full_matrices=False)

    tol  = 1e-8
    rx   = (Sx > tol).sum()
    ry   = (Sy > tol).sum()
    Ux, Sx = Ux[:, :rx], Sx[:rx]
    Uy, Sy = Uy[:, :ry], Sy[:ry]

    # Cross-covariance in whitened space
    C = Ux.T @ Uy                        # (rx, ry)
    Uc, S, Vct = np.linalg.svd(C, full_matrices=False)

    k      = min(n_components, len(S), rx, ry)
    corrs  = np.clip(S[:k], 0, 1)

    # Canonical weights in original space
    A = (Vxt[:rx].T / Sx[:, None]).T @ Uc[:, :k]    # (d, k)  EEG weights
    B = (Vyt[:ry].T / Sy[:, None]).T @ Vct.T[:, :k]  # (d, k)  fMRI weights

    return corrs, A, B


# ── CCM ───────────────────────────────────────────────────────────────────────
def delay_embed(x, E, tau):
    """
    Build delay-embedding shadow manifold from scalar time series x (length T).
    Returns matrix of shape (T - (E-1)*tau, E).
    Columns ordered newest-first: [x_t, x_{t-τ}, ..., x_{t-(E-1)τ}].
    """
    T   = len(x)
    N   = T - (E - 1) * tau
    M   = np.empty((N, E), dtype=np.float64)
    for i in range(E):
        start = (E - 1 - i) * tau
        end   = T - i * tau if i > 0 else T
        M[:, i] = x[start:end]
    return M


def ccm_skill(x, y, E, tau, L):
    """
    Cross-map skill ρ(L): use the shadow manifold of x (library size L)
    to predict y, return Pearson r of predictions vs actuals.

    x, y : 1-D arrays of the same length T.
    """
    T  = len(x)
    Mx = delay_embed(x, E, tau)   # (T-(E-1)τ, E)
    My = delay_embed(y, E, tau)   # same shape — we predict the y-variate
    Tn = len(Mx)
    L  = min(L, Tn)

    lib_M = Mx[:L]

    preds, actuals = [], []
    for t in range(L, Tn):
        dists  = np.sum((lib_M - Mx[t]) ** 2, axis=1)
        nn_idx = np.argsort(dists)[:CCM_NN]

        d0 = dists[nn_idx[0]]
        if d0 < 1e-12:
            w = np.zeros(CCM_NN); w[0] = 1.0
        else:
            w = np.exp(-dists[nn_idx] / d0)
        w /= w.sum()

        preds.append(float(w @ My[nn_idx, 0]))   # predict y-variate at t
        actuals.append(float(My[t, 0]))

    if len(preds) < 5:
        return np.nan
    r, _ = pearsonr(actuals, preds)
    return r


def ccm_curve(x, y, E, tau, L_steps):
    """Return ρ(L) for each library size in L_steps."""
    return np.array([ccm_skill(x, y, E, tau, L) for L in L_steps])


# ── per-block analysis ────────────────────────────────────────────────────────
def analyse_block(block, L_steps):
    """
    Run CCA + CCM on one recording block.
    Returns dict with 'corrs' and 'ccm_e2f', 'ccm_f2e' curves (or None).
    """
    Ze, Zf = block["e"], block["f"]    # (n, 128)
    n      = len(Ze)

    if n < MIN_BLOCK:
        return None

    k     = min(CCA_COMPS, n - 2)
    corrs, A, B = svd_cca(Ze, Zf, k)

    # First canonical variates as 1-D time series
    x = Ze @ A[:, 0]   # EEG variate
    y = Zf @ B[:, 0]   # fMRI variate

    # Normalise to zero mean, unit std for CCM stability
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    L_valid = [L for L in L_steps if L <= n - 1]
    if len(L_valid) < 2:
        ccm_e2f = ccm_f2e = None
    else:
        ccm_e2f = ccm_curve(x, y, CCM_E, CCM_TAU, L_valid)
        ccm_f2e = ccm_curve(y, x, CCM_E, CCM_TAU, L_valid)

    return {"corrs": corrs, "ccm_e2f": ccm_e2f, "ccm_f2e": ccm_f2e,
            "n": n, "L_valid": L_valid}


# ── aggregation ───────────────────────────────────────────────────────────────
def aggregate_by_subject(blocks, results, L_steps):
    """
    Average CCA correlations and CCM curves first within subject,
    then across subjects.
    """
    sub_data = defaultdict(lambda: {"corrs": [], "e2f": [], "f2e": []})

    for block, res in zip(blocks, results):
        if res is None:
            continue
        sub = block["key"][0]
        sub_data[sub]["corrs"].append(res["corrs"])
        if res["ccm_e2f"] is not None:
            # interpolate to common L grid
            lv = np.array(res["L_valid"])
            e2f = np.interp(L_steps, lv, res["ccm_e2f"],
                            left=np.nan, right=np.nan)
            f2e = np.interp(L_steps, lv, res["ccm_f2e"],
                            left=np.nan, right=np.nan)
            sub_data[sub]["e2f"].append(e2f)
            sub_data[sub]["f2e"].append(f2e)

    sub_corrs, sub_e2f, sub_f2e = [], [], []
    for data in sub_data.values():
        if data["corrs"]:
            # pad shorter corrs arrays
            max_k = max(len(c) for c in data["corrs"])
            padded = [np.pad(c, (0, max_k - len(c)), constant_values=np.nan)
                      for c in data["corrs"]]
            sub_corrs.append(np.nanmean(padded, axis=0))
        if data["e2f"]:
            sub_e2f.append(np.nanmean(data["e2f"], axis=0))
            sub_f2e.append(np.nanmean(data["f2e"], axis=0))

    mean_corrs = np.nanmean(sub_corrs, axis=0) if sub_corrs else None
    std_corrs  = np.nanstd(sub_corrs,  axis=0) if sub_corrs else None
    mean_e2f   = np.nanmean(sub_e2f,   axis=0) if sub_e2f  else None
    sem_e2f    = np.nanstd(sub_e2f,    axis=0) / np.sqrt(len(sub_e2f)) if sub_e2f else None
    mean_f2e   = np.nanmean(sub_f2e,   axis=0) if sub_f2e  else None
    sem_f2e    = np.nanstd(sub_f2e,    axis=0) / np.sqrt(len(sub_f2e)) if sub_f2e else None

    return (mean_corrs, std_corrs, mean_e2f, sem_e2f, mean_f2e, sem_f2e,
            len(sub_data))


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_results(mean_corrs, std_corrs, mean_e2f, sem_e2f,
                 mean_f2e, sem_f2e, L_steps, n_subs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── CCA ───────────────────────────────────────────────────────────────
    k = len(mean_corrs)
    x = np.arange(1, k + 1)
    ax1.bar(x, mean_corrs, yerr=std_corrs, color="steelblue", alpha=0.75,
            capsize=4, edgecolor="white")
    ax1.set_xlabel("Canonical variate index")
    ax1.set_ylabel("Canonical correlation $\\rho_k$")
    ax1.set_title(f"CCA — mean canonical correlations\n"
                  f"(averaged over {n_subs} subjects, blocks $n\\geq{MIN_BLOCK}$)",
                  fontsize=10)
    ax1.set_xticks(x)
    ax1.set_ylim(0, 1)
    ax1.axhline(0, color="0.5", lw=0.7)

    # ── CCM ───────────────────────────────────────────────────────────────
    if mean_e2f is not None:
        valid = ~np.isnan(mean_e2f) & ~np.isnan(mean_f2e)
        Lv = np.array(L_steps)[valid]
        ax2.plot(Lv, mean_e2f[valid], color="tab:blue",  lw=1.8,
                 label="EEG $\\to$ fMRI")
        ax2.fill_between(Lv,
                         mean_e2f[valid] - sem_e2f[valid],
                         mean_e2f[valid] + sem_e2f[valid],
                         alpha=0.2, color="tab:blue")
        ax2.plot(Lv, mean_f2e[valid], color="tab:orange", lw=1.8,
                 label="fMRI $\\to$ EEG")
        ax2.fill_between(Lv,
                         mean_f2e[valid] - sem_f2e[valid],
                         mean_f2e[valid] + sem_f2e[valid],
                         alpha=0.2, color="tab:orange")
        ax2.axhline(0, color="0.5", lw=0.7, ls="--")
        ax2.set_xlabel("Library size $L$")
        ax2.set_ylabel("Cross-map skill $\\rho(L)$")
        ax2.set_title(f"CCM — convergence of cross-map skill\n"
                      f"($E={CCM_E}$, $\\tau={CCM_TAU}$\,TR, ±1 SEM over subjects)",
                      fontsize=10)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No blocks long enough for CCM",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle("CCA and CCM on EEG–fMRI embedding trajectories "
                 "$\\Gamma_s^e$, $\\Gamma_s^f$", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[INFO] plot saved → {OUT_PATH}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    config = TrainConfig()
    config.train.using_aug = False

    print("[INFO] building model …")
    model = build_model(config)
    ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(DEVICE).eval()

    _, val_ds, _ = split_dataset(config)
    print(f"[INFO] val_ds size = {len(val_ds)}")

    cache = CKPT.replace(".ckpt", "_val_emb.npz")
    if os.path.exists(cache):
        print(f"[INFO] loading cached embeddings from {cache} …")
        c    = np.load(cache, allow_pickle=False)
        z_e, z_f = c["z_e"], c["z_f"]
        subs, runs = c["subs"].tolist(), c["runs"].tolist()
    else:
        print("[INFO] encoding val set …")
        z_e, z_f, subs, runs = encode_val(model, val_ds)
        np.savez(cache, z_e=z_e, z_f=z_f,
                 subs=np.array(subs), runs=np.array(runs))

    blocks = build_blocks(z_e, z_f, subs, runs)
    usable = [b for b in blocks if len(b["e"]) >= MIN_BLOCK]
    print(f"[INFO] {len(blocks)} blocks total, {len(usable)} with n ≥ {MIN_BLOCK}")

    # common L grid: up to max block length
    max_n   = max(len(b["e"]) for b in usable) if usable else 0
    L_steps = list(range(CCM_E + 2, max_n, max(1, max_n // 20)))

    print("[INFO] running CCA + CCM per block …")
    results = [analyse_block(b, L_steps) for b in blocks]

    (mean_corrs, std_corrs,
     mean_e2f, sem_e2f,
     mean_f2e, sem_f2e, n_subs) = aggregate_by_subject(blocks, results, L_steps)

    if mean_corrs is not None:
        print("\n  CCA canonical correlations (mean ± std across subjects):")
        for i, (m, s) in enumerate(zip(mean_corrs, std_corrs)):
            print(f"    ρ_{i+1} = {m:.4f} ± {s:.4f}")

    if mean_e2f is not None:
        valid = ~np.isnan(mean_e2f)
        print(f"\n  CCM  EEG→fMRI  ρ(L_max) = {mean_e2f[valid][-1]:.4f}")
        print(f"  CCM fMRI→EEG  ρ(L_max) = {mean_f2e[valid][-1]:.4f}")

    plot_results(mean_corrs, std_corrs, mean_e2f, sem_e2f,
                 mean_f2e, sem_f2e, L_steps, n_subs)


if __name__ == "__main__":
    main()
