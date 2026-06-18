"""
Plot EEG and fMRI embedding trajectories from a trained checkpoint.

A "trajectory" is the sequence of embeddings produced by sliding the model over
ALL windows of a single recording, in temporal (TR) order. Each window t yields
an EEG embedding ze[t] and an fMRI embedding zf[t] (both L2-normalized, 128-d).

Two regimes are compared:

  ALIGNED      EEG and fMRI come from the SAME recording (same subject, same
               activity) -> window t is a true matched pair. A well-trained model
               should keep ze[t] and zf[t] close, so the two trajectories track
               each other in embedding space and per-window cos sim is high.

  NOT ALIGNED  EEG from recording A (subject S, activity X) vs fMRI from a
               DIFFERENT activity Y of the same subject. Window t pairs unrelated
               moments -> trajectories should be unrelated and cos sim near zero.

Outputs eeg_fmri_trajectories.png:
  top    : 2D PCA of each pair's two trajectories (time = colour, connectors show
           the window-by-window EEG<->fMRI distance)
  bottom : per-window cosine similarity, aligned vs not-aligned

Usage:
  python plot_trajectories.py [CKPT] [SUBJECT] [ACT_ALIGNED] [ACT_OTHER]
All args optional; sensible defaults are auto-picked from the dataset.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train.config import TrainConfig
from train.train import build_model
from src.utils.dataset import SimultEEG_fMRI, collate_fn


# ── config / CLI ────────────────────────────────────────────────────────────────
CKPT       = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/last.ckpt"
SUBJECT    = sys.argv[2] if len(sys.argv) > 2 else None   # e.g. "sub-03"; None -> auto
ACT_ALIGN  = sys.argv[3] if len(sys.argv) > 3 else None   # e.g. "task-dme"; None -> auto
ACT_OTHER  = sys.argv[4] if len(sys.argv) > 4 else None   # different activity; None -> auto
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ENC_BATCH  = 32
PROJ       = "tsne"   # "tsne" | "pca"
TSNE_PERP  = 30
OUT_PATH   = "eeg_fmri_trajectories.png"
# ────────────────────────────────────────────────────────────────────────────────


def load_model(config):
    """Build the model and load the checkpoint state dict (mirrors eval_perm_null.py)."""
    model = build_model(config)
    print(f"[INFO] loading checkpoint {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(DEVICE).eval()


def recording_rows(ds, pair_id):
    """Dataset row indices for one recording (pair_id), sorted by TR (temporal order)."""
    rows = np.where(ds.index[:, 0] == pair_id)[0]
    return rows[np.argsort(ds.index[rows, 1])].tolist()


@torch.no_grad()
def encode_recording(model, ds, pair_id):
    """Return (ze, zf) trajectories [n_windows, D] over all windows of one recording.

    collate_fn sorts a chunk by moment_id = (activity, tr); for a single recording
    that is exactly TR order, so concatenating chunks preserves the temporal sequence.
    """
    rows = recording_rows(ds, pair_id)
    ze_all, zf_all = [], []
    for i in range(0, len(rows), ENC_BATCH):
        batch = collate_fn([ds[r] for r in rows[i:i + ENC_BATCH]])
        eeg  = batch["eeg"].to(DEVICE)
        fmri = batch["fmri"].to(DEVICE).float()
        ch   = batch.get("ch_names")
        sub_idx = batch.get("sub_idx")
        if sub_idx is not None:
            sub_idx = sub_idx.to(DEVICE)
        if eeg.dim() == 4 and eeg.size(1) == 1:       # squeeze degenerate HRF-shift axis
            eeg = eeg.squeeze(1)
        ze = model._encode_eeg(eeg, ch_names=ch, sub_idx=sub_idx)
        zf = model.fmri_encoder(fmri, sub_offset=model._fmri_offset(sub_idx))
        ze_all.append(ze.cpu())
        zf_all.append(zf.cpu())
    return torch.cat(ze_all).numpy(), torch.cat(zf_all).numpy()


def pick_recordings(ds):
    """Auto-select (subject, pair_id_aligned, pair_id_other) honouring any CLI overrides."""
    by_sub = defaultdict(list)                         # sub -> [(pair_id, activity), ...]
    for pid, p in enumerate(ds.pairs):
        by_sub[p["sub"]].append((pid, p["activity"]))

    # subject: CLI override, else first subject that has >= 2 distinct activities
    if SUBJECT is not None:
        subject = SUBJECT
    else:
        subject = next((s for s, rec in by_sub.items()
                        if len({a for _, a in rec}) >= 2), None)
    if subject is None or subject not in by_sub:
        raise SystemExit(f"no usable subject (asked for {SUBJECT!r}); "
                         f"available: {sorted(by_sub)}")
    recs = by_sub[subject]

    def find(activity):
        return next((pid for pid, a in recs if a == activity), None)

    acts = sorted({a for _, a in recs})
    act_align = ACT_ALIGN or acts[0]
    act_other = ACT_OTHER or next((a for a in acts if a != act_align), None)
    pid_align = find(act_align)
    pid_other = find(act_other)
    if pid_align is None or pid_other is None or pid_align == pid_other:
        raise SystemExit(f"subject {subject}: need two different activities; "
                         f"has {acts}")
    print(f"[INFO] subject={subject}  aligned={act_other!r}->{act_align!r}")
    print(f"[INFO]   aligned  recording = {ds.pairs[pid_align]['h5_grp']}")
    print(f"[INFO]   other    recording = {ds.pairs[pid_other]['h5_grp']}")
    return subject, act_align, act_other, pid_align, pid_other


def _project(A, B):
    """Return 2-D projections of A and B using the method set by PROJ."""
    combined = np.vstack([A, B])
    if PROJ == "tsne":
        perp = min(TSNE_PERP, (len(combined) - 1) // 3)
        coords = TSNE(n_components=2, perplexity=perp, random_state=42,
                      n_iter=1000).fit_transform(combined)
    else:
        pca = PCA(n_components=2).fit(combined)
        coords = pca.transform(combined)
    return coords[:len(A)], coords[len(A):]


def plot_pca_traj(ax, A, B, labels, title):
    """2-D projection of two trajectories A, B [n, D]; colour = window index, draw paths +
    window-by-window connectors between the matched prefix."""
    n = min(len(A), len(B))
    a2, b2 = _project(A, B)

    ta, tb = np.arange(len(A)), np.arange(len(B))
    ax.plot(a2[:, 0], a2[:, 1], "-", color="0.7", lw=0.6, zorder=1)
    ax.plot(b2[:, 0], b2[:, 1], "-", color="0.7", lw=0.6, zorder=1)
    # window-by-window connectors over the matched prefix
    for t in range(n):
        ax.plot([a2[t, 0], b2[t, 0]], [a2[t, 1], b2[t, 1]],
                color="0.85", lw=0.4, zorder=0)
    sa = ax.scatter(a2[:, 0], a2[:, 1], c=ta, cmap="Blues", s=18,
                    edgecolor="k", linewidth=0.2, zorder=2, label=labels[0])
    ax.scatter(b2[:, 0], b2[:, 1], c=tb, cmap="Oranges", s=18, marker="^",
               edgecolor="k", linewidth=0.2, zorder=2, label=labels[1])
    ax.set_title(title, fontsize=10)
    lbl = "t-SNE" if PROJ == "tsne" else "PCA"
    ax.set_xlabel(f"{lbl}-1"); ax.set_ylabel(f"{lbl}-2")
    ax.legend(fontsize=8, loc="best")
    return sa


def main():
    config = TrainConfig()
    config.train.using_aug = False        # deterministic windows: no aug, no TR jitter

    model = load_model(config)

    print("[INFO] building full dataset (all subjects, all recordings)")
    ds = SimultEEG_fMRI(config)

    subject, act_align, act_other, pid_align, pid_other = pick_recordings(ds)

    print("[INFO] encoding aligned recording (EEG + fMRI from same recording)")
    ze_A, zf_A = encode_recording(model, ds, pid_align)
    print("[INFO] encoding other-activity recording (for the fMRI mismatch)")
    _,    zf_B = encode_recording(model, ds, pid_other)
    print(f"[INFO] windows: aligned={len(ze_A)}  other={len(zf_B)}")

    # per-window cosine similarity (embeddings are already L2-normalized by the projector)
    def cos(x, y):
        n = min(len(x), len(y))
        return (x[:n] * y[:n]).sum(1)
    cos_aligned = cos(ze_A, zf_A)
    cos_mismatch = cos(ze_A, zf_B)

    # ── figure ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    plot_pca_traj(
        ax0, ze_A, zf_A,
        labels=[f"EEG  {act_align}", f"fMRI {act_align}"],
        title=f"ALIGNED — {subject} / {act_align}\n(EEG & fMRI same recording)",
    )
    ax1 = fig.add_subplot(gs[0, 1])
    plot_pca_traj(
        ax1, ze_A, zf_B,
        labels=[f"EEG  {act_align}", f"fMRI {act_other}"],
        title=f"NOT ALIGNED — {subject}\n(EEG {act_align} vs fMRI {act_other})",
    )

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(cos_aligned, color="tab:green", lw=1.4,
             label=f"aligned (mean {cos_aligned.mean():+.3f})")
    ax2.plot(cos_mismatch, color="tab:red", lw=1.4,
             label=f"not aligned (mean {cos_mismatch.mean():+.3f})")
    ax2.axhline(cos_aligned.mean(), color="tab:green", ls="--", lw=0.8)
    ax2.axhline(cos_mismatch.mean(), color="tab:red", ls="--", lw=0.8)
    ax2.axhline(0.0, color="0.6", lw=0.6)
    ax2.set_xlabel("window index (TR order)")
    ax2.set_ylabel("cos(EEG[t], fMRI[t])")
    ax2.set_title("Per-window EEG↔fMRI cosine similarity", fontsize=10)
    ax2.legend(fontsize=9)

    fig.suptitle("EEG / fMRI embedding trajectories — aligned vs not aligned", fontsize=13)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[INFO] saved {OUT_PATH}")
    print(f"[RESULT] mean cos  aligned={cos_aligned.mean():+.4f}  "
          f"not-aligned={cos_mismatch.mean():+.4f}  "
          f"gap={cos_aligned.mean() - cos_mismatch.mean():+.4f}")


if __name__ == "__main__":
    main()
