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
CKPT       = sys.argv[1] if len(sys.argv) > 1 else "model.ckpt"
SUBJECT    = sys.argv[2] if len(sys.argv) > 2 else None   # e.g. "sub-03"; None -> auto
ACT_ALIGN  = sys.argv[3] if len(sys.argv) > 3 else None   # e.g. "task-dme"; None -> auto
ACT_OTHER  = sys.argv[4] if len(sys.argv) > 4 else None   # different activity; None -> auto
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ENC_BATCH  = 32
PROJ       = "pca"    # "pca" preserves temporal drift; "tsne" shows cluster separation
TSNE_PERP  = 30
DRAW_TRAJ  = True     # overlay a smoothed temporal path on the scatter
SMOOTH_WIN = 20      # moving-average window (TRs) for the smoothed trajectory spine
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


def pick_subject(ds):
    """Return (subject_id, [(pair_id, activity), ...]) for the chosen subject."""
    by_sub = defaultdict(list)
    for pid, p in enumerate(ds.pairs):
        by_sub[p["sub"]].append((pid, p["activity"]))

    if SUBJECT is not None:
        if SUBJECT not in by_sub:
            raise SystemExit(f"subject {SUBJECT!r} not found; "
                             f"available: {sorted(by_sub)}")
        return SUBJECT, by_sub[SUBJECT]

    subject = next((s for s, rec in sorted(by_sub.items())
                    if len({a for _, a in rec}) >= 2), None)
    if subject is None:
        raise SystemExit("no subject with >= 2 distinct activities found")
    return subject, by_sub[subject]


def encode_subject_recordings(model, ds, recordings):
    """Encode every (pair_id, activity) for one subject.

    Returns ze [N, D], zf [N, D] and a parallel list of activity labels.
    Windows within each activity are in temporal (TR) order.
    """
    ze_parts, zf_parts, labels = [], [], []
    for pid, activity in recordings:
        print(f"  [INFO]   {activity}  (pair_id={pid})")
        ze, zf = encode_recording(model, ds, pid)
        ze_parts.append(ze)
        zf_parts.append(zf)
        labels.extend([activity] * len(ze))
    return np.vstack(ze_parts), np.vstack(zf_parts), labels


def _project_all(combined):
    """2-D projection of combined [N, D] using PROJ method."""
    if PROJ == "tsne":
        perp = min(TSNE_PERP, (len(combined) - 1) // 3)
        print(f"[INFO] running t-SNE on {len(combined)} points (perplexity={perp}) …")
        return TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=1000).fit_transform(combined)
    print(f"[INFO] running PCA on {len(combined)} points …")
    return PCA(n_components=2, random_state=42).fit_transform(combined)


def plot_subject_tsne(ax, ze_all, zf_all, act_labels, subject):
    """2-D projection of all EEG and fMRI embeddings for one subject.

    Colour = activity, shape = modality (○ EEG, △ fMRI).
    DRAW_TRAJ=True draws a temporal path line through each activity's EEG embeddings
    in window order, making time visible as spatial direction (works best with PCA).
    """
    activities = sorted(set(act_labels))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(activities)))
    act2c = {a: palette[i] for i, a in enumerate(activities)}
    act_arr = np.array(act_labels)

    combined = np.vstack([ze_all, zf_all])
    coords = _project_all(combined)
    ze_2d = coords[:len(ze_all)]
    zf_2d = coords[len(ze_all):]

    for act in activities:
        mask = act_arr == act
        c = act2c[act]

        if DRAW_TRAJ:
            pts = ze_2d[mask]
            n = len(pts)
            if n > SMOOTH_WIN:
                w = min(SMOOTH_WIN, n)
                kernel = np.ones(w) / w
                smooth = np.stack([np.convolve(pts[:, d], kernel, mode="same")
                                   for d in range(2)], axis=1)[w // 2: n - w // 2]
                ax.plot(smooth[:, 0], smooth[:, 1], "-", color=c,
                        lw=2.0, alpha=0.75, zorder=3)
                ax.scatter(smooth[0, 0],  smooth[0, 1],  s=70, color=c,
                           marker=">", zorder=5, edgecolor="k", linewidth=0.4)
                ax.scatter(smooth[-1, 0], smooth[-1, 1], s=70, color=c,
                           marker="s", zorder=5, edgecolor="k", linewidth=0.4)

        ax.scatter(ze_2d[mask, 0], ze_2d[mask, 1], s=16, color=c,
                   alpha=0.7, marker="o", edgecolor="none", zorder=2)
        ax.scatter(zf_2d[mask, 0], zf_2d[mask, 1], s=16, color=c,
                   alpha=0.7, marker="^", edgecolor="none", zorder=2)

    act_handles = [
        plt.Line2D([0], [0], color=act2c[a], marker="o", ls="none",
                   markersize=7, label=a)
        for a in activities
    ]
    shape_handles = [
        plt.Line2D([0], [0], color="0.3", marker="o", ls="none",
                   markersize=7, label="EEG  ○"),
        plt.Line2D([0], [0], color="0.3", marker="^", ls="none",
                   markersize=7, label="fMRI △"),
    ]
    ax.legend(handles=act_handles + shape_handles, fontsize=8,
              loc="best", ncol=2, framealpha=0.8)
    method = PROJ.upper()
    ax.set_title(f"{method} — {subject} — all activities  (○ EEG, △ fMRI)", fontsize=11)
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")


def plot_traj_panel(ax, embed_list, labels, cmaps, title):
    """PCA projection of one or more temporal trajectories onto a single axis.

    embed_list : list of (n_i, D) arrays, each in temporal (TR) order
    labels     : display name for each trajectory
    cmaps      : matplotlib colormap name per trajectory (time → colour)

    All trajectories are projected jointly so their positions are comparable.
    Each trajectory gets a path line and a midpoint direction arrow (if DRAW_TRAJ).
    """
    combined = np.vstack(embed_list)
    pca = PCA(n_components=2).fit(combined)
    coords = [pca.transform(e) for e in embed_list]

    handles = []
    for pts, label, cmap in zip(coords, labels, cmaps):
        n = len(pts)
        w = min(SMOOTH_WIN, n)
        kernel = np.ones(w) / w
        smooth = np.stack([np.convolve(pts[:, d], kernel, mode="same")
                           for d in range(2)], axis=1)[w // 2: n - w // 2]

        # colour the line by time using a multicoloured LineCollection
        from matplotlib.collections import LineCollection
        segs = np.stack([smooth[:-1], smooth[1:]], axis=1)
        cmap_obj = plt.get_cmap(cmap)
        t_norm = np.linspace(0, 1, len(segs))
        lc = LineCollection(segs, cmap=cmap, linewidth=2.5, alpha=0.9, zorder=3)
        lc.set_array(t_norm)
        ax.add_collection(lc)

        c_start = cmap_obj(0.2)
        c_end   = cmap_obj(0.85)
        ax.scatter(smooth[0, 0],  smooth[0, 1],  s=100, color=c_start,
                   marker=">", zorder=5, edgecolor="k", linewidth=0.6)
        ax.scatter(smooth[-1, 0], smooth[-1, 1], s=100, color=c_end,
                   marker="s", zorder=5, edgecolor="k", linewidth=0.6)

        handles.append(plt.Line2D([0], [0], color=cmap_obj(0.6), lw=2.5,
                                  label=label))

    ax.autoscale_view()   # rescale axes to the LineCollections

    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def main():
    config = TrainConfig()
    config.train.using_aug = False

    model = load_model(config)

    print("[INFO] building full dataset")
    ds = SimultEEG_fMRI(config)

    subject, recordings = pick_subject(ds)
    acts = sorted({a for _, a in recordings})
    print(f"[INFO] subject={subject}  activities={acts}")

    act_align = ACT_ALIGN or acts[0]
    act_other = ACT_OTHER or next((a for a in acts if a != act_align), None)
    pid_align = next(pid for pid, a in recordings if a == act_align)
    pid_other = next(pid for pid, a in recordings if a == act_other)

    print(f"[INFO] encoding aligned:  {act_align}")
    ze_A, zf_A = encode_recording(model, ds, pid_align)
    print(f"[INFO] encoding other:    {act_other}")
    _,    zf_B = encode_recording(model, ds, pid_other)
    print(f"[INFO] windows: aligned={len(ze_A)}  other={len(zf_B)}")

    def cos(x, y):
        n = min(len(x), len(y))
        return (x[:n] * y[:n]).sum(1)

    cos_aligned  = cos(ze_A, zf_A)
    cos_mismatch = cos(ze_A, zf_B)

    # ── figure: 3 trajectory panels + cosine similarity ─────────────────────────
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], hspace=0.4, wspace=0.3)

    # panel 1 — single EEG trajectory: how does the EEG embedding evolve over time?
    ax0 = fig.add_subplot(gs[0, 0])
    plot_traj_panel(
        ax0,
        [ze_A], [f"EEG  {act_align}"], ["Blues"],
        f"(1) Single trajectory\n{subject} / {act_align} — EEG",
    )

    # panel 2 — aligned: EEG and fMRI from the same recording
    ax1 = fig.add_subplot(gs[0, 1])
    plot_traj_panel(
        ax1,
        [ze_A, zf_A], [f"EEG  {act_align}", f"fMRI {act_align}"],
        ["Blues", "Oranges"],
        f"(2) Aligned\n{subject} / {act_align} — EEG & fMRI",
    )

    # panel 3 — misaligned: EEG from one activity, fMRI from another
    ax2 = fig.add_subplot(gs[0, 2])
    plot_traj_panel(
        ax2,
        [ze_A, zf_B], [f"EEG  {act_align}", f"fMRI {act_other}"],
        ["Blues", "Reds"],
        f"(3) Misaligned\n{subject} — EEG:{act_align} / fMRI:{act_other}",
    )

    # bottom — per-window cosine similarity
    ax_cos = fig.add_subplot(gs[1, :])
    ax_cos.plot(cos_aligned, color="tab:green", lw=1.2,
                label=f"aligned  EEG↔fMRI {act_align}"
                      f"  (mean {cos_aligned.mean():+.3f})")
    ax_cos.plot(cos_mismatch, color="tab:red", lw=1.2,
                label=f"mismatch EEG {act_align} ↔ fMRI {act_other}"
                      f"  (mean {cos_mismatch.mean():+.3f})")
    ax_cos.axhline(cos_aligned.mean(),  color="tab:green", ls="--", lw=0.8)
    ax_cos.axhline(cos_mismatch.mean(), color="tab:red",   ls="--", lw=0.8)
    ax_cos.axhline(0.0, color="0.6", lw=0.6)
    ax_cos.set_xlabel("window index (TR order)")
    ax_cos.set_ylabel("cos(EEG[t], fMRI[t])")
    ax_cos.set_title("Per-window cosine similarity — aligned vs mismatched", fontsize=10)
    ax_cos.legend(fontsize=9)

    fig.suptitle(f"EEG / fMRI embedding trajectories — {subject}", fontsize=13)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[INFO] saved {OUT_PATH}")
    print(f"[RESULT] mean cos  aligned={cos_aligned.mean():+.4f}  "
          f"mismatch={cos_mismatch.mean():+.4f}  "
          f"gap={cos_aligned.mean() - cos_mismatch.mean():+.4f}")


if __name__ == "__main__":
    main()
