"""Diagnostic step 3 — within-subject cross-modal EEG<->fMRI coupling.

The contrastive positive pairs EEG with fMRI of the same moment. If EEG band
power carries no information about BOLD even within one subject, there is no
cross-modal signal for any loss to align.

Canonical probe: occipital ALPHA (8-12 Hz) power vs BOLD, scanned over HRF lags.
Tested against a phase-shuffle null of the alpha timecourse (kills the temporal
relationship while preserving spectra/autocorr). Pools several subjects.
"""
import sys
import h5py
import numpy as np
from scipy.signal import butter, filtfilt
from train.config import TrainConfig
from src.utils.dataset import SimultEEG_fMRI

ACT = sys.argv[1] if len(sys.argv) > 1 else "task-dme"
MAXSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BAND = (8.0, 12.0)            # alpha
OCC = ["O1", "O2", "Oz", "POz", "PO3", "PO4", "PO7", "PO8"]
BLOCK, SKIP = 4, 8
SR, TR = 200, 2.1
SAMP_PER_TR = int(round(TR * SR))   # 420

cfg = TrainConfig()
ds = SimultEEG_fMRI(cfg)
pmap = {}
for p in ds.pairs:
    if p['activity'] == ACT and p['sub'] not in pmap:
        pmap[p['sub']] = p
subs = sorted(pmap)[:MAXSUB]

b, a = butter(4, [BAND[0]/(SR/2), BAND[1]/(SR/2)], btype="band")

def coarse(v):
    T = v.shape[0]
    return v.reshape(T, 24, BLOCK, 24, BLOCK, 24, BLOCK).mean((2, 4, 6)).reshape(T, -1)

f = h5py.File("dataset.h5", "r")
LAGS = list(range(0, 5))
# accumulate best-voxel |r| across subjects at each lag, matched and null
matched_best, null_best = {L: [] for L in LAGS}, {L: [] for L in LAGS}
matched_meanabs = {L: [] for L in LAGS}

rng = np.random.default_rng(0)
for s in subs:
    p = pmap[s]
    chans = p['channels']
    occ_idx = [chans.index(c) for c in OCC if c in chans]
    if len(occ_idx) < 3:
        continue
    eeg = f[p['h5_grp']]["eeg"][:].astype(np.float32)      # (T_eeg, 60)
    bold = coarse(f[p['h5_grp']]["fmri"][:].astype(np.float32))  # (T,V)
    Tfmri = bold.shape[0]

    # occipital alpha power per TR bin
    occ = eeg[:, occ_idx]                                  # (T_eeg, n_occ)
    occ = filtfilt(b, a, occ, axis=0)
    pw = occ**2
    nbins = min(Tfmri, eeg.shape[0] // SAMP_PER_TR)
    alpha = np.array([pw[i*SAMP_PER_TR:(i+1)*SAMP_PER_TR].mean() for i in range(nbins)])
    alpha = np.log(alpha + 1e-8)
    bold = bold[:nbins]
    n = nbins

    az = (alpha - alpha.mean()) / (alpha.std() + 1e-6)
    Bz = (bold - bold.mean(0)) / (bold.std(0) + 1e-6)
    az_null = np.roll(az, rng.integers(10, n - 10))

    for L in LAGS:
        x = az[:n-L] if L else az
        xn = az_null[:n-L] if L else az_null
        Y = Bz[L:] if L else Bz
        r = (x[:, None] * Y).mean(0)                       # corr per voxel
        rn = (xn[:, None] * Y).mean(0)
        matched_best[L].append(np.abs(r).max())
        null_best[L].append(np.abs(rn).max())
        matched_meanabs[L].append(np.abs(r).mean())

print(f"activity={ACT}  subjects used={len(matched_best[0])}  (occipital alpha vs BOLD)\n")
print(f"{'lag':>10} {'matched max|r|':>16} {'null max|r|':>14} {'matched mean|r|':>16}")
for L in LAGS:
    mb = np.mean(matched_best[L]); nb = np.mean(null_best[L]); mm = np.mean(matched_meanabs[L])
    star = "  <-- HRF" if abs(L*TR-4.2) < 1.1 else ""
    print(f"{L:>3d} ({L*TR:+.1f}s) {mb:>16.3f} {nb:>14.3f} {mm:>16.3f}{star}")
print("\nmatched max|r| >> null max|r| at HRF lag => real within-subject cross-modal coupling.")
