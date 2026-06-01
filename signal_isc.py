"""Diagnostic step 2 — cross-subject fMRI ISC at matched moments.

If subjects watching the same movie share a stimulus-locked BOLD signal, their
fMRI timecourses at matched TRs should correlate (inter-subject correlation).
ISC ~ 0 everywhere => no shared signal => cross-subject multi-positives are noise
and cross-subject retrieval is hopeless regardless of the loss.

Leave-one-out: each subject's voxel timecourse vs the mean of the other subjects,
averaged over subjects. Coarse 4x4x4 block-averaged grid to denoise + cut memory.
"""
import sys
from collections import defaultdict

import h5py
import numpy as np

from train.config import TrainConfig
from src.utils.dataset import SimultEEG_fMRI

ACT = sys.argv[1] if len(sys.argv) > 1 else "task-tp"
MAXSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BLOCK = 4         # spatial block-average factor: 96 -> 24
SKIP = 8          # drop first TRs (HRF ramp / steady-state)

cfg = TrainConfig()
ds = SimultEEG_fMRI(cfg)

# one recording per subject for this activity
by_sub = {}
for p in ds.pairs:
    if p['activity'] == ACT and p['sub'] not in by_sub:
        by_sub[p['sub']] = p['h5_grp']
subs = sorted(by_sub)[:MAXSUB]
print(f"activity={ACT}  subjects={len(subs)}: {subs}")

def coarse(vol4d):  # (T, 96,96,96) -> (T, 24,24,24)
    T = vol4d.shape[0]
    v = vol4d.reshape(T, 24, BLOCK, 24, BLOCK, 24, BLOCK)
    return v.mean(axis=(2, 4, 6))

f = h5py.File("dataset.h5", "r")
series = []
lengths = []
for s in subs:
    arr = f[by_sub[s]]["fmri"][:].astype(np.float32)   # (T,96,96,96)
    c = coarse(arr)                                      # (T,24,24,24)
    series.append(c)
    lengths.append(c.shape[0])
    print(f"  {s}: {by_sub[s]}  T={c.shape[0]}")

L = min(lengths)
X = np.stack([c[SKIP:L] for c in series], axis=0)   # (S, T, 24,24,24)
S, T = X.shape[0], X.shape[1]
X = X.reshape(S, T, -1)                              # (S, T, V)
print(f"\ncommon T={T}, voxels(coarse)={X.shape[2]}, subjects={S}")

# in-brain mask: voxels with decent mean signal & temporal variance
mean_sig = X.mean(axis=(0, 1))
tvar = X.std(axis=1).mean(axis=0)
mask = (mean_sig > np.percentile(mean_sig, 60)) & (tvar > 1e-3)
print(f"masked voxels: {mask.sum()} / {mask.size}")

# z-score each voxel timecourse per subject
Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)

# leave-one-out ISC per voxel
isc = np.zeros(X.shape[2])
for i in range(S):
    others = np.delete(Xz, i, axis=0).mean(axis=0)      # (T,V)
    a = Xz[i]                                            # (T,V)
    num = (a * others).sum(axis=0)
    den = np.sqrt((a**2).sum(axis=0) * (others**2).sum(axis=0)) + 1e-9
    isc += num / den
isc /= S

m = isc[mask]
print("\n=== cross-subject fMRI ISC (leave-one-out, coarse grid) ===")
print(f"  mean ISC (in-brain):   {m.mean():+.3f}")
print(f"  median ISC:            {np.median(m):+.3f}")
print(f"  max ISC:               {m.max():+.3f}")
print(f"  %% voxels ISC > 0.10:   {100*(m>0.10).mean():.1f}%")
print(f"  %% voxels ISC > 0.20:   {100*(m>0.20).mean():.1f}%")
print(f"  %% voxels ISC > 0.30:   {100*(m>0.30).mean():.1f}%")
print("\nInterpretation: ISC>0.2 in a chunk of voxels => real shared stimulus signal.")
print("Near 0 everywhere => no cross-subject signal to align (or volumes misaligned).")
