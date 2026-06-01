"""Diagnostic step 2b — is the cross-subject fMRI ISC REAL and ALIGNED?

(1) Matched ISC vs a circular-time-shift NULL: if matched >> null, the shared
    signal is genuine stimulus-locking, not a global/motion artifact.
(2) Temporal lag scan: shift the leave-one-out mean by L TRs. Peak at L=0 means
    subjects are stimulus-aligned on the (activity, tr) grid the sampler assumes;
    a peak off 0 means a systematic onset/dummy-scan misalignment that would by
    itself sink cross-subject training.
"""
import sys
from collections import defaultdict
import h5py
import numpy as np
from train.config import TrainConfig
from src.utils.dataset import SimultEEG_fMRI

ACT = sys.argv[1] if len(sys.argv) > 1 else "task-dme"
MAXSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BLOCK, SKIP = 4, 8

cfg = TrainConfig()
ds = SimultEEG_fMRI(cfg)
by_sub = {}
for p in ds.pairs:
    if p['activity'] == ACT and p['sub'] not in by_sub:
        by_sub[p['sub']] = p['h5_grp']
subs = sorted(by_sub)[:MAXSUB]

def coarse(v):
    T = v.shape[0]
    return v.reshape(T, 24, BLOCK, 24, BLOCK, 24, BLOCK).mean(axis=(2, 4, 6))

f = h5py.File("dataset.h5", "r")
series, lengths = [], []
for s in subs:
    c = coarse(f[by_sub[s]]["fmri"][:].astype(np.float32))
    series.append(c); lengths.append(c.shape[0])
L = min(lengths)
X = np.stack([c[SKIP:L] for c in series], 0).reshape(len(subs), L - SKIP, -1)
S, T, V = X.shape
print(f"activity={ACT}  S={S}  T={T}  Vcoarse={V}")

mean_sig = X.mean((0, 1)); tvar = X.std(1).mean(0)
mask = (mean_sig > np.percentile(mean_sig, 60)) & (tvar > 1e-3)
Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)

def isc_at_lag(lag):
    vals = np.zeros(V)
    for i in range(S):
        others = np.delete(Xz, i, 0).mean(0)
        a = Xz[i]
        if lag > 0:   a, others = a[lag:], others[:-lag]
        elif lag < 0: a, others = a[:lag], others[-lag:]
        num = (a * others).sum(0)
        den = np.sqrt((a**2).sum(0) * (others**2).sum(0)) + 1e-9
        vals += num / den
    return vals / S

matched = isc_at_lag(0)
# circular-shift null: each subject shifted by a distinct large offset
rng = np.random.default_rng(0)
Xz_null = np.stack([np.roll(Xz[i], rng.integers(15, T - 15), axis=0) for i in range(S)], 0)
nullvals = np.zeros(V)
for i in range(S):
    others = np.delete(Xz_null, i, 0).mean(0); a = Xz_null[i]
    nullvals += (a * others).sum(0) / (np.sqrt((a**2).sum(0)*(others**2).sum(0))+1e-9)
nullvals /= S

m, nl = matched[mask], nullvals[mask]
print("\n=== matched vs phase-shuffle null (in-brain) ===")
print(f"  matched  mean={m.mean():+.3f}  median={np.median(m):+.3f}  max={m.max():+.3f}  %>0.2={100*(m>0.2).mean():.1f}%")
print(f"  null     mean={nl.mean():+.3f}  median={np.median(nl):+.3f}  max={nl.max():+.3f}  %>0.2={100*(nl>0.2).mean():.1f}%")
print(f"  signal-to-null ratio (mean): {m.mean()/ (abs(nl.mean())+1e-6):.1f}x")

print("\n=== temporal lag scan (mean in-brain ISC) ===")
for lag in range(-3, 4):
    v = isc_at_lag(lag)[mask].mean()
    bar = "#" * int(max(0, v) * 300)
    print(f"  lag {lag:+d} TR ({lag*cfg.data.tr:+.1f}s): {v:+.3f} {bar}")
