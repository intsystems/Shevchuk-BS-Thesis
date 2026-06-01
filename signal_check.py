"""Diagnostic: is there a signal to train on? Step 1 — data inventory.

Reuses the real dataset indexing (reads only shapes, not volumes) to answer:
  - how many subjects / recordings / activities exist
  - the distribution of distinct subjects per (activity, tr) moment
  - how many moments / activities actually qualify for K-subject multi-positives
"""
import sys
from collections import defaultdict, Counter

import numpy as np

from train.config import TrainConfig
from src.utils.dataset import SimultEEG_fMRI

cfg = TrainConfig()
ds = SimultEEG_fMRI(cfg)  # all subjects

print(f"samples (index rows): {len(ds.index)}")
print(f"recordings (pairs):   {len(ds.pairs)}")

subs = sorted({p['sub'] for p in ds.pairs})
acts = sorted({p['activity'] for p in ds.pairs})
print(f"subjects ({len(subs)}): {subs}")
print(f"activities ({len(acts)}): {acts}")

# recordings per subject, per activity
rec_per_act = Counter(p['activity'] for p in ds.pairs)
print("\nrecordings per activity:")
for a, n in rec_per_act.most_common():
    nsub = len({p['sub'] for p in ds.pairs if p['activity'] == a})
    print(f"  {a:30s} recs={n:3d}  distinct_subs={nsub}")

# activity -> tr -> set(subjects)   (same key the sampler uses)
act = defaultdict(lambda: defaultdict(set))
for m in ds.meta:
    act[m['activity']][int(m['tr'])].add(m['sub'])

# distribution: subjects-per-moment over ALL moments
subs_per_moment = []
for a, trs in act.items():
    for tr, s in trs.items():
        subs_per_moment.append(len(s))
spm = np.array(subs_per_moment)
print(f"\ntotal moments (activity,tr): {len(spm)}")
print("subjects-per-moment distribution:")
for k in [1, 2, 3, 4, 6, 8, 10, 12, 16]:
    print(f"  moments with >= {k:2d} subjects: {(spm >= k).sum():6d}  ({100*(spm>=k).mean():5.1f}%)")
print(f"  max subjects in any moment: {spm.max()}   mean: {spm.mean():.2f}")

# how many activities can offer >= T moments at each K (sampler's usability test)
T = cfg.data.num_timestamps
print(f"\nusable activities (>= T={T} margin-ignored moments with >= K subjects):")
for K in [2, 4, 8]:
    n_act = 0
    n_moments_total = 0
    for a, trs in act.items():
        good = [tr for tr, s in trs.items() if len(s) >= K]
        n_moments_total += len(good)
        if len(good) >= T:
            n_act += 1
    print(f"  K={K:2d}: activities usable={n_act:3d}/{len(acts)}   total qualifying moments={n_moments_total}")
