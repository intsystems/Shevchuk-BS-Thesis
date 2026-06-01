"""Diagnostic step 3b — group-level multi-band EEG<->fMRI coupling.

Stronger than 3: (a) accumulate per-subject correlation MAPS so consistent
coupling sums across subjects instead of comparing inconsistent per-subject maxima;
(b) scan 5 bands; (c) also test the robust scalar: band power vs GLOBAL-mean BOLD,
sign-consistency across subjects. All compared to a circular-shift null.
"""
import sys
import h5py
import numpy as np
from scipy.signal import butter, filtfilt
from train.config import TrainConfig
from src.utils.dataset import SimultEEG_fMRI

ACT = sys.argv[1] if len(sys.argv) > 1 else "task-dme"
MAXSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LAG = 2                      # ~4.2s HRF
OCC = ["O1","O2","Oz","POz","PO3","PO4","PO7","PO8"]
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,12),"beta":(13,30),"gamma":(30,45)}
BLOCK, SKIP, SR, TR = 4, 8, 200, 2.1
SPT = int(round(TR*SR))

cfg = TrainConfig(); ds = SimultEEG_fMRI(cfg)
pmap = {}
for p in ds.pairs:
    if p['activity']==ACT and p['sub'] not in pmap: pmap[p['sub']]=p
subs = sorted(pmap)[:MAXSUB]

def coarse(v):
    T=v.shape[0]; return v.reshape(T,24,BLOCK,24,BLOCK,24,BLOCK).mean((2,4,6)).reshape(T,-1)

f=h5py.File("dataset.h5","r")
filts={k:butter(4,[lo/(SR/2),hi/(SR/2)],btype="band") for k,(lo,hi) in BANDS.items()}
rng=np.random.default_rng(0)

# collect per-subject voxelwise r maps and global-BOLD scalar, per band, matched + null
vox_maps={k:[] for k in BANDS}; vox_null={k:[] for k in BANDS}
glob_r={k:[] for k in BANDS}; glob_null={k:[] for k in BANDS}
mask_acc=None
for s in subs:
    p=pmap[s]; chans=p['channels']
    occ=[chans.index(c) for c in OCC if c in chans]
    if len(occ)<3: continue
    eeg=f[p['h5_grp']]["eeg"][:].astype(np.float32)
    bold=coarse(f[p['h5_grp']]["fmri"][:].astype(np.float32))
    Tf=bold.shape[0]; nb=min(Tf,eeg.shape[0]//SPT)
    bold=bold[SKIP:nb]
    Bz=(bold-bold.mean(0))/(bold.std(0)+1e-6)
    gb=bold.mean(1); gz=(gb-gb.mean())/(gb.std()+1e-6)
    msk=(bold.mean(0)>np.percentile(bold.mean(0),60))
    mask_acc = msk if mask_acc is None else (mask_acc&msk)
    for k,(b,a) in filts.items():
        sig=filtfilt(b,a,eeg[:,occ],axis=0)**2
        pw=np.array([sig[i*SPT:(i+1)*SPT].mean() for i in range(nb)])
        pw=np.log(pw+1e-8)[SKIP:]
        az=(pw-pw.mean())/(pw.std()+1e-6)
        azn=np.roll(az,rng.integers(10,len(az)-10))
        n=len(az)
        x=az[:n-LAG]; xn=azn[:n-LAG]; Y=Bz[LAG:]; yg=gz[LAG:]
        vox_maps[k].append((x[:,None]*Y).mean(0))
        vox_null[k].append((xn[:,None]*Y).mean(0))
        glob_r[k].append(float((x*yg).mean()))
        glob_null[k].append(float((xn*yg).mean()))

nsub=len(glob_r["alpha"])
print(f"activity={ACT}  subjects={nsub}  lag={LAG} ({LAG*TR:.1f}s)\n")
print("=== voxelwise GROUP-MEAN |r| (consistent coupling accumulates) ===")
print(f"{'band':>7} {'matched max|gr|':>16} {'null max|gr|':>14} {'#vox gr<-.1':>12} {'#vox gr>.1':>11}")
for k in BANDS:
    gm=np.mean(vox_maps[k],0)[mask_acc]; gn=np.mean(vox_null[k],0)[mask_acc]
    print(f"{k:>7} {np.abs(gm).max():>16.3f} {np.abs(gn).max():>14.3f} {(gm<-0.1).sum():>12d} {(gm>0.1).sum():>11d}")

print("\n=== EEG band power vs GLOBAL-mean BOLD (scalar per subject) ===")
print(f"{'band':>7} {'mean r':>9} {'std':>7} {'null mean':>10} {'t vs0':>8} {'consistent sign':>16}")
for k in BANDS:
    r=np.array(glob_r[k]); rn=np.array(glob_null[k])
    t=r.mean()/(r.std(ddof=1)/np.sqrt(len(r))+1e-9)
    frac=max((r>0).mean(),(r<0).mean())
    print(f"{k:>7} {r.mean():>9.3f} {r.std():>7.3f} {rn.mean():>10.3f} {t:>8.2f} {frac*100:>14.0f}%")
print("\n|t|>~2.5 or group-mean |r| map >> null => real cross-modal coupling. Else: none recoverable.")
