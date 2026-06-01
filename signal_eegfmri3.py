"""Diagnostic step 3c — cross-modal coupling with PROPER BOLD filtering.

Data is 'nofilt' => BOLD has strong low-freq drift that hides weak coupling.
Standard fix: band-pass both the BOLD and the EEG-power regressor to 0.01-0.15 Hz,
then group voxelwise one-sample t across subjects at the HRF lag. Compare the
matched group-t map to a circular-shift null. This is the definitive within-
subject cross-modal test.
"""
import sys
import h5py
import numpy as np
from scipy.signal import butter, filtfilt
from train.config import TrainConfig
from src.utils.dataset import SimultEEG_fMRI

ACT = sys.argv[1] if len(sys.argv) > 1 else "task-dme"
MAXSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 16
LAG = 2
OCC = ["O1","O2","Oz","POz","PO3","PO4","PO7","PO8"]
BANDS = {"theta":(4,8),"alpha":(8,12),"beta":(13,30)}
BLOCK,SKIP,SR,TR = 4,8,200,2.1
SPT=int(round(TR*SR)); FS=1.0/TR
bp_lo,bp_hi = 0.01, 0.15
bb,ba = butter(2,[bp_lo/(FS/2),bp_hi/(FS/2)],btype="band")

cfg=TrainConfig(); ds=SimultEEG_fMRI(cfg)
pmap={}
for p in ds.pairs:
    if p['activity']==ACT and p['sub'] not in pmap: pmap[p['sub']]=p
subs=sorted(pmap)[:MAXSUB]

def coarse(v):
    T=v.shape[0]; return v.reshape(T,24,BLOCK,24,BLOCK,24,BLOCK).mean((2,4,6)).reshape(T,-1)

f=h5py.File("dataset.h5","r")
efilt={k:butter(4,[lo/(SR/2),hi/(SR/2)],btype="band") for k,(lo,hi) in BANDS.items()}
rng=np.random.default_rng(0)
rmaps={k:[] for k in BANDS}; rnull={k:[] for k in BANDS}; sig_acc=None; nseen=0
for s in subs:
    p=pmap[s]; chans=p['channels']; occ=[chans.index(c) for c in OCC if c in chans]
    if len(occ)<3: continue
    eeg=f[p['h5_grp']]["eeg"][:].astype(np.float32)
    bold=coarse(f[p['h5_grp']]["fmri"][:].astype(np.float32))
    Tf=bold.shape[0]; nb=min(Tf,eeg.shape[0]//SPT); bold=bold[:nb]
    ms=bold.mean(0); sig_acc=ms if sig_acc is None else sig_acc+ms; nseen+=1
    Bf=filtfilt(bb,ba,bold,axis=0)[SKIP:]
    Bz=(Bf-Bf.mean(0))/(Bf.std(0)+1e-6)
    for k,(b,a) in efilt.items():
        pw=filtfilt(b,a,eeg[:,occ],axis=0)**2
        pw=np.array([pw[i*SPT:(i+1)*SPT].mean() for i in range(nb)])
        pw=np.log(pw+1e-8)
        pf=filtfilt(bb,ba,pw)[SKIP:]
        az=(pf-pf.mean())/(pf.std()+1e-6); azn=np.roll(az,rng.integers(10,len(az)-10))
        n=len(az); x=az[:n-LAG]; xn=azn[:n-LAG]; Y=Bz[LAG:]
        rmaps[k].append((x[:,None]*Y).mean(0)); rnull[k].append((xn[:,None]*Y).mean(0))

mask_acc = (sig_acc/nseen) > np.percentile(sig_acc/nseen, 55)
nsub=len(rmaps["alpha"]); print(f"activity={ACT} subjects={nsub} maskvox={mask_acc.sum()} bandpass {bp_lo}-{bp_hi}Hz lag={LAG*TR:.1f}s\n")
print(f"{'band':>7} {'max|groupT|':>12} {'nullmaxT':>9} {'#|T|>3':>7} {'#null|T|>3':>10} {'occ-mean r':>10}")
for k in BANDS:
    R=np.array(rmaps[k]); N=np.array(rnull[k])
    t=R.mean(0)/(R.std(0,ddof=1)/np.sqrt(nsub)+1e-9)
    tn=N.mean(0)/(N.std(0,ddof=1)/np.sqrt(nsub)+1e-9)
    m=mask_acc
    occ_meanr=R.mean(0)[m].mean()
    print(f"{k:>7} {np.abs(t[m]).max():>12.2f} {np.abs(tn[m]).max():>9.2f} {(np.abs(t[m])>3).sum():>7d} {(np.abs(tn[m])>3).sum():>10d} {occ_meanr:>10.3f}")
print("\nmatched #|T|>3 >> null and clustered => real coupling. matched ~ null => none recoverable.")
