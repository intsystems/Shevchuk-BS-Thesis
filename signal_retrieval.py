"""Diagnostic step 1 — MODEL-FREE EEG->fMRI retrieval ceiling (drift-controlled).

Linear analog of the contrastive model: classical features (EEG per-channel
band-power, coarse BOLD voxels) -> fit CCA on TRAIN -> held-out retrieval MRR.

Controls vs the naive version:
  - channel INTERSECTION across subjects (keep all subjects, not just full-montage)
  - both BOLD and EEG-power timecourses BAND-PASSED 0.01-0.15 Hz (kill shared drift /
    motion / physiology that otherwise lets within-subject CCA cheat)
  - WITHIN-SUBJECT uses a CONTIGUOUS time split (first 70% train / last 30% test) so
    test TRs are not temporal neighbours of train TRs (no autocorrelation leakage)

Regimes: (A) within-subject (isolates cross-modal coupling), (B) cross-subject
(split subjects; matches the model's val task). Pairing matches dataset.py:
fMRI = mean BOLD over [tr:tr+8 TRs]; EEG = mean band-power over the 15s window
ending at tr*TR - 4.2s (TRs ~ [tr-9 : tr-2]).
"""
import sys
import h5py
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from train.config import TrainConfig
from src.utils.dataset import SimultEEG_fMRI

ACT = sys.argv[1] if len(sys.argv) > 1 else "task-dme"
MAXSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 14
STRIDE = 2
BLOCK, SR, TR = 4, 200, 2.1
WIN_SEC, HRF = 15.0, 4.2
SPT = int(round(TR*SR))
WIN_TR = int(np.ceil(WIN_SEC/TR))            # 8
LAG_TR = int(round(HRF/TR))                  # 2
MIN_TR = WIN_TR + LAG_TR + 1
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,12),"beta":(13,30)}  # gamma dropped (artifact-prone)
N_PCA_E, N_PCA_F, N_CC = 50, 80, 20
FS = 1.0/TR
bb, ba = butter(2, [0.01/(FS/2), 0.15/(FS/2)], btype="band")

cfg = TrainConfig(); ds = SimultEEG_fMRI(cfg)
pmap = {}
for p in ds.pairs:
    if p['activity']==ACT and p['sub'] not in pmap:
        pmap[p['sub']]=p
subs = sorted(pmap)[:MAXSUB]
# channel intersection across selected subjects
common = set(pmap[subs[0]]['channels'])
for s in subs: common &= set(pmap[s]['channels'])
CH = [c for c in cfg.data.ch_names if c in common]
efilt = {k:butter(4,[lo/(SR/2),hi/(SR/2)],btype="band") for k,(lo,hi) in BANDS.items()}

def coarse(v):
    T=v.shape[0]; return v.reshape(T,24,BLOCK,24,BLOCK,24,BLOCK).mean((2,4,6)).reshape(T,-1)

f = h5py.File("dataset.h5","r")
fmri_tc, eeg_tc, sig_acc = {}, {}, None     # per-TR timecourses, band-passed
for s in subs:
    p=pmap[s]
    bold=coarse(f[p['h5_grp']]["fmri"][:].astype(np.float32))   # (T,V)
    sig_acc = bold.mean(0) if sig_acc is None else sig_acc+bold.mean(0)
    eeg=f[p['h5_grp']]["eeg"][:].astype(np.float32)
    ci=[p['channels'].index(c) for c in CH]; eeg=eeg[:,ci]
    nb=min(bold.shape[0], eeg.shape[0]//SPT)
    # per-TR EEG log band-power per channel
    cols=[]
    for k,(b,a) in efilt.items():
        pw=filtfilt(b,a,eeg,axis=0)**2                          # (T_eeg,C)
        pwt=pw[:nb*SPT].reshape(nb,SPT,len(CH)).mean(1)         # (nb,C)
        cols.append(np.log(pwt+1e-8))
    etc=np.concatenate(cols,axis=1)                             # (nb, C*5)
    btc=bold[:nb]
    # band-pass both timecourses over time (remove drift)
    etc=filtfilt(bb,ba,etc,axis=0); btc=filtfilt(bb,ba,btc,axis=0)
    eeg_tc[s]=etc; fmri_tc[s]=btc
mask=(sig_acc/len(subs))>np.percentile(sig_acc/len(subs),55)
print(f"activity={ACT} subjects={len(subs)} commonCH={len(CH)} maskvox={mask.sum()} featEEG={len(CH)*5}")

rows=[]
for s in subs:
    etc=eeg_tc[s]; btc=fmri_tc[s]; T=btc.shape[0]
    for tr in range(MIN_TR, T-WIN_TR, STRIDE):
        fe=etc[tr-WIN_TR-LAG_TR:tr-LAG_TR].mean(0)              # EEG window mean
        ff=btc[tr:tr+WIN_TR].mean(0)[mask]
        rows.append((s,tr,fe,ff))
Xe=np.stack([r[2] for r in rows]); Xf=np.stack([r[3] for r in rows])
subj=np.array([r[0] for r in rows]); trs=np.array([r[1] for r in rows])
print(f"total windows={len(rows)}")

def chance_mrr(n): return float(np.mean([1.0/r for r in range(1,n+1)]))

def retrieval(tr_i,te_i,pos,permute=False):
    me,se=Xe[tr_i].mean(0),Xe[tr_i].std(0)+1e-6
    mf,sf=Xf[tr_i].mean(0),Xf[tr_i].std(0)+1e-6
    Pe=PCA(min(N_PCA_E,len(tr_i)-1),random_state=0).fit((Xe[tr_i]-me)/se)
    Pf=PCA(min(N_PCA_F,len(tr_i)-1),random_state=0).fit((Xf[tr_i]-mf)/sf)
    Ae=Pe.transform((Xe[tr_i]-me)/se); Af=Pf.transform((Xf[tr_i]-mf)/sf)
    if permute:  # break EEG<->fMRI correspondence in train -> null
        Af=Af[np.random.default_rng(7).permutation(len(Af))]
    k=min(N_CC,Ae.shape[1],Af.shape[1])
    cca=CCA(n_components=k,max_iter=1000).fit(Ae,Af)
    Ue,Uf=cca.transform(Pe.transform((Xe[te_i]-me)/se),Pf.transform((Xf[te_i]-mf)/sf))
    cc=[np.corrcoef(Ue[:,i],Uf[:,i])[0,1] for i in range(k)]
    A=Ue/(np.linalg.norm(Ue,axis=1,keepdims=True)+1e-9)
    B=Uf/(np.linalg.norm(Uf,axis=1,keepdims=True)+1e-9)
    sim=A@B.T; rank=sim.argsort(1)[:,::-1].argsort(1)+1
    first=np.where(pos,rank,sim.shape[1]+1).min(1)
    return float(np.mean(1.0/first)), np.array(cc)

print("\n=== (A) WITHIN-SUBJECT (contiguous 70/30 time split, drift-removed) ===")
mrrs,nulls,cc1,pools=[],[],[],[]
for s in subs:
    idx=np.where(subj==s)[0]                                    # already time-ordered
    if len(idx)<40: continue
    cut=int(0.7*len(idx)); tr_i,te_i=idx[:cut],idx[cut:]
    pos=(trs[te_i][:,None]==trs[te_i][None,:])
    m,cc=retrieval(tr_i,te_i,pos); mn,_=retrieval(tr_i,te_i,pos,permute=True)
    mrrs.append(m); nulls.append(mn); cc1.append(cc[0]); pools.append(len(te_i))
n=int(np.median(pools))
print(f"  mean held-out MRR={np.mean(mrrs):.4f}  chance(~{n})={chance_mrr(n):.4f}  ratio={np.mean(mrrs)/chance_mrr(n):.2f}x  topCC(mean)={np.mean(cc1):+.3f}")
print(f"  permuted-pairing null MRR={np.mean(nulls):.4f} ({np.mean(nulls)/chance_mrr(n):.2f}x) -- matched~null => artifact, not coupling")

print("\n=== (B) CROSS-SUBJECT (split subjects; pos=same moment, any subj) ===")
rng=np.random.default_rng(1); sl=list(subs); rng.shuffle(sl)
n_te=max(3,len(sl)//4); te_s=set(sl[:n_te])
trm=np.where(np.array([s not in te_s for s in subj]))[0]
tem=np.where(np.array([s in te_s for s in subj]))[0]
pos=(trs[tem][:,None]==trs[tem][None,:])
m,cc=retrieval(trm,tem,pos); N=len(tem)
mnull,_=retrieval(trm,tem,pos,permute=True)
print(f"  train subj={len(subs)-n_te} test subj={n_te} test pool={N}")
print(f"  EEG->fMRI MRR={m:.4f}  chance={chance_mrr(N):.4f}  ratio={m/chance_mrr(N):.2f}x")
print(f"  permuted-pairing null MRR={mnull:.4f} ({mnull/chance_mrr(N):.2f}x) -- should be ~1x if real")
print(f"  top-5 held-out canonical corr={np.round(cc[:5],3)}")
print("\nMRR>>chance & topCC>0 on held-out => recoverable signal a model could exploit.")
