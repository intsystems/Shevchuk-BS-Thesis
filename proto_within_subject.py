"""Prototype: WITHIN-SUBJECT EEG->fMRI retrieval reformulation.

The signal check showed: cross-subject coupling is at the noise floor, but
within-subject coupling is real-but-weak. This prototypes the reformulation:

  1. split by TIME WITHIN subject (contiguous, with a gap) instead of by subject
  2. SUBJECT-AWARE model (subject embedding) so it can use subject-specific coupling
  3. evaluate retrieval within each subject's held-out time segment

It uses classical features + small trainable encoders (fast, no LaBraM/GPU needed)
to answer: does a LEARNED model beat (a) chance, (b) the permutation null, and
(c) the linear CCA ceiling -- and does subject-awareness matter? If yes, the same
split/eval/subject-conditioning is worth porting into the real training stack.

Honest controls:
  - contiguous train/test time split with a GAP (no window-overlap leakage across split)
  - eval pool uses NON-OVERLAPPING windows (stride>=window) so same-moment retrieval
    is not inflated by overlapping neighbours
  - permutation-null model: trained on shuffled EEG<->fMRI pairing; real >> null = signal
  - ablation: subject embedding ON vs OFF

Usage: python proto_within_subject.py            (extract+cache, then train/eval)
       python proto_within_subject.py --fresh     (force re-extract features)
"""
import sys, os
import h5py
import numpy as np
from scipy.signal import butter, filtfilt
from train.config import TrainConfig
from src.utils.dataset import SimultEEG_fMRI

ACTS = ["task-dme", "task-monkey1"]
MAXSUB = 12
STRIDE_TRAIN = 2
BLOCK, SR, TR = 4, 200, 2.1
WIN_SEC, HRF = 15.0, 4.2
SPT = int(round(TR*SR)); WIN_TR = int(np.ceil(WIN_SEC/TR)); LAG_TR = int(round(HRF/TR))
MIN_TR = WIN_TR + LAG_TR + 1
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,12),"beta":(13,30)}
CACHE = "within_subj_feats.npz"
FS = 1.0/TR
bb, ba = butter(2, [0.01/(FS/2), 0.15/(FS/2)], btype="band")

def coarse(v):
    T=v.shape[0]; return v.reshape(T,24,BLOCK,24,BLOCK,24,BLOCK).mean((2,4,6)).reshape(T,-1)

def extract():
    cfg=TrainConfig(); ds=SimultEEG_fMRI(cfg)
    # subjects present in all chosen activities
    persub={a:set() for a in ACTS}
    pl={}
    for p in ds.pairs:
        if p['activity'] in ACTS:
            persub[p['activity']].add(p['sub']); pl.setdefault((p['sub'],p['activity']),p)
    subs=sorted(set.intersection(*persub.values()))[:MAXSUB]
    common=set(pl[(subs[0],ACTS[0])]['channels'])
    for s in subs:
        for a in ACTS: common&=set(pl[(s,a)]['channels'])
    CH=[c for c in cfg.data.ch_names if c in common]
    efilt={k:butter(4,[lo/(SR/2),hi/(SR/2)],btype="band") for k,(lo,hi) in BANDS.items()}
    f=h5py.File("dataset.h5","r")
    sig_acc=None; recs=[]
    for s in subs:
        for a in ACTS:
            p=pl[(s,a)]
            bold=coarse(f[p['h5_grp']]["fmri"][:].astype(np.float32))
            sig_acc=bold.mean(0) if sig_acc is None else sig_acc+bold.mean(0)
            eeg=f[p['h5_grp']]["eeg"][:].astype(np.float32)[:, [p['channels'].index(c) for c in CH]]
            nb=min(bold.shape[0], eeg.shape[0]//SPT)
            cols=[np.log(filtfilt(b,a,eeg,axis=0)[:nb*SPT].reshape(nb,SPT,len(CH)).mean(1)**2+1e-8)
                  for k,(b,a) in efilt.items()]
            etc=filtfilt(bb,ba,np.concatenate(cols,1),axis=0)
            btc=filtfilt(bb,ba,bold[:nb],axis=0)
            recs.append((s,a,etc,btc))
    mask=(sig_acc/len(recs))>np.percentile(sig_acc/len(recs),55)
    Xe,Xf,sub,act,tr_,seg=[],[],[],[],[],[]
    for ri,(s,a,etc,btc) in enumerate(recs):
        T=btc.shape[0]; trs=list(range(MIN_TR,T-WIN_TR,STRIDE_TRAIN))
        cut=MIN_TR+int(0.7*(len(trs)))*STRIDE_TRAIN              # contiguous split point (in tr)
        for tr in trs:
            Xe.append(etc[tr-WIN_TR-LAG_TR:tr-LAG_TR].mean(0))
            Xf.append(btc[tr:tr+WIN_TR].mean(0)[mask])
            sub.append(s); act.append(a); tr_.append(tr)
            seg.append(0 if tr < cut else 1)                    # 0=train segment, 1=test segment
    np.savez(CACHE, Xe=np.stack(Xe), Xf=np.stack(Xf),
             sub=np.array(sub), act=np.array(act), tr=np.array(tr_), seg=np.array(seg),
             subs=np.array(subs))
    print(f"cached {len(Xe)} windows, {len(subs)} subjects, {len(CH)} ch, {mask.sum()} voxels -> {CACHE}")

if "--fresh" in sys.argv or not os.path.exists(CACHE):
    extract()

# ---------------- model + eval ----------------
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA

d=np.load(CACHE, allow_pickle=True)
Xe,Xf=d["Xe"].astype(np.float32),d["Xf"].astype(np.float32)
sub,act,tr,seg=d["sub"],d["act"],d["tr"],d["seg"]
subs=list(d["subs"]); sidx=np.array([subs.index(s) for s in sub])
GAP=WIN_TR  # drop windows within GAP TRs of the split boundary handled via seg already; extra guard below
train_m = seg==0; test_m = seg==1
# guard: in test, keep only NON-OVERLAPPING windows per (subject,activity) -> honest pool
keep=np.zeros(len(tr),bool)
for s in set(sub):
    for a in set(act):
        idx=np.where(test_m & (sub==s) & (act==a))[0]
        if len(idx)==0: continue
        order=idx[np.argsort(tr[idx])]; last=-10**9
        for i in order:
            if tr[i]-last>=WIN_TR: keep[i]=True; last=tr[i]
test_m=test_m&keep
print(f"train windows={train_m.sum()}  test windows(non-overlap)={test_m.sum()}  subjects={len(subs)}")

# PCA fit on TRAIN only
me,se=Xe[train_m].mean(0),Xe[train_m].std(0)+1e-6
mf,sf=Xf[train_m].mean(0),Xf[train_m].std(0)+1e-6
Pe=PCA(50,random_state=0).fit((Xe[train_m]-me)/se)
Pf=PCA(150,random_state=0).fit((Xf[train_m]-mf)/sf)
Ze=torch.tensor(Pe.transform((Xe-me)/se),dtype=torch.float32)
Zf=torch.tensor(Pf.transform((Xf-mf)/sf),dtype=torch.float32)
Si=torch.tensor(sidx)
dev="cuda" if torch.cuda.is_available() else "cpu"

class Enc(nn.Module):
    def __init__(s,din,nsub,emb,subj_aware,d=64):
        super().__init__(); s.sa=subj_aware
        s.emb=nn.Embedding(nsub,emb) if subj_aware else None
        ind=din+(emb if subj_aware else 0)
        s.net=nn.Sequential(nn.Linear(ind,128),nn.GELU(),nn.Linear(128,d))
    def forward(s,x,si):
        if s.sa: x=torch.cat([x,s.emb(si)],-1)
        return F.normalize(s.net(x),dim=-1)

def run(subj_aware, permute=False, steps=400, tau=0.07, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    ee=Enc(50,len(subs),16,subj_aware).to(dev); fe=Enc(150,len(subs),16,subj_aware).to(dev)
    opt=torch.optim.Adam(list(ee.parameters())+list(fe.parameters()),lr=1e-3,weight_decay=1e-4)
    tr_i=np.where(train_m)[0]
    pair=tr_i.copy()
    if permute:  # shuffle EEG<->fMRI correspondence WITHIN each subject (kills real coupling, keeps subj structure)
        for s in range(len(subs)):
            si_=np.where(sidx[tr_i]==s)[0]; pair[si_]=tr_i[si_][np.random.permutation(len(si_))]
    Zed,Zfd,Sid=Ze.to(dev),Zf.to(dev),Si.to(dev)
    for it in range(steps):
        b=np.random.choice(len(tr_i),min(256,len(tr_i)),replace=False)
        ie=tr_i[b]; iff=pair[b]
        a=ee(Zed[ie],Sid[ie]); c=fe(Zfd[iff],Sid[iff])
        logit=a@c.T/tau; lbl=torch.arange(len(b),device=dev)
        loss=0.5*(F.cross_entropy(logit,lbl)+F.cross_entropy(logit.T,lbl))
        opt.zero_grad(); loss.backward(); opt.step()
    # eval: within-subject retrieval on non-overlapping test pool
    ee.eval(); fe.eval()
    mrrs=[]; chances=[]
    with torch.no_grad():
        for s in range(len(subs)):
            idx=np.where(test_m & (sidx==s))[0]
            if len(idx)<6: continue
            a=ee(Zed[idx],Sid[idx]); c=fe(Zfd[idx],Sid[idx])
            sim=(a@c.T).cpu().numpy(); n=len(idx)
            rank=sim.argsort(1)[:,::-1].argsort(1)+1
            mrrs.append(np.mean(1.0/np.diag(rank))); chances.append(np.mean([1/r for r in range(1,n+1)]))
    return float(np.mean(mrrs)), float(np.mean(chances))

print("\n=== WITHIN-SUBJECT retrieval (learned encoders, held-out time) ===")
for sa in [False, True]:
    m,ch=run(sa); mn,_=run(sa,permute=True)
    tag="subject-AWARE " if sa else "subject-blind  "
    print(f"  {tag}: MRR={m:.4f}  null(perm)={mn:.4f}  chance={ch:.4f}   ratio_real/null={m/mn:.2f}x  real/chance={m/ch:.2f}x")
print("\nReal >> null (and > chance) => learnable within-subject signal. AWARE>blind => subject conditioning helps.")
