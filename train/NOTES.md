# Session Notes — EEG-fMRI Contrastive Training

## 1. Preprocessing Pipeline Audit

### Critical bugs fixed

**`FMRIEncoder1D.proj` API crash** (`src/fmri_encoder.py`)
- `Projector(hidden_dim // 2, embed_dim)` passed integers as `config` → `AttributeError` at runtime
- Fixed: changed `FMRIEncoder1D.__init__` to accept `config: TrainConfig`, reads `n_roi` from `config.model.n_roi`, passes `Projector(config, in_dim=hidden_dim // 2)`
- `build_model` in `train/train.py` simplified to `FMRIEncoder1D(config)`

**EEG window direction** (`src/utils/dataset.py`)
- `ts_eeg = t_fmri - hrf_shifts` caused windows `[t_fmri - shift, t_fmri - shift + 15s]`
- With shift=2.1s, 12.9s of the EEG window was *after* t_fmri — future neural activity that hadn't caused the current BOLD response
- Fixed: `ts_eeg = t_fmri - hrf_shifts - eeg_win_sec` → window `[t_fmri - shift - 15s, t_fmri - shift]`, entirely before the BOLD frame
- Boundary calculations updated consistently:
  - `min_tr_off`: `ceil(max_shift / TR)` → `ceil((max_shift + win_sec) / TR)` (3 → 11 TRs)
  - `max_t_fmri`: removed `- eeg_win_sec` term (last valid EEG sample is now `t_fmri - min_shift`, not `t_fmri - min_shift + win_sec`)

### Significant issues identified (not yet fixed)

**fMRI volumetric normalization** (`src/utils/preprocess_data.py:123-126`)
- Global z-normalization across all brain voxels × all timepoints combined
- Conflates spatial intensity heterogeneity with temporal BOLD variance
- Should be per-voxel (normalize each voxel's time series independently)
- Parcellated path already does this correctly (per-ROI, `axis=1`)

### Minor issues (stale config, no functional impact)
- `ModelConfig.n_eeg_times = 500` should be `eeg_win_sec * eeg_sr = 3000`
- `DataConfig.n_eeg_channels = 60` vs `ModelConfig.n_eeg_channels = 64` — neither used in hot path

---

## 2. Training Diagnostics

### Overfit sanity check (K=16, M=8) — PASS
- `sim_gap` reached 0.27, loss dropped to 0.50
- Confirmed: architecture and loss are sound, gradient flows correctly

### Full training (K=16, M=8, tau=0.07) — STALLED
- `sim_gap` oscillated near 0 for 900 steps
- `erank_eeg ≈ 20`, `erank_fmri ≈ 25` (no collapse)
- Diagnosis: two well-structured but uncorrelated embedding spaces, cold-start alignment problem

### H3 gradient coherence diagnostic added (`src/lit_model.py`)
- `train/grad_cos`: cosine similarity between consecutive step gradients
- Interpretation: ~0 means incoherent gradient direction across batches

### K=2, M=8, margin=5 run — FAILED
- `grad_cos ≈ 0` (oscillating [−0.07, +0.12]) — incoherent gradients confirmed
- `erank` collapsed 14 → 10 (BatchNorm instability from batch size = 16)
- `train/align ≈ 1.0` throughout — zero cross-modal alignment

### Root cause analysis
Three hypotheses were tested:
1. **H1 (EEG window direction)** — structural fix applied (see above)
2. **H2 (multi-subject positives)** — ISC ≈ 0.3–0.5 means 50–70% of "positive" pairs were noisy; treated all K subjects at same TR as mutual positives
3. **H3 (gradient coherence)** — diagnostic added; confirmed incoherent gradients with K>1

---

## 3. Current Configuration

### Key config values (`train/config.py`)
```
num_subjects  = 1      # K=1: clean diagonal positives, no cross-subject ISC assumption
num_timestamps = 32    # M=32: batch size = M×K = 32, sufficient for stable BN
margin_tr     = 2      # 4.2s separation between timestamps (was 5 → 10.5s)
tau           = 0.15   # raised from 0.07 for sharper gradients during alignment phase
overfit_batches = 1    # currently in overfit mode — set to 0 for full training
max_steps     = 500
```

### Overfit result (K=1, M=32, margin=2) — STRONG PASS
- `sim_gap` → 0.75 (vs 0.27 with K=16, vs ~0.04 with K=2)
- `train/align` → 0.20 (from 1.0 — real cross-modal alignment learned)
- `grad_cos` ≈ 0.05–0.20, consistently positive (coherent gradient direction)
- `erank` stable at 25–27 (no collapse)

### Feasibility check for K=1, M=32, margin=2
- 270/272 recordings have enough valid TRs (≥32 margin-separated TRs available)
- Expected ~270 batches/epoch (vs 52 with K=2)

---

## 4. Next Steps

1. **Set `overfit_batches=0`** and launch full training
2. Watch `train/grad_cos` — if it stays ≥0.05 in full training, the alignment signal generalises
3. Watch `train/align` — should trend down from 1.0; any consistent decrease is progress
4. **Fix volumetric fMRI normalization** (per-voxel z-score) when re-preprocessing data
5. Consider annealing `tau` from 0.15 back to 0.07 once `sim_gap > 0.2` in full training
