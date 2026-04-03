import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict

from src.eeg_encoder import EEGEncoderBIOT
from src.fmri_encoder import FMRIEncoder1D
from src.utils.dataset import SimultEEG_fMRI, ContrastiveBatchSampler
from src.utils.utils import multi_positive_clip_loss, count_params

class TrainConfig:
    data_dir = "data/projects/EEG_FMRI/data_indi_preproc/"
    checkpoint_dir = "checkpoints/"
    
    #dataset params
    eeg_sr = 250
    tr = 2.1
    eeg_win_sec = 2
    hrf_shifts_sec = [4.5, 5.0, 5.5]
    stride_tr = 1
    
    #models params
    n_eeg_channels = 64
    n_eeg_times = 500  # eeg_win_sec * eeg_sr
    n_roi = 200  # Schaefer 200 parcellation
    embed_dim = 128

    # BIOT EEG encoder
    biot_emb_size = 256
    biot_heads = 8
    biot_depth = 4
    biot_n_fft = 200
    biot_hop_length = 100
    biot_pretrained_path = "../BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"  # path to pretrained BIOT .pth, or None to train from scratch
    freeze_eeg_backbone = False  # freeze BIOT backbone (not recommended — breaks channel_tokens alignment)
    freeze_fmri_backbone = False
    eeg_backbone_lr_scale = 0.1  # backbone LR = learning_rate * this; projector uses full LR
    
    #training params
    batch_size = 64
    subs_per_batch = 2
    min_temp_dist = 10
    learning_rate = 1e-4
    weight_decay = 5e-2
    num_epochs = 100
    tau = 0.1
    early_stop_patience = 15

    split_mode = "time"  # "subject" or "time". Use "time" to diagnose learning; "subject" for cross-subject eval
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0
    save_every = 5
    batches_per_epoch = 50


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset_by_time(dataset, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Split by time within each subject/recording — val/test see same subjects as train."""
    random.seed(seed)

    # group by (sub, run) so we split each recording independently
    from collections import defaultdict
    pair_to_indices = defaultdict(list)
    for idx, (pair_id, tr_idx) in enumerate(dataset.index):
        pair_to_indices[pair_id].append((tr_idx, idx))

    train_indices, val_indices, test_indices = [], [], []
    for pair_id, items in pair_to_indices.items():
        items_sorted = [idx for _, idx in sorted(items)]
        n = len(items_sorted)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_indices.extend(items_sorted[:n_train])
        val_indices.extend(items_sorted[n_train:n_train + n_val])
        test_indices.extend(items_sorted[n_train + n_val:])

    subs = set(m["sub"] for m in dataset.meta)
    print(f"Subjects (all splits share): {sorted(subs)}")
    print(f"Samples: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    train_ds = SimultEEG_fMRI.from_subset(dataset, train_indices)
    val_ds = SimultEEG_fMRI.from_subset(dataset, val_indices)
    test_ds = SimultEEG_fMRI.from_subset(dataset, test_indices)
    return train_ds, val_ds, test_ds


def split_dataset_by_subjects(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    
    sub_to_indices = defaultdict(list)
    for idx, meta in enumerate(dataset.meta):
        sub_to_indices[meta["sub"]].append(idx)
    
    subjects = list(sub_to_indices.keys())
    random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, int(n_subjects * val_ratio))
    
    train_subs = subjects[:n_train]
    val_subs = subjects[n_train:n_train + n_val]
    test_subs = subjects[n_train + n_val:]
    
    if n_subjects == 1:
        all_indices = sub_to_indices[subjects[0]]
        n = len(all_indices)
        n_train_idx = int(n * train_ratio)
        n_val_idx = int(n * val_ratio)
        
        train_indices = all_indices[:n_train_idx]
        val_indices = all_indices[n_train_idx:n_train_idx + n_val_idx]
        test_indices = all_indices[n_train_idx + n_val_idx:]
    else:
        train_indices = [idx for sub in train_subs for idx in sub_to_indices[sub]]
        val_indices = [idx for sub in val_subs for idx in sub_to_indices[sub]]
        test_indices = [idx for sub in test_subs for idx in sub_to_indices[sub]]
    
    print(f"Subjects: train={len(train_subs)}, val={len(val_subs)}, test={len(test_subs)}")
    print(f"Samples: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    train_ds = SimultEEG_fMRI.from_subset(dataset, train_indices)
    val_ds = SimultEEG_fMRI.from_subset(dataset, val_indices)
    test_ds = SimultEEG_fMRI.from_subset(dataset, test_indices)
    
    return train_ds, val_ds, test_ds


def augment_eeg(eeg, noise_std=0.05, channel_drop_prob=0.1):
    """
    eeg: [B, K, C, T]
    Gaussian noise + per-sample channel dropout, training only.
    """
    eeg = eeg + torch.randn_like(eeg) * noise_std
    if channel_drop_prob > 0:
        # [B, 1, C, 1] — same mask across all K shifts and all time steps
        mask = torch.bernoulli(
            torch.full(
                (eeg.shape[0], 1, eeg.shape[2], 1),
                1.0 - channel_drop_prob,
                device=eeg.device,
            )
        )
        eeg = eeg * mask
    return eeg


def collate_fn(batch):
    """
    Custom collate function
    
    Element of batch:
        eeg: list of K numpy arrays, list of [C, T]
        fmri: numpy array [R]
        meta info
    
    Returns:
        eeg: tensor [B, K, C, T]
        fmri: tensor [B, R]
    """
    B = len(batch)
    K = len(batch[0]["eeg"])
    
    eeg_list = []
    for sample in batch:
        eeg_windows = np.stack(sample["eeg"], axis=0)
        eeg_list.append(eeg_windows)
    
    eeg = np.stack(eeg_list, axis=0)
    eeg = torch.from_numpy(eeg).float()
    
    fmri = np.stack([sample["fmri"] for sample in batch], axis=0)
    fmri = torch.from_numpy(fmri).float()
    
    return {
        "eeg": eeg,
        "fmri": fmri,
    }


def create_dataloaders(config):
    """Creates dataloader from config"""
    
    print("Loading dataset...")
    full_dataset = SimultEEG_fMRI(
        dirname=config.data_dir,
        eeg_sr=config.eeg_sr,
        tr=config.tr,
        eeg_win_sec=config.eeg_win_sec,
        hrf_shifts_sec=config.hrf_shifts_sec,
        stride_tr=config.stride_tr,
        n_eeg_channels=config.n_eeg_channels,
    )
    
    print(f"Total samples: {len(full_dataset)}")
    
    if config.split_mode == "time":
        train_ds, val_ds, test_ds = split_dataset_by_time(
            full_dataset,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
    else:
        train_ds, val_ds, test_ds = split_dataset_by_subjects(
            full_dataset,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
    
    train_sampler = ContrastiveBatchSampler(
        train_ds,
        batch_size=config.batch_size,
        subs_per_batch=min(config.subs_per_batch, len(set(m["sub"] for m in train_ds.meta))),
        min_temp_dist=config.min_temp_dist,
        drop_last=True,
        max_batches=config.batches_per_epoch,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        drop_last=False,
    )
    
    return train_loader, val_loader, test_loader


def create_models(config):
    eeg_encoder = EEGEncoderBIOT(
        n_channels=config.n_eeg_channels,
        emb_size=config.biot_emb_size,
        heads=config.biot_heads,
        depth=config.biot_depth,
        n_fft=config.biot_n_fft,
        hop_length=config.biot_hop_length,
        proj_dim=config.embed_dim,
    )
    if config.biot_pretrained_path is not None:
        eeg_encoder.load_pretrained(config.biot_pretrained_path)

    fmri_encoder = FMRIEncoder1D(
        embed_dim=config.embed_dim,
    )

    return eeg_encoder, fmri_encoder


def forward_batch(eeg_encoder, fmri_encoder, batch, device, augment=False):
    """
    Propagating batch through models
    
    Args:
        batch: dict with eeg [B, K, C, T] and fmri [B, R]
    
    Returns:
        z_f: [B, D]
        z_e: [B, K, D]
    """
    eeg = batch["eeg"].to(device)
    fmri = batch["fmri"].to(device)

    if augment:
        eeg = augment_eeg(eeg)

    B, K, C, T = eeg.shape

    z_f = fmri_encoder(fmri)

    eeg_flat = eeg.view(B * K, C, T)
    z_e_flat = eeg_encoder(eeg_flat)
    
    D = z_e_flat.shape[-1]
    z_e = z_e_flat.view(B, K, D)
    
    return z_f, z_e


def train_epoch(eeg_encoder, fmri_encoder, train_loader, optimizer, criterion, config):
    eeg_encoder.train()
    fmri_encoder.train()

    total_loss = 0.0
    num_batches = 0
    print("a")
    pbar = tqdm(train_loader, desc="Training", total=config.batches_per_epoch, leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        if num_batches == 0:
            print("Successfully started first training batch!")

        z_f, z_e = forward_batch(eeg_encoder, fmri_encoder, batch, config.device, augment=True)
        loss = criterion(z_f, z_e)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(eeg_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(fmri_encoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(eeg_encoder, fmri_encoder, val_loader, criterion, config, log_collapse=False):
    eeg_encoder.eval()
    fmri_encoder.eval()

    total_loss = 0.0
    num_batches = 0
    all_z_f, all_z_e0 = [], []

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        z_f, z_e = forward_batch(eeg_encoder, fmri_encoder, batch, config.device)
        loss = criterion(z_f, z_e)
        total_loss += loss.item()
        num_batches += 1
        if log_collapse:
            all_z_f.append(z_f)
            all_z_e0.append(z_e[:, 0, :])

    if log_collapse and all_z_f:
        import torch.nn.functional as F
        z_f_cat = F.normalize(torch.cat(all_z_f), dim=-1)
        z_e_cat = F.normalize(torch.cat(all_z_e0), dim=-1)
        # mean pairwise cosine sim (sampled) — close to 1.0 means collapse
        n = min(512, z_f_cat.shape[0])
        sim_f = (z_f_cat[:n] @ z_f_cat[:n].T).fill_diagonal_(0).sum() / (n * (n - 1))
        sim_e = (z_e_cat[:n] @ z_e_cat[:n].T).fill_diagonal_(0).sum() / (n * (n - 1))
        print(f"  [collapse check] mean cosine sim — fMRI: {sim_f:.3f}, EEG: {sim_e:.3f}  (>0.9 = collapsed)")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def compute_retrieval_accuracy(eeg_encoder, fmri_encoder, loader, config, top_k=(1, 5, 10)):
    eeg_encoder.eval()
    fmri_encoder.eval()
    
    all_z_f = []
    all_z_e = []
    
    for batch in loader:
        z_f, z_e = forward_batch(eeg_encoder, fmri_encoder, batch, config.device)
        all_z_f.append(z_f)
        all_z_e.append(z_e[:, 0, :])
    
    if len(all_z_f) == 0:
        return {}
    
    z_f = torch.cat(all_z_f, dim=0)
    z_e = torch.cat(all_z_e, dim=0)
    
    z_f = nn.functional.normalize(z_f, dim=-1)
    z_e = nn.functional.normalize(z_e, dim=-1)
    
    sim = z_f @ z_e.T
    
    N = sim.shape[0]
    targets = torch.arange(N, device=config.device)
    
    results = {}
    
    for k in top_k:
        if k > N:
            continue
        _, topk_indices = sim.topk(k, dim=1)
        correct = (topk_indices == targets.unsqueeze(1)).any(dim=1).float().mean()
        results[f"f2e_R@{k}"] = correct.item() * 100

    sim_t = sim.T
    for k in top_k:
        if k > N:
            continue
        _, topk_indices = sim_t.topk(k, dim=1)
        correct = (topk_indices == targets.unsqueeze(1)).any(dim=1).float().mean()
        results[f"e2f_R@{k}"] = correct.item() * 100
    
    return results


def save_checkpoint(eeg_encoder, fmri_encoder, optimizer, epoch, val_loss, config, is_best=False):
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "eeg_encoder_state_dict": eeg_encoder.state_dict(),
        "fmri_encoder_state_dict": fmri_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": vars(config),
    }
    
    torch.save(checkpoint, checkpoint_dir / "last_checkpoint.pt")
    
    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best_checkpoint.pt")
        print(f"Saving best checkpoint (val_loss={val_loss:.4f})")


def load_checkpoint(eeg_encoder, fmri_encoder, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    eeg_encoder.load_state_dict(checkpoint["eeg_encoder_state_dict"])
    fmri_encoder.load_state_dict(checkpoint["fmri_encoder_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"], checkpoint["val_loss"]


def train(config):
    set_seed(config.seed)
    
    print(f"device: {config.device}")
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    print(f"DEBUG: Number of batches in train_loader: {len(train_loader)}")
    if len(train_loader) == 0:
        print("ERROR: train_loader is empty! Check your Sampler constraints or Subject count.")
        return

    eeg_encoder, fmri_encoder = create_models(config)
    eeg_encoder.to(config.device)
    fmri_encoder.to(config.device)

    if config.freeze_eeg_backbone:
        if config.biot_pretrained_path is None:
            print("WARNING: freeze_eeg_backbone=True but no pretrained weights loaded — freezing a random backbone is not useful.")
        for name, param in eeg_encoder.encoder.named_parameters():
            if "channel_tokens" not in name:  # keep channel_tokens trainable — they are randomly initialized for this montage
                param.requires_grad = False

    if config.freeze_fmri_backbone:
        for name, param in fmri_encoder.named_parameters():
            if "proj" not in name:
                param.requires_grad = False

    eeg_total, eeg_trainable = count_params(eeg_encoder)
    fmri_total, fmri_trainable = count_params(fmri_encoder)
    print(f"EEGEncoder: {eeg_total:,} params ({eeg_trainable:,} trainable)")
    print(f"FMRIEncoder: {fmri_total:,} params ({fmri_trainable:,} trainable)")

    criterion = lambda z_f, z_e: multi_positive_clip_loss(z_f, z_e, tau=config.tau)

    backbone_lr = config.learning_rate * config.eeg_backbone_lr_scale
    optimizer = optim.AdamW(
        [
            {"params": [p for p in eeg_encoder.encoder.parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": [p for p in eeg_encoder.proj.parameters() if p.requires_grad]},
            {"params": [p for p in fmri_encoder.parameters() if p.requires_grad]},
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    print(f"Optimizer: backbone LR={backbone_lr:.2e}, projectors/fMRI LR={config.learning_rate:.2e}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=config.learning_rate * 0.01,
    )
    
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nЭпоха {epoch}/{config.num_epochs}")
        
        train_loss = train_epoch(eeg_encoder, fmri_encoder, train_loader, optimizer, criterion, config)

        val_loss = validate(eeg_encoder, fmri_encoder, val_loader, criterion, config, log_collapse=(epoch % 10 == 0))
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % config.save_every == 0 or is_best:
            save_checkpoint(eeg_encoder, fmri_encoder, optimizer, epoch, val_loss, config, is_best)

        if epoch % 10 == 0:
            metrics = compute_retrieval_accuracy(eeg_encoder, fmri_encoder, val_loader, config)
            if metrics:
                metrics_str = " | ".join([f"{k}: {v:.1f}%" for k, v in metrics.items()])
                print(f"Retrieval: {metrics_str}")

        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {config.early_stop_patience} epochs)")
            break
    
    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    
    print("\nLoading best model to test...")
    best_path = Path(config.checkpoint_dir) / "best_checkpoint.pt"
    if best_path.exists():
        load_checkpoint(eeg_encoder, fmri_encoder, None, best_path, config.device)
    
    test_loss = validate(eeg_encoder, fmri_encoder, test_loader, criterion, config)
    test_metrics = compute_retrieval_accuracy(eeg_encoder, fmri_encoder, test_loader, config)
    
    print(f"Test Loss: {test_loss:.4f}")
    if test_metrics:
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.1f}%")
    
    return eeg_encoder, fmri_encoder

from src.utils.preprocess_data import download_natview_subjects, extract_all_tar_gz
if __name__ == "__main__":
    
    download_natview_subjects(start_sub=1, end_sub=5, out_dir="data")
    #extract_all_tar_gz()
    config = TrainConfig()
    train(config)