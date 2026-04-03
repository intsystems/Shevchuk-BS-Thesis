import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict

from src.eeg_encoder import EEGNet
from src.fmri_encoder import FMRIEncoderROI
from src.utils.dataset import SimultEEG_fMRI, ContrastiveBatchSampler
from src.utils.utils import multi_positive_clip_loss, count_params

class TrainConfig:
    data_dir = "data/"
    checkpoint_dir = "checkpoints/"
    
    #dataset params
    eeg_sr = 250
    tr = 2
    eeg_win_sec = 2
    hrf_shifts_sec = [4.0, 5.0, 6.0]
    stride_tr = 1
    
    #models params
    n_eeg_channels = 60
    n_eeg_times = 500  #eeg_win_sec * eeg_sr
    n_roi = 200  #Schaefer 200 parcellation
    embed_dim = 128
    
    eegnet_F1 = 8
    eegnet_D = 2
    eegnet_kernel_length = 64
    eegnet_separable_kernel = 16
    eegnet_pool1 = 4
    eegnet_pool2 = 8
    eegnet_dropout = 0.25
    
    #training params
    batch_size = 64
    subs_per_batch = 8
    min_temp_dist = 5
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 100
    tau = 0.07
    
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0
    save_every = 5
    batches_per_epoch = 100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    for i, sample in enumerate(batch):
        eeg_windows = np.stack(sample["eeg"], axis=0)
        eeg_list.append(eeg_windows)
    
    shapes = [arr.shape for arr in eeg_list]
    
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
    eeg_encoder = EEGNet(
        n_channels=config.n_eeg_channels,
        n_times=config.n_eeg_times,
        n_classes=256,
        F1=config.eegnet_F1,
        D=config.eegnet_D,
        kernel_length=config.eegnet_kernel_length,
        separable_kernel=config.eegnet_separable_kernel,
        pool1=config.eegnet_pool1,
        pool2=config.eegnet_pool2,
        dropout=config.eegnet_dropout,
        proj_dim=config.embed_dim,
    )
    
    fmri_encoder = FMRIEncoderROI(
        roi_dim=config.n_roi,
        embed_dim=config.embed_dim,
    )
    
    return eeg_encoder, fmri_encoder


def forward_batch(eeg_encoder, fmri_encoder, batch, device):
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
    
    B, K, C, T = eeg.shape

    z_f = fmri_encoder(fmri)
    
    eeg_flat = eeg.view(B * K, C, T)
    z_e_flat = eeg_encoder(eeg_flat)
    
    D = z_e_flat.shape[-1]
    z_e = z_e_flat.view(B, K, D)
    
    return z_f, z_e


def train_epoch(eeg_encoder, fmri_encoder, train_loader, optimizer, config):   
    eeg_encoder.train()
    fmri_encoder.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", total=config.batches_per_epoch, leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        
        z_f, z_e = forward_batch(eeg_encoder, fmri_encoder, batch, config.device)
        loss = multi_positive_clip_loss(z_f, z_e, tau=config.tau)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(eeg_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(fmri_encoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if num_batches >= config.batches_per_epoch:
            break
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(eeg_encoder, fmri_encoder, val_loader, config):
    eeg_encoder.eval()
    fmri_encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation", leave=False):
        z_f, z_e = forward_batch(eeg_encoder, fmri_encoder, batch, config.device)
        loss = multi_positive_clip_loss(z_f, z_e, tau=config.tau)
        
        total_loss += loss.item()
        num_batches += 1
    
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
    
    eeg_encoder, fmri_encoder = create_models(config)
    eeg_encoder.to(config.device)
    fmri_encoder.to(config.device)
    
    eeg_total, eeg_trainable = count_params(eeg_encoder)
    fmri_total, fmri_trainable = count_params(fmri_encoder)
    print(f"EEGNet: {eeg_total:,} params ({eeg_trainable:,} trainable)")
    print(f"FMRIEncoder: {fmri_total:,} params ({fmri_trainable:,} trainable)")
    
    params = list(eeg_encoder.parameters()) + list(fmri_encoder.parameters())
    optimizer = optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=config.learning_rate * 0.01,
        verbose=True,
    )
    
    best_val_loss = float("inf")
    
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nЭпоха {epoch}/{config.num_epochs}")
        
        train_loss = train_epoch(eeg_encoder, fmri_encoder, train_loader, optimizer, config)
        
        val_loss = validate(eeg_encoder, fmri_encoder, val_loader, config)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % config.save_every == 0 or is_best:
            save_checkpoint(eeg_encoder, fmri_encoder, optimizer, epoch, val_loss, config, is_best)
        
        if epoch % 10 == 0:
            metrics = compute_retrieval_accuracy(eeg_encoder, fmri_encoder, val_loader, config)
            if metrics:
                metrics_str = " | ".join([f"{k}: {v:.1f}%" for k, v in metrics.items()])
                print(f"Retrieval: {metrics_str}")
    
    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    
    print("\nLoading best model to test...")
    best_path = Path(config.checkpoint_dir) / "best_checkpoint.pt"
    if best_path.exists():
        load_checkpoint(eeg_encoder, fmri_encoder, None, best_path, config.device)
    
    test_loss = validate(eeg_encoder, fmri_encoder, test_loader, config)
    test_metrics = compute_retrieval_accuracy(eeg_encoder, fmri_encoder, test_loader, config)
    
    print(f"Test Loss: {test_loss:.4f}")
    if test_metrics:
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.1f}%")
    
    return eeg_encoder, fmri_encoder


if __name__ == "__main__":
    config = TrainConfig()
    train(config)