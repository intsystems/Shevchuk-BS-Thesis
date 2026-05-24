import torch
import numpy as np
import random
from dataclasses import asdict
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.eeg_encoder import EEGEncoder
from src.fmri_encoder import FMRIEncoderVolume, FMRIEncoder1D
from src.lit_model import ContrastiveModel
from src.utils.dataset import SimultEEG_fMRI, ContrastiveBatchSampler, collate_fn
from src.utils.utils import count_params, multi_positive_clip_loss
from train.config import TrainConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_subjects(config: TrainConfig):
    rng = random.Random(config.train.seed)
    subjects = [f"sub-{i:02d}" for i in range(config.data.start_sub, config.data.end_sub + 1)]
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * config.train.train_ratio)
    n_val   = int(n * config.train.val_ratio)

    train_subs = subjects[:n_train]
    val_subs   = subjects[n_train:n_train + n_val]
    test_subs  = subjects[n_train + n_val:]
    return train_subs, val_subs, test_subs


def build_loaders(config: TrainConfig):
    train_subs, val_subs, test_subs = split_subjects(config)
    print(f"[SPLIT] train={len(train_subs)}  val={len(val_subs)}  test={len(test_subs)}")

    train_ds = SimultEEG_fMRI(config, subjects=train_subs)
    val_ds   = SimultEEG_fMRI(config, subjects=val_subs)
    test_ds  = SimultEEG_fMRI(config, subjects=test_subs)

    train_sampler = ContrastiveBatchSampler(train_ds, config)
    nw = config.train.num_workers
    # Workers are spawned (not forked) so each starts fresh without inheriting
    # the parent's model weights in CPU memory (~6 GB per forked worker → OOM with 8 workers).
    ctx = "spawn" if nw > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=1 if nw > 0 else None,
        persistent_workers=nw > 0,
        multiprocessing_context=ctx,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=1 if nw > 0 else None,
        persistent_workers=nw > 0,
        multiprocessing_context=ctx,
    )
    return train_loader, val_loader, test_loader


def build_model(config: TrainConfig):
    eeg_encoder = EEGEncoder(config)
    if config.data.use_parcellation:
        fmri_encoder = FMRIEncoder1D(config)
    else:
        fmri_encoder = FMRIEncoderVolume(config)

    print(f"[EEG]  total/trainable: {count_params(eeg_encoder)}")
    print(f"[fMRI] total/trainable: {count_params(fmri_encoder)}")

    return ContrastiveModel(eeg_encoder, fmri_encoder, config)


def profile_run(config: TrainConfig, n_steps: int = 8):
    """
    Run a few forward/backward passes under torch.profiler to pinpoint the bottleneck.
    Writes a Chrome trace to logs/profiler/ (open with chrome://tracing or Perfetto).
    Prints a summary table sorted by CPU and CUDA time.
    """
    from torch.profiler import (
        profile, record_function, ProfilerActivity,
        schedule, tensorboard_trace_handler,
    )

    train_loader, _, _ = build_loaders(config)
    model = build_model(config).cuda().train()
    opt1, opt2 = model.configure_optimizers()
    device = next(model.parameters()).device

    Path("logs/profiler").mkdir(parents=True, exist_ok=True)
    wait, warmup, active = 1, 2, n_steps
    loader_iter = iter(train_loader)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler("logs/profiler"),
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for step in range(wait + warmup + active):
            with record_function("data_load"):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    batch = next(loader_iter)

            with record_function("to_gpu"):
                eeg      = batch["eeg"].to(device)
                fmri     = batch["fmri"].to(device)
                ch_names = batch.get("ch_names")
                if eeg.dim() == 4 and eeg.size(1) == 1:
                    eeg = eeg.squeeze(1)

            with record_function("forward"):
                eeg_pred  = model.eeg_encoder(eeg, ch_names=ch_names)
                fmri_pred = model.fmri_encoder(fmri)
                K, D = config.data.num_subjects, eeg_pred.shape[-1]
                loss = multi_positive_clip_loss(
                    fmri_pred.reshape(-1, K, D),
                    eeg_pred.reshape(-1, K, D),
                    config.train.tau,
                )

            with record_function("backward"):
                opt1.zero_grad()
                opt2.zero_grad()
                loss.backward()
                opt1.step()
                opt2.step()

            prof.step()

    print("\n=== Bottleneck report — sorted by CPU time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print("\n=== Bottleneck report — sorted by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print("\nChrome trace written to logs/profiler/ — open with Perfetto (https://ui.perfetto.dev)")


def main():
    config = TrainConfig()
    set_seed(config.train.seed)

    if config.train.profile:
        profile_run(config)
        return

    train_loader, val_loader, test_loader = build_loaders(config)

    print(f"[DATA] train_ds={len(train_loader.dataset)} val_ds={len(val_loader.dataset)} test_ds={len(test_loader.dataset)}", flush=True)
    print(f"[DATA] train_loader={len(train_loader)} val_loader={len(val_loader)} test_loader={len(test_loader)}", flush=True)

    # overfit_batches=N doesn't cache — it re-reads from HDF5 and re-transfers
    # CPU→GPU every step. Pre-load once and park the batch on GPU so the
    # transfer (0.3s/step per profiler) becomes a no-op on every subsequent step.
    if config.train.overfit_batches:
        print("[OVERFIT] Pre-loading single batch onto GPU to avoid per-step H2D transfer...")
        _cached_cpu = next(iter(train_loader))
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cached = {
            k: v.to(_device) if isinstance(v, torch.Tensor) else v
            for k, v in _cached_cpu.items()
        }
        del _cached_cpu

        class _CachedLoader:
            def __iter__(self):
                yield _cached
            def __len__(self):
                return 1

        train_loader = _CachedLoader()

    model = build_model(config)

    ckpt_dir = Path(config.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch={epoch:02d}-mrr={val/mrr_e2f:.4f}",
            monitor="val/mrr_e2f",
            mode="max",
            save_top_k=3,
            every_n_epochs=config.train.save_every,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = WandbLogger(
        project=config.train.wandb_project,
        group=config.train.wandb_group,
        save_dir="logs",
        config=asdict(config),
    )

    trainer = L.Trainer(
        max_epochs=-1,
        max_steps=config.train.max_steps,
        overfit_batches=config.train.overfit_batches,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1 if config.train.overfit_batches else 10,
        num_sanity_val_steps=0 if config.train.overfit_batches else 2,
        check_val_every_n_epoch=10**9 if config.train.overfit_batches else 1,
        gradient_clip_val=None,  # manual optimization → clipping done in training_step via clip_grad_norm_
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if not config.train.overfit_batches:
        trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
