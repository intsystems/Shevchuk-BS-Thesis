import torch
import numpy as np
import random
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.eeg_encoder import EEGEncoder
from src.fmri_encoder import FMRIEncoderVolume, FMRIEncoder1D
from src.lit_model import ContrastiveModel
from src.utils.dataset import SimultEEG_fMRI, ContrastiveBatchSampler, collate_fn
from src.utils.utils import count_params
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
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # val/test: simple sequential — used for retrieval metrics, not contrastive batches
    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def build_model(config: TrainConfig):
    eeg_encoder = EEGEncoder(config)
    if config.data.use_parcellation:
        fmri_encoder = FMRIEncoder1D(
            n_roi=config.model.n_roi,
            embed_dim=config.model.projector_out_dim,
        )
    else:
        fmri_encoder = FMRIEncoderVolume(config)

    print(f"[EEG]  total/trainable: {count_params(eeg_encoder)}")
    print(f"[fMRI] total/trainable: {count_params(fmri_encoder)}")

    return ContrastiveModel(eeg_encoder, fmri_encoder, config)


def main():
    config = TrainConfig()
    set_seed(config.train.seed)

    #train_loader, val_loader, test_loader = build_loaders(config)
    model = build_model(config)

    ckpt_dir = Path(config.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch={epoch:02d}-loss={train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_top_k=3,
            every_n_epochs=config.train.save_every,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(save_dir="logs", name="contrastive_eeg_fmri")

    trainer = L.Trainer(
        max_epochs=config.train.num_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=None,  # manual optimization → clipping must be done inside training_step if needed
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
