import torch
import lightning as L
from src.eeg_encoder import EEGAugmentor
from src.fmri_encoder import FmriAugmentor
from train.config import TrainConfig
from src.utils.utils import multi_positive_clip_loss
from peft import LoraConfig, get_peft_model

class ContrastiveModel(L.LightningModule):
    def __init__(self, eeg_encoder, fmri_encoder, config: TrainConfig):
        super().__init__();

        self.eeg_encoder = eeg_encoder
        self.fmri_encoder = fmri_encoder

        config = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            target_modules=["query", "value"], 
            exclude_modules=["projector"],
            lora_dropout=config.model.lora_dropout,
            bias="none"
        )

        self.eeg_augmentor = EEGAugmentor(config)
        self.fmri_augmentor = EEGAugmentor(config)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x2 = x2.unsqueeze(1) #shape will be (B, 1, X, Y, Z, T)
        return self.eeg_encoder(x1), self.fmri_encoder(x2)
        
    def on_after_batch_transfer(self, batch, batch_idx):
        #applying augmentations
        eeg, fmri = batch["eeg"], batch["fmri"]

        eeg_aug = self.eeg_augmentor(eeg)
        fmri_aug = self.fmri_augmentor(fmri)

        return {"eeg": eeg_aug, "fmri": fmri_aug}
        
    def _calculate_loss(self, x):

    def training_step(self, batch, batch_idx):

        
    def configure_optimizers(self):
        return super().configure_optimizers()