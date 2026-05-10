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

        self.automatic_optimization = False

        self.eeg_encoder = eeg_encoder
        self.fmri_encoder = fmri_encoder

        eeg_lora = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            target_modules=["qkv"],
            exclude_modules=["projector"],
            lora_dropout=config.model.lora_dropout,
            bias="none",
        )
        fmri_lora = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],
            exclude_modules=["projector"],
            lora_dropout=config.model.lora_dropout,
            bias="none",
        )

        self.config = config

        self.eeg_encoder  = get_peft_model(self.eeg_encoder, eeg_lora)
        self.fmri_encoder = get_peft_model(self.fmri_encoder, fmri_lora)

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
        
    def training_step(self, batch, batch_idx):
        eeg, fmri = batch["eeg"], batch["fmri"]

        eeg_pred = self.eeg_encoder(eeg)
        fmri_pred = self.fmri_encoder(fmri)

        eeg_pred = torch.reshape(
            eeg_pred,
            (eeg_pred.shape[0] // self.config.data.num_subjects,
              self.config.data.num_subjects, -1)
              )
        fmri_pred = torch.reshape(
            fmri_pred,
            (fmri_pred.shape[0] // self.config.data.num_subjects,
              self.config.data.num_subjects, -1)
              )

        loss = multi_positive_clip_loss(fmri_pred, eeg_pred, self.config.train.tau)

        backbone_opt, projector_opt = self.optimizers()
        backbone_opt.zero_grad()
        projector_opt.zero_grad()
        self.manual_backward(loss)
        backbone_opt.step()
        projector_opt.step()

        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        backbone_params, projector_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "projector" in name:
                projector_params.append(p)
            else:
                backbone_params.append(p)

        backbone_opt = torch.optim.AdamW(
            backbone_params,
            lr=self.config.train.backbone_lr,
            weight_decay=self.config.train.weight_decay,
        )
        projector_opt = torch.optim.AdamW(
            projector_params,
            lr=self.config.train.proj_lr,
            weight_decay=self.config.train.weight_decay,
        )
        return [backbone_opt, projector_opt]