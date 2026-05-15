import torch
import torch.nn.functional as F
import lightning as L
from src.eeg_encoder import EEGAugmentor
from src.fmri_encoder import FmriAugmentor
from train.config import TrainConfig
from src.utils.utils import multi_positive_clip_loss, alignment_metric, effective_rank
from peft import LoraConfig, get_peft_model

class ContrastiveModel(L.LightningModule):
    def __init__(self, eeg_encoder, fmri_encoder, config: TrainConfig):
        super().__init__();

        self.automatic_optimization = False

        self.eeg_encoder = eeg_encoder
        self.fmri_encoder = fmri_encoder

        eeg_lora = LoraConfig(
            r=config.train.lora_rank,
            lora_alpha=config.train.lora_alpha,
            target_modules=["qkv"],
            modules_to_save=["projector"],
            lora_dropout=config.train.lora_dropout,
            bias="none",
        )
        fmri_lora = LoraConfig(
            r=config.train.lora_rank,
            lora_alpha=config.train.lora_alpha,
            target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],
            modules_to_save=["projector"],
            lora_dropout=config.train.lora_dropout,
            bias="none",
        )

        self.config = config

        self.eeg_encoder  = get_peft_model(self.eeg_encoder, eeg_lora)
        self.fmri_encoder = get_peft_model(self.fmri_encoder, fmri_lora)

        self.eeg_augmentor = EEGAugmentor(config)
        self.fmri_augmentor = FmriAugmentor(config)

        # buffers for accumulating val embeddings across batches
        self._val_eeg, self._val_fmri = [], []

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x2 = x2.unsqueeze(1) #shape will be (B, 1, X, Y, Z, T)
        return self.eeg_encoder(x1), self.fmri_encoder(x2)
        
    def on_after_batch_transfer(self, batch, batch_idx):
        eeg, fmri = batch["eeg"], batch["fmri"]
        ch_names  = batch.get("ch_names")   # carried through as Python list (collate keeps it)

        # dataset stacks K HRF-shift windows: (B, K, C, T). With K=1 this dim is degenerate.
        if eeg.dim() == 4 and eeg.size(1) == 1:
            eeg = eeg.squeeze(1)

        if not self.training:
            return {"eeg": eeg, "fmri": fmri, "ch_names": ch_names}

        eeg_aug  = self.eeg_augmentor(eeg)
        fmri_aug = self.fmri_augmentor(fmri)
        return {"eeg": eeg_aug, "fmri": fmri_aug, "ch_names": ch_names}

    def training_step(self, batch, batch_idx):
        eeg, fmri = batch["eeg"], batch["fmri"]
        ch_names  = batch.get("ch_names")

        eeg_pred  = self.eeg_encoder(eeg, ch_names=ch_names)
        fmri_pred = self.fmri_encoder(fmri)

        K = self.config.data.num_subjects
        D = eeg_pred.shape[-1]
        eeg_mk  = eeg_pred.reshape(-1, K, D)
        fmri_mk = fmri_pred.reshape(-1, K, D)

        loss = multi_positive_clip_loss(fmri_mk, eeg_mk, self.config.train.tau)

        backbone_opt, projector_opt = self.optimizers()
        backbone_opt.zero_grad()
        projector_opt.zero_grad()
        self.manual_backward(loss)
        backbone_opt.step()
        projector_opt.step()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # cheap collapse / alignment diagnostics — SVD is fast on (B≤256, D≤256)
        with torch.no_grad():
            self.log("train/erank_eeg",  effective_rank(eeg_pred))
            self.log("train/erank_fmri", effective_rank(fmri_pred))
            self.log("train/align",      alignment_metric(fmri_mk, eeg_mk))

        return loss

    def validation_step(self, batch, batch_idx):
        eeg_pred  = self.eeg_encoder(batch["eeg"], ch_names=batch.get("ch_names"))
        fmri_pred = self.fmri_encoder(batch["fmri"])
        # store on CPU to avoid filling GPU memory during long val epochs
        self._val_eeg.append(eeg_pred.detach().float().cpu())
        self._val_fmri.append(fmri_pred.detach().float().cpu())

    def on_validation_epoch_end(self):
        if not self._val_eeg:
            return

        z_e = torch.cat(self._val_eeg, dim=0)   # [B, D]
        z_f = torch.cat(self._val_fmri, dim=0)
        self._val_eeg.clear()
        self._val_fmri.clear()

        z_en = F.normalize(z_e, dim=-1)
        z_fn = F.normalize(z_f, dim=-1)

        # per-pair alignment (positives = diagonal: same idx → same time/subject)
        align = (1.0 - (z_fn * z_en).sum(dim=-1)).mean()
        self.log("val/align", align)

        # collapse detection
        self.log("val/erank_eeg",  effective_rank(z_e))
        self.log("val/erank_fmri", effective_rank(z_f))

        # diagonal-positive retrieval — rank of true pair among all val candidates
        sim   = z_fn @ z_en.T                       # [B, B]
        diag  = sim.diag().unsqueeze(1)             # similarity of the true pair
        rank  = (sim >= diag).sum(dim=1).float()    # 1 = best
        rank_T = (sim.T >= sim.T.diag().unsqueeze(1)).sum(dim=1).float()

        for k in (1, 5, 10):
            self.log(f"val/r{k}_f2e", (rank   <= k).float().mean())
            self.log(f"val/r{k}_e2f", (rank_T <= k).float().mean())

        self.log("val/mrr_f2e", (1.0 / rank).mean())
        self.log("val/mrr_e2f", (1.0 / rank_T).mean(), prog_bar=True)
        
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