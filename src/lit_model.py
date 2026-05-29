import torch
import torch.nn.functional as F
import lightning as L
from src.eeg_encoder import EEGAugmentor
from src.fmri_encoder import FmriAugmentor
from train.config import TrainConfig
from src.utils.utils import multi_positive_clip_loss, within_subject_clip_loss, alignment_metric, effective_rank, _variance_loss
from peft import LoraConfig, get_peft_model

class ContrastiveModel(L.LightningModule):
    def __init__(self, eeg_encoder, fmri_encoder, config: TrainConfig):
        super().__init__();

        self.automatic_optimization = False

        self.eeg_encoder = eeg_encoder
        self.fmri_encoder = fmri_encoder

        eeg_lora = LoraConfig(
            r=config.train.eeg_lora_rank,
            lora_alpha=config.train.eeg_lora_alpha,
            target_modules=["proj"],   # "qkv" is bypassed via F.linear(weight=.weight); "proj" is called normally
            modules_to_save=["projector"],
            lora_dropout=config.train.lora_dropout,
            bias="none",
        )
        fmri_lora = LoraConfig(
            r=config.train.fmri_lora_rank,
            lora_alpha=config.train.fmri_lora_alpha,
            # NeuroSTORM backbone is pure Mamba; Mamba projections (in_proj/out_proj/etc.)
            # use a CUDA fast path that accesses .weight directly, bypassing the LoRA wrapper.
            # mlp.linear1/linear2 are standard PyTorch Linear called normally via MLPBlock.forward.
            target_modules=["linear1", "linear2"],
            modules_to_save=["projector"],
            lora_dropout=config.train.lora_dropout,
            bias="none",
        )

        self.config = config

        self.eeg_encoder  = get_peft_model(self.eeg_encoder, eeg_lora)
        self.fmri_encoder = get_peft_model(self.fmri_encoder, fmri_lora)

        def _param_breakdown(label, peft_model):
            lora  = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_A" in n or "lora_B" in n)
            proj  = sum(p.numel() for n, p in peft_model.named_parameters() if "projector" in n and p.requires_grad)
            total = sum(p.numel() for p in peft_model.parameters())
            print(f"[{label}]  LoRA={lora:,}  projector={proj:,}  total_trainable={lora+proj:,}  frozen={total-lora-proj:,}")

        _param_breakdown("EEG ", self.eeg_encoder)
        _param_breakdown("fMRI", self.fmri_encoder)

        self.eeg_augmentor = EEGAugmentor(config)
        self.fmri_augmentor = FmriAugmentor(config)

        # buffers for accumulating val embeddings across batches
        self._val_eeg, self._val_fmri, self._val_subs = [], [], []

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x2 = x2.unsqueeze(1) #shape will be (B, 1, X, Y, Z, T)
        return self._encode_eeg(x1), self.fmri_encoder(x2)

    def _encode_eeg(self, eeg: torch.Tensor, ch_names=None) -> torch.Tensor:
        """
        eeg: (B, C, T) with single HRF shift, or (B, n_shifts, C, T) with multiple.
        For multiple shifts: encode each shift sequentially to avoid n_shifts × memory spike,
        then average embeddings to get an HRF-robust representation.
        """
        if eeg.dim() == 3:
            return self.eeg_encoder(eeg, ch_names=ch_names)
        embs = [self.eeg_encoder(eeg[:, i], ch_names=ch_names) for i in range(eeg.size(1))]
        return torch.stack(embs, dim=1).mean(dim=1)    # (B, D)
        
    def on_after_batch_transfer(self, batch, batch_idx):
        eeg, fmri = batch["eeg"], batch["fmri"]
        ch_names  = batch.get("ch_names")   # carried through as Python list (collate keeps it)
        sub_id    = batch.get("sub_id")     # [B] long, for the identity-controlled loss term
        slot_id   = batch.get("slot_id")    # [B] long, task-moment grouping

        # fMRI is kept as float16 through CPU/worker pipeline to halve memory; cast here on GPU.
        fmri = fmri.float()

        # dataset stacks K HRF-shift windows: (B, K, C, T). With K=1 this dim is degenerate.
        if eeg.dim() == 4 and eeg.size(1) == 1:
            eeg = eeg.squeeze(1)

        if not self.training or not self.config.train.using_aug:
            return {"eeg": eeg, "fmri": fmri, "ch_names": ch_names,
                    "sub_id": sub_id, "slot_id": slot_id}

        # EEGAugmentor expects (B, C, T). For multi-HRF input (B, K_hrf, C, T)
        # fold K_hrf into batch so each shift is independently augmented, then
        # restore the original layout.
        if eeg.dim() == 4:
            B, K_hrf, C, T = eeg.shape
            eeg_aug = self.eeg_augmentor(eeg.reshape(B * K_hrf, C, T)).reshape(B, K_hrf, C, T)
        else:
            eeg_aug = self.eeg_augmentor(eeg)
        fmri_aug = self.fmri_augmentor(fmri)
        return {"eeg": eeg_aug, "fmri": fmri_aug, "ch_names": ch_names,
                "sub_id": sub_id, "slot_id": slot_id}

    def training_step(self, batch, batch_idx):
        eeg, fmri = batch["eeg"], batch["fmri"]
        ch_names  = batch.get("ch_names")

        eeg_pred  = self._encode_eeg(eeg, ch_names=ch_names)
        fmri_pred = self.fmri_encoder(fmri)

        K = self.config.data.num_subjects
        D = eeg_pred.shape[-1]
        eeg_mk  = eeg_pred.reshape(-1, K, D)
        fmri_mk = fmri_pred.reshape(-1, K, D)

        loss = multi_positive_clip_loss(fmri_mk, eeg_mk, self.config.train.tau)

        # identity-controlled hard-negative term: rank the true moment above OTHER
        # moments of the SAME subject. Always logged as a diagnostic; only folded
        # into the loss when weight > 0 (at weight=0 it's a pure metric, so compute
        # under no_grad to skip building the unused autograd graph).
        w = self.config.train.within_subject_weight
        with torch.set_grad_enabled(w > 0):
            loss_within = within_subject_clip_loss(
                fmri_pred, eeg_pred,
                batch["sub_id"], batch["slot_id"],
                self.config.train.tau,
            )
        self.log("train/loss_within", loss_within, on_step=True, on_epoch=False)
        if w > 0:
            loss = loss + w * loss_within

        vw = self.config.train.variance_weight
        if vw > 0:
            loss_var = _variance_loss(eeg_pred) + _variance_loss(fmri_pred)
            self.log("train/loss_var", loss_var, on_step=True, on_epoch=False)
            loss = loss + vw * loss_var

        backbone_opt, projector_opt = self.optimizers()
        backbone_sched, projector_sched = self.lr_schedulers()
        backbone_opt.zero_grad()
        projector_opt.zero_grad()
        self.manual_backward(loss)

        lora_grads = [p.grad for n, p in self.named_parameters() if "projector" not in n and p.requires_grad and p.grad is not None]
        proj_grads = [p.grad for n, p in self.named_parameters() if "projector" in n and p.requires_grad and p.grad is not None]
        lora_norm = torch.stack([g.norm(2) for g in lora_grads]).norm(2) if lora_grads else torch.zeros(1, device=self.device)
        proj_norm = torch.stack([g.norm(2) for g in proj_grads]).norm(2) if proj_grads else torch.zeros(1, device=self.device)

        trainable = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=self.config.train.grad_clip_val)
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)
        self.log("train/lora_grad_norm", lora_norm, on_step=True, on_epoch=False)
        self.log("train/proj_grad_norm", proj_norm, on_step=True, on_epoch=False)
        if proj_norm > 0:
            self.log("train/lora_proj_grad_ratio", lora_norm / proj_norm, on_step=True, on_epoch=False)

        curr_grads = torch.cat([p.grad.flatten() for p in trainable]).detach()
        if hasattr(self, "_prev_grads"):
            cos = F.cosine_similarity(curr_grads.unsqueeze(0), self._prev_grads.unsqueeze(0)).item()
            self.log("train/grad_cos", cos, on_step=True, on_epoch=False)
        self._prev_grads = curr_grads

        backbone_opt.step()
        projector_opt.step()
        backbone_sched.step()
        projector_sched.step()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # cheap collapse / alignment diagnostics — SVD is fast on (B≤256, D≤256)
        with torch.no_grad():
            self.log("train/erank_eeg",  effective_rank(eeg_pred))
            self.log("train/erank_fmri", effective_rank(fmri_pred))
            self.log("train/align",      alignment_metric(fmri_mk, eeg_mk))

            # pos vs neg similarity gap — direct collapse indicator
            B, D = eeg_pred.shape
            zf = F.normalize(fmri_pred, dim=-1)
            ze = F.normalize(eeg_pred,  dim=-1)
            sim = zf @ ze.T                          # [B, B]
            K   = self.config.data.num_subjects
            T   = B // K
            mask_pos = torch.kron(
                torch.eye(T, device=sim.device),
                torch.ones(K, K, device=sim.device),
            ).bool()
            sim_pos = sim[mask_pos].mean()
            sim_neg = sim[~mask_pos].mean()
            self.log("train/sim_pos", sim_pos)
            self.log("train/sim_neg", sim_neg)
            self.log("train/sim_gap", sim_pos - sim_neg, prog_bar=True)
            self.log("train/sim_pos_std", sim[mask_pos].std())
            self.log("train/sim_neg_std", sim[~mask_pos].std())

            mu_e = F.normalize(ze.mean(0, keepdim=True), dim=-1)
            mu_f = F.normalize(zf.mean(0, keepdim=True), dim=-1)
            self.log("train/modality_gap", 1.0 - (mu_e * mu_f).sum())

            off_diag = ~torch.eye(B, dtype=torch.bool, device=sim.device)
            self.log("train/intra_eeg_sim",  (ze @ ze.T)[off_diag].mean())
            self.log("train/intra_fmri_sim", (zf @ zf.T)[off_diag].mean())
            self.log("train/cross_modal_sim", sim.mean())

            # subject-identity diagnostic: split negatives into same-subject vs cross-subject
            subs = batch.get("sub")
            if subs is not None:
                sub_ids = torch.tensor(
                    [hash(s) & 0x7FFF_FFFF for s in subs], device=sim.device
                )
                same_sub  = sub_ids.unsqueeze(0) == sub_ids.unsqueeze(1)   # [B, B]
                neg_same  = ~mask_pos &  same_sub
                neg_cross = ~mask_pos & ~same_sub
                if neg_same.any():
                    self.log("train/sim_neg_same_sub",  sim[neg_same].mean(), on_step=True)
                if neg_cross.any():
                    self.log("train/sim_neg_diff_sub",  sim[neg_cross].mean(), on_step=True)
                if neg_same.any() and neg_cross.any():
                    self.log("train/sim_neg_id_gap",
                             sim[neg_same].mean() - sim[neg_cross].mean(), on_step=True)

        return loss

    def on_validation_epoch_start(self):
        with open("val_debug.log", "a") as f:
            f.write(f"START step={self.global_step}\n")

    def validation_step(self, batch, batch_idx):
        with open("val_debug.log", "a") as f:
            f.write(f"  step {batch_idx} eeg={tuple(batch['eeg'].shape)}\n")

        eeg_pred  = self._encode_eeg(batch["eeg"], ch_names=batch.get("ch_names"))
        fmri_pred = self.fmri_encoder(batch["fmri"])
        # store on CPU to avoid filling GPU memory during long val epochs
        self._val_eeg.append(eeg_pred.detach())
        self._val_fmri.append(fmri_pred.detach())
        if batch.get("sub") is not None:
            self._val_subs.extend(batch["sub"])

    def on_validation_epoch_end(self):
        with open("val_debug.log", "a") as f:
            f.write(f"END accumulated={len(self._val_eeg)}\n")
        if not self._val_eeg:
            return

        z_e = torch.cat(self._val_eeg, dim=0).float()   # [B, D]
        z_f = torch.cat(self._val_fmri, dim=0).float()
        val_subs = list(self._val_subs)
        self._val_eeg.clear()
        self._val_fmri.clear()
        self._val_subs.clear()

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

        # --- diagnostic metrics ---
        B = z_fn.shape[0]
        off_diag = ~torch.eye(B, dtype=torch.bool, device=z_fn.device)

        # sim_gap on val: how much positives outrank negatives
        sim_pos_vals = sim.diag()                          # [B]
        sim_neg_vals = sim[off_diag]                       # [B*(B-1)]
        self.log("val/sim_gap",      sim_pos_vals.mean() - sim_neg_vals.mean())
        self.log("val/sim_pos_std",  sim_pos_vals.std())
        self.log("val/sim_neg_std",  sim_neg_vals.std())

        # modality gap: cosine distance between embedding cloud centroids
        mu_e = F.normalize(z_en.mean(0, keepdim=True), dim=-1)
        mu_f = F.normalize(z_fn.mean(0, keepdim=True), dim=-1)
        self.log("val/modality_gap", 1.0 - (mu_e * mu_f).sum())

        # intra-modal similarity: how similar embeddings are within each modality
        sim_ee = z_en @ z_en.T
        sim_ff = z_fn @ z_fn.T
        self.log("val/intra_eeg_sim",  sim_ee[off_diag].mean())
        self.log("val/intra_fmri_sim", sim_ff[off_diag].mean())
        self.log("val/cross_modal_sim", sim.mean())

        # subject identity decoding — nearest-centroid probe on val embeddings.
        # If accuracy >> 1/n_subjects, the embeddings encode who, not what.
        if len(val_subs) == z_e.shape[0] and len(set(val_subs)) >= 2:
            unique_subs = sorted(set(val_subs))
            n_subs = len(unique_subs)
            sub2idx = {s: i for i, s in enumerate(unique_subs)}
            labels  = torch.tensor([sub2idx[s] for s in val_subs], device=z_en.device)

            def _nearest_centroid_acc(z):
                centroids = torch.zeros(n_subs, z.shape[1], device=z.device)
                for i in range(n_subs):
                    mask = labels == i
                    if mask.any():
                        centroids[i] = z[mask].mean(0)
                centroids = F.normalize(centroids, dim=-1)
                preds = (z @ centroids.T).argmax(dim=1)
                return (preds == labels).float().mean()

            self.log("val/sub_decode_eeg",  _nearest_centroid_acc(z_en))
            self.log("val/sub_decode_fmri", _nearest_centroid_acc(z_fn))
            self.log("val/sub_chance",      torch.tensor(1.0 / n_subs))
        
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
        def make_scheduler(opt):
            if self.config.train.scheduler == "warmup_cosine":
                warmup = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.1, end_factor=1.0,
                    total_iters=self.config.train.warmup_steps,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, self.config.train.max_steps - self.config.train.warmup_steps),
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    opt, schedulers=[warmup, cosine],
                    milestones=[self.config.train.warmup_steps],
                )
            elif self.config.train.scheduler == "const":
                return torch.optim.lr_scheduler.LinearLR(opt,
                start_factor=1,
                end_factor=1)

        return (
            [backbone_opt, projector_opt],
            [
                {"scheduler": make_scheduler(backbone_opt), "interval": "step"},
                {"scheduler": make_scheduler(projector_opt), "interval": "step"},
            ],
        )