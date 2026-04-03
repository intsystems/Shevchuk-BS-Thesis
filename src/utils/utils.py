import torch
import torch.nn.functional as F
import torch.nn as nn

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def multi_positive_clip_loss(z_f, z_e, tau=0.07):
    """
    z_f: [B, D], fMRI embeddings
    z_e: [B, K, D], EEG embeddings (K positives per fMRI)
    """
    B, D = z_f.shape
    _, K, _ = z_e.shape

    z_f = F.normalize(z_f, dim=-1)
    z_e = F.normalize(z_e, dim=-1)

    z_e_flat = z_e.reshape(B*K, D)
    logits = (z_f @ z_e_flat.T) / tau

    mask_pos = torch.zeros((B, B*K), dtype=torch.bool, device=logits.device)
    row = torch.arange(B, device=logits.device).unsqueeze(1)
    cols = row * K + torch.arange(K, device=logits.device)
    mask_pos[row, cols] = True

    logits_pos = logits.masked_fill(~mask_pos, float('-inf'))
    num = torch.logsumexp(logits_pos, dim=1)

    den = torch.logsumexp(logits, dim=1)

    loss_f2e = -(num - den).mean()

    logits_e2f = (z_e_flat @ z_f.T) / tau
    target = torch.arange(B, device=logits.device).repeat_interleave(K)
    loss_e2f = F.cross_entropy(logits_e2f, target)

    return 0.5 * (loss_f2e + loss_e2f)

class SymmetricMultimodalTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        """
        Symmetric Batch-Hard Triplet Loss for Multimodal Contrastive Learning.
        Applies Cross-Modal and Intra-Modal negative mining.
        """
        super().__init__()
        self.margin = margin

    def forward(self, eeg_emb, fmri_emb):
        """
        Args:
            eeg_emb (torch.Tensor): [batch_size, embed_dim]
            fmri_emb (torch.Tensor): [batch_size, embed_dim]
        """
        batch_size = eeg_emb.size(0)
        
        # 1. Normalize embeddings (L2 norm)
        eeg_emb = F.normalize(eeg_emb, p=2, dim=1)
        fmri_emb = F.normalize(fmri_emb, p=2, dim=1)

        # 2. Compute Distances (1 - Cosine Similarity)
        # Positives (Diagonal of the cross-modal matrix)
        pos_dist = 1.0 - (eeg_emb * fmri_emb).sum(dim=1) 

        # Distance Matrices for Negative Mining
        dist_cross      = 1.0 - torch.matmul(eeg_emb, fmri_emb.t()) # [B, B]
        dist_intra_eeg  = 1.0 - torch.matmul(eeg_emb, eeg_emb.t())  # [B, B]
        dist_intra_fmri = 1.0 - torch.matmul(fmri_emb, fmri_emb.t())# [B, B]

        # 3. Create Identity Mask to ignore diagonals (positives / self-matches)
        mask = torch.eye(batch_size, dtype=torch.bool, device=eeg_emb.device)

        # Apply mask (fill diagonal with infinity so they are never picked as minimums)
        neg_cross_masked = dist_cross.masked_fill(mask, float('inf'))
        neg_intra_eeg_masked = dist_intra_eeg.masked_fill(mask, float('inf'))
        neg_intra_fmri_masked = dist_intra_fmri.masked_fill(mask, float('inf'))

        # 4. Mine the Hardest Negatives (Minimum distance)
        # Cross-Modal
        hard_neg_fmri_for_eeg = neg_cross_masked.min(dim=1)[0] # Row min
        hard_neg_eeg_for_fmri = neg_cross_masked.min(dim=0)[0] # Col min
        
        # Intra-Modal
        hard_neg_eeg_for_eeg = neg_intra_eeg_masked.min(dim=1)[0]
        hard_neg_fmri_for_fmri = neg_intra_fmri_masked.min(dim=1)[0]

        # 5. Compute the 4 Triplet Losses: L = max(0, d_pos - d_neg + margin)
        loss_cross_eeg  = F.relu(pos_dist - hard_neg_fmri_for_eeg + self.margin)
        loss_cross_fmri = F.relu(pos_dist - hard_neg_eeg_for_fmri + self.margin)
        loss_intra_eeg  = F.relu(pos_dist - hard_neg_eeg_for_eeg + self.margin)
        loss_intra_fmri = F.relu(pos_dist - hard_neg_fmri_for_fmri + self.margin)

        # 6. Average the losses
        total_loss = (
            loss_cross_eeg.mean() + 
            loss_cross_fmri.mean() + 
            loss_intra_eeg.mean() + 
            loss_intra_fmri.mean()
        ) / 4.0

        return total_loss