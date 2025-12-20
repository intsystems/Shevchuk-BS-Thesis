import torch
import torch.nn.functional as F

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
    logits = (z_f @ z_e_flat.T) / tau]

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
