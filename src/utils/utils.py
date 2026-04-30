import torch
import torch.nn.functional as F
import torch.nn as nn

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def _variance_loss(z, eps=1e-4):
    """Penalize embedding dimensions whose std across the batch is below 1."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()

def multi_positive_clip_loss(z_f: torch.Tensor, z_e: torch.Tensor, tau=0.07):
    """
    z_f: [T, K, D], fMRI embeddings
    z_e: [T, K, D], EEG embeddings
    """
    T, K, D = z_f.shape
    B = T * K
    
    # 1. Подготовка и нормализация
    z_f = F.normalize(z_f.reshape(B, D), dim=-1)
    z_e = F.normalize(z_e.reshape(B, D), dim=-1)

    # 2. Матрица сходства (fMRI -> EEG)
    # Строки - fMRI, Столбцы - EEG
    logits = (z_f @ z_e.T) / tau

    # 3. Создание блочно-диагональной маски (позитивы по времени)
    # Маска будет [B, B], где блоки KxK по диагонали - True
    mask_pos = torch.kron(
        torch.eye(T, device=logits.device), 
        torch.ones((K, K), device=logits.device)
    ).bool()

    # 4. Вспомогательная функция для расчета лосса в одну сторону
    def contrastive_step(l):
        # l: матрица логитов [B, B]
        # num: logsumexp только по позитивам в строке
        l_pos = l.masked_fill(~mask_pos, float('-inf'))
        num = torch.logsumexp(l_pos, dim=1)
        
        # den: logsumexp по всем элементам строки
        den = torch.logsumexp(l, dim=1)
        
        return -(num - den).mean()

    # 5. Симметричный лосс
    loss_f2e = contrastive_step(logits)           # Ищем правильные ЭЭГ для фМРТ
    loss_e2f = contrastive_step(logits.T)         # Ищем правильные фМРТ для ЭЭГ

    return 0.5 * (loss_f2e + loss_e2f)


def alignment_metric(z_f: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
    """
    Mean cosine distance between EEG and fMRI embeddings for the same time point.

    For each timestamp t, averages distances over all K subjects, then averages over T.
    Distance = 1 - cosine_similarity, so 0 means perfect alignment, 2 means opposite.

    z_f: [T, K, D] fMRI embeddings (raw, will be normalized inside)
    z_e: [T, K, D] EEG embeddings

    Returns scalar in [0, 2].
    """
    z_f = F.normalize(z_f, dim=-1)   # [T, K, D]
    z_e = F.normalize(z_e, dim=-1)

    cos_sim = (z_f * z_e).sum(dim=-1)   # [T, K]  — per-pair cosine similarity
    return (1.0 - cos_sim).mean()


def effective_rank(z: torch.Tensor) -> torch.Tensor:
    """
    Effective rank of an embedding matrix via the entropy of its singular value spectrum.
    Roy & Vetterli (2007): erank(Z) = exp(H),  H = -sum(p_i * log(p_i))
    where p_i = sigma_i / sum(sigma_i).

    Range: [1, D] — 1 means one dimension dominates, D means all dimensions used equally.

    z: [B, D]  batch of (optionally unnormalized) embeddings
    """
    with torch.no_grad():
        sigma = torch.linalg.svdvals(z.float())   # [min(B,D)]
        sigma = sigma[sigma > 0]
        p = sigma / sigma.sum()
        entropy = -(p * p.log()).sum()
    return entropy.exp()


def recall_at_k(z_f: torch.Tensor, z_e: torch.Tensor, k: int) -> dict:
    """
    Recall@k: fraction of queries where at least one positive is in the top-k retrieved.

    For each fMRI query, positives are the K EEG samples at the same timestamp (and vice-versa).
    Returns separate values for both directions.

    z_f, z_e: [T, K, D]
    Returns: {"f2e": scalar, "e2f": scalar}
    """
    T, K, D = z_f.shape
    B = T * K

    z_f = F.normalize(z_f.reshape(B, D), dim=-1)
    z_e = F.normalize(z_e.reshape(B, D), dim=-1)

    # Block-diagonal mask: True where query and key share a timestamp
    mask_pos = torch.kron(
        torch.eye(T, device=z_f.device),
        torch.ones(K, K, device=z_f.device),
    ).bool()                                                        # [B, B]

    def _recall(queries, keys):
        sim = queries @ keys.T                                      # [B, B]
        rank = sim.argsort(dim=1, descending=True).argsort(dim=1) + 1  # [B, B]
        hit = ((rank <= k) & mask_pos).any(dim=1).float()
        return hit.mean().item()

    return {"f2e": _recall(z_f, z_e), "e2f": _recall(z_e, z_f)}


def mean_reciprocal_rank(z_f: torch.Tensor, z_e: torch.Tensor) -> dict:
    """
    MRR: for each query, takes the reciprocal rank of the first (highest-ranked) positive,
    then averages over all queries.

    MRR = 1 means every query's top-1 result is a positive.
    MRR → 0 means positives are buried at the bottom of the ranking.

    z_f, z_e: [T, K, D]
    Returns: {"f2e": scalar, "e2f": scalar}
    """
    T, K, D = z_f.shape
    B = T * K

    z_f = F.normalize(z_f.reshape(B, D), dim=-1)
    z_e = F.normalize(z_e.reshape(B, D), dim=-1)

    mask_pos = torch.kron(
        torch.eye(T, device=z_f.device),
        torch.ones(K, K, device=z_f.device),
    ).bool()                                                            # [B, B]

    def _mrr(queries, keys):
        sim = queries @ keys.T                                          # [B, B]
        rank = sim.argsort(dim=1, descending=True).argsort(dim=1) + 1  # [B, B]
        # For each query, take the rank of its highest-ranked positive
        first_pos_rank = rank.masked_fill(~mask_pos, B + 1).min(dim=1).values
        return (1.0 / first_pos_rank.float()).mean().item()

    return {"f2e": _mrr(z_f, z_e), "e2f": _mrr(z_e, z_f)}